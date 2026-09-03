"""Atomic and schema-validated checkpoints for DagobertDuckDQN."""

from __future__ import annotations

import os
import pickle
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import sleep
from typing import Any

import numpy as np
import torch

from .config import (
    ACTIONS,
    FEATURE_SCHEMA_VERSION,
    REWARDS,
    DQNConfig,
)
from .model import (
    CPU_DEVICE,
    DQNLearner,
    QNetwork,
    build_q_network,
)
from .replay import ReplayBuffer

CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_REPLACE_ATTEMPTS = 10
CHECKPOINT_REPLACE_RETRY_SECONDS = 0.1
MODEL_SCHEMA_VERSION = 1
CHECKPOINT_PATH = Path(__file__).resolve().parent / "checkpoint.pt"


@dataclass(frozen=True)
class LoadedTrainingCheckpoint:
    """Complete state required to resume training."""

    config: DQNConfig
    learner: DQNLearner
    replay_buffer: ReplayBuffer
    action_rng: np.random.Generator
    epsilon: float
    completed_episodes: int
    agent_seed: int


@dataclass(frozen=True)
class LoadedEvaluationCheckpoint:
    """Frozen evaluation-only model and metadata."""

    config: DQNConfig
    network: QNetwork
    completed_episodes: int


def _replace_with_retry(source: Path, destination: Path) -> None:
    """Replace the checkpoint atomically, retrying transient Windows locks.

    Windows refuses os.replace while any process holds a handle on either file,
    which antivirus and search indexers do briefly and unpredictably. This runs
    after every episode, so one unlucky moment out of tens of thousands of
    attempts ends a multi-hour training run: both the issue #41 and the issue
    #58 series lost their fifth run to exactly this. POSIX rename is already
    atomic and never raises here, so the retry is inert off Windows.
    """
    for attempt in range(CHECKPOINT_REPLACE_ATTEMPTS):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == CHECKPOINT_REPLACE_ATTEMPTS - 1:
                raise
            sleep(CHECKPOINT_REPLACE_RETRY_SECONDS * (attempt + 1))


def save_checkpoint(
    *,
    learner: DQNLearner,
    replay_buffer: ReplayBuffer,
    action_rng: np.random.Generator,
    epsilon: float,
    completed_episodes: int,
    agent_seed: int,
    path: Path = CHECKPOINT_PATH,
) -> Path:
    """Atomically save all state required to resume training."""
    if replay_buffer.capacity != learner.config.replay_capacity:
        raise ValueError(
            "Replay capacity does not match learner configuration."
        )

    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1].")

    if type(completed_episodes) is not int or completed_episodes < 0:
        raise ValueError(
            "completed_episodes must be a non-negative integer."
        )

    if type(agent_seed) is not int:
        raise ValueError("agent_seed must be an integer.")

    payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "actions": list(ACTIONS),
        "rewards": dict(REWARDS),
        "config": asdict(learner.config),
        "agent_seed": agent_seed,
        "epsilon": float(epsilon),
        "completed_episodes": completed_episodes,
        "learner_state": learner.state_dict(),
        "replay_state": _serialize_replay(replay_buffer),
        "action_rng_state": deepcopy(
            action_rng.bit_generator.state
        ),
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with NamedTemporaryFile(
            mode="w+b",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            torch.save(payload, temporary_file)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        _replace_with_retry(temporary_path, path)
    except Exception:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise

    return path


def load_training_checkpoint(
    path: Path = CHECKPOINT_PATH,
) -> LoadedTrainingCheckpoint:
    """Load and validate a complete resumable training checkpoint."""
    payload, config = _load_payload(path)
    agent_seed = int(payload["agent_seed"])

    learner = DQNLearner(
        config=config,
        seed=agent_seed,
    )
    learner.load_state_dict(payload["learner_state"])

    replay_buffer = _restore_replay(
        payload["replay_state"],
        config=config,
        seed=agent_seed,
    )
    action_rng = _restore_numpy_rng(payload["action_rng_state"])

    return LoadedTrainingCheckpoint(
        config=config,
        learner=learner,
        replay_buffer=replay_buffer,
        action_rng=action_rng,
        epsilon=float(payload["epsilon"]),
        completed_episodes=int(payload["completed_episodes"]),
        agent_seed=agent_seed,
    )


def load_evaluation_checkpoint(
    path: Path = CHECKPOINT_PATH,
) -> LoadedEvaluationCheckpoint:
    """Load only the frozen online policy required for evaluation."""
    payload, config = _load_payload(path)
    learner_state = payload["learner_state"]

    if not isinstance(learner_state, dict):
        raise ValueError("Checkpoint learner state must be a dictionary.")

    required_learner_fields = {
        "online_network",
        "target_network",
        "optimizer",
        "update_steps",
    }

    if set(learner_state) != required_learner_fields:
        raise ValueError("Checkpoint learner state has unexpected fields.")

    network = build_q_network(
        config,
        seed=int(payload["agent_seed"]),
    )

    try:
        network.load_state_dict(
            learner_state["online_network"],
            strict=True,
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError(
            "Checkpoint online-network state is incompatible."
        ) from error

    network.eval()
    network.requires_grad_(False)

    return LoadedEvaluationCheckpoint(
        config=config,
        network=network,
        completed_episodes=int(payload["completed_episodes"]),
    )


def _load_payload(
    path: Path,
) -> tuple[dict[str, Any], DQNConfig]:
    """Read a checkpoint with restricted PyTorch deserialization."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")

    try:
        payload = torch.load(
            path,
            map_location=CPU_DEVICE,
            weights_only=True,
        )
    except (
        EOFError,
        OSError,
        pickle.UnpicklingError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            f"Could not load checkpoint: {path}"
        ) from error

    required_fields = {
        "checkpoint_schema_version",
        "model_schema_version",
        "feature_schema_version",
        "actions",
        "rewards",
        "config",
        "agent_seed",
        "epsilon",
        "completed_episodes",
        "learner_state",
        "replay_state",
        "action_rng_state",
    }

    if not isinstance(payload, dict) or set(payload) != required_fields:
        raise ValueError("Checkpoint has unexpected fields.")

    if payload["checkpoint_schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Checkpoint schema version mismatch.")

    if payload["model_schema_version"] != MODEL_SCHEMA_VERSION:
        raise ValueError("Model schema version mismatch.")

    if payload["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
        raise ValueError("Feature schema version mismatch.")

    if payload["actions"] != list(ACTIONS):
        raise ValueError("Checkpoint action order mismatch.")

    if payload["rewards"] != REWARDS:
        raise ValueError("Checkpoint reward mapping mismatch.")

    config = _restore_config(payload["config"])

    agent_seed = payload["agent_seed"]
    epsilon = payload["epsilon"]
    completed_episodes = payload["completed_episodes"]

    if type(agent_seed) is not int:
        raise ValueError("Stored agent_seed must be an integer.")

    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float))
        or not np.isfinite(float(epsilon))
        or not 0.0 <= float(epsilon) <= 1.0
    ):
        raise ValueError("Stored epsilon must be finite and in [0, 1].")

    if (
        type(completed_episodes) is not int
        or completed_episodes < 0
    ):
        raise ValueError(
            "Stored completed_episodes must be non-negative."
        )

    return payload, config


def _restore_config(value: Any) -> DQNConfig:
    """Validate and reconstruct the stored DQN configuration."""
    if not isinstance(value, dict):
        raise ValueError("Stored configuration must be a dictionary.")

    expected_defaults = asdict(DQNConfig())

    if set(value) != set(expected_defaults):
        raise ValueError("Stored configuration has unexpected fields.")

    for name, default_value in expected_defaults.items():
        stored_value = value[name]

        if isinstance(default_value, tuple):
            if (
                not isinstance(stored_value, tuple)
                or any(type(item) is not int for item in stored_value)
            ):
                raise ValueError(
                    f"Stored configuration field {name} has invalid type."
                )
        elif type(stored_value) is not type(default_value):
            raise ValueError(
                f"Stored configuration field {name} has invalid type."
            )

    try:
        return DQNConfig(**value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Stored DQN configuration is invalid."
        ) from error


def _serialize_replay(
    replay_buffer: ReplayBuffer,
) -> dict[str, Any]:
    """Convert replay arrays to restricted-load-compatible tensors."""
    state = replay_buffer.state_dict()

    return {
        "capacity": state["capacity"],
        "states": torch.from_numpy(state["states"]).clone(),
        "action_indices": torch.from_numpy(
            state["action_indices"]
        ).clone(),
        "rewards": torch.from_numpy(state["rewards"]).clone(),
        "next_states": torch.from_numpy(
            state["next_states"]
        ).clone(),
        "terminals": torch.from_numpy(
            state["terminals"]
        ).clone(),
        "rng_state": deepcopy(state["rng_state"]),
    }


def _restore_replay(
    value: Any,
    *,
    config: DQNConfig,
    seed: int,
) -> ReplayBuffer:
    """Restore replay arrays and sampling RNG."""
    if not isinstance(value, dict):
        raise ValueError("Stored replay state must be a dictionary.")

    required_fields = {
        "capacity",
        "states",
        "action_indices",
        "rewards",
        "next_states",
        "terminals",
        "rng_state",
    }

    if set(value) != required_fields:
        raise ValueError("Stored replay state has unexpected fields.")

    replay_state = {
        "capacity": value["capacity"],
        "states": _tensor_to_numpy(
            value["states"],
            dtype=torch.float32,
            name="states",
        ),
        "action_indices": _tensor_to_numpy(
            value["action_indices"],
            dtype=torch.int64,
            name="action_indices",
        ),
        "rewards": _tensor_to_numpy(
            value["rewards"],
            dtype=torch.float32,
            name="rewards",
        ),
        "next_states": _tensor_to_numpy(
            value["next_states"],
            dtype=torch.float32,
            name="next_states",
        ),
        "terminals": _tensor_to_numpy(
            value["terminals"],
            dtype=torch.bool,
            name="terminals",
        ),
        "rng_state": deepcopy(value["rng_state"]),
    }

    replay_buffer = ReplayBuffer(
        capacity=config.replay_capacity,
        seed=seed,
    )
    replay_buffer.load_state_dict(replay_state)

    return replay_buffer


def _tensor_to_numpy(
    value: Any,
    *,
    dtype: torch.dtype,
    name: str,
) -> np.ndarray:
    """Validate a CPU tensor and return an independent NumPy array."""
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Stored replay {name} must be a tensor.")

    if value.dtype != dtype:
        raise ValueError(
            f"Stored replay {name} has an incompatible dtype."
        )

    if value.device.type != "cpu":
        raise ValueError(f"Stored replay {name} must be on CPU.")

    return value.detach().numpy().copy()


def _restore_numpy_rng(value: Any) -> np.random.Generator:
    """Restore a validated NumPy generator state."""
    if not isinstance(value, dict):
        raise ValueError("Stored RNG state must be a dictionary.")

    rng = np.random.default_rng()

    try:
        rng.bit_generator.state = deepcopy(value)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Stored RNG state is invalid.") from error

    return rng