"""Atomic and schema-validated evaluation artifact for DagobertDuckDQN.

DagobertDuckDQN is a frozen Task 1 baseline (issue #42): training is
rejected unconditionally (see train.py), so the only state this module ever
needs to read or write is what evaluation actually uses -- the online
network's weights, its architecture/configuration, and the schema and
provenance needed to validate them. It deliberately does not define a
resumable-training schema (target network, optimizer, replay buffer, RNG
state, epsilon, agent seed): keeping those fields, even reset to empty
values, would leave the committed artifact structurally a resumable
training checkpoint rather than the evaluation-only artifact issue #42
requires. The Task 2 successor (issue #43) forks its own persistence.py
with the full resumable schema restored, since it does train.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import sleep
from typing import Any

import torch

from .config import (
    ACTIONS,
    FEATURE_SCHEMA_VERSION,
    REWARDS,
    DQNConfig,
)
from .model import CPU_DEVICE, QNetwork, build_q_network

ARTIFACT_SCHEMA_VERSION = 1
CHECKPOINT_REPLACE_ATTEMPTS = 10
CHECKPOINT_REPLACE_RETRY_SECONDS = 0.1
MODEL_SCHEMA_VERSION = 1
CHECKPOINT_PATH = Path(__file__).resolve().parent / "checkpoint.pt"


@dataclass(frozen=True)
class LoadedEvaluationCheckpoint:
    """Frozen evaluation-only model and metadata."""

    config: DQNConfig
    network: QNetwork
    completed_episodes: int


def _replace_with_retry(source: Path, destination: Path) -> None:
    """Replace the artifact atomically, retrying transient Windows locks.

    Windows refuses os.replace while any process holds a handle on either
    file, which antivirus and search indexers do briefly and unpredictably.
    This artifact is written once by the freeze script rather than every
    episode, so the retry matters less here than it did for training -- kept
    for the same defensive reason regardless, mirroring the successor's
    per-episode save.
    """
    for attempt in range(CHECKPOINT_REPLACE_ATTEMPTS):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == CHECKPOINT_REPLACE_ATTEMPTS - 1:
                raise
            sleep(CHECKPOINT_REPLACE_RETRY_SECONDS * (attempt + 1))


def save_evaluation_artifact(
    *,
    network: QNetwork,
    config: DQNConfig,
    completed_episodes: int,
    path: Path = CHECKPOINT_PATH,
) -> Path:
    """Atomically save the online network and the metadata evaluation needs."""
    if type(completed_episodes) is not int or completed_episodes < 0:
        raise ValueError(
            "completed_episodes must be a non-negative integer."
        )

    payload = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "actions": list(ACTIONS),
        "rewards": dict(REWARDS),
        "config": asdict(config),
        "completed_episodes": completed_episodes,
        "network_state": network.state_dict(),
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


def load_evaluation_checkpoint(
    path: Path = CHECKPOINT_PATH,
) -> LoadedEvaluationCheckpoint:
    """Load the frozen online policy with restricted deserialization."""
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
        "artifact_schema_version",
        "model_schema_version",
        "feature_schema_version",
        "actions",
        "rewards",
        "config",
        "completed_episodes",
        "network_state",
    }

    if not isinstance(payload, dict) or set(payload) != required_fields:
        raise ValueError("Checkpoint has unexpected fields.")

    if payload["artifact_schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Artifact schema version mismatch.")

    if payload["model_schema_version"] != MODEL_SCHEMA_VERSION:
        raise ValueError("Model schema version mismatch.")

    if payload["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
        raise ValueError("Feature schema version mismatch.")

    if payload["actions"] != list(ACTIONS):
        raise ValueError("Checkpoint action order mismatch.")

    if payload["rewards"] != REWARDS:
        raise ValueError("Checkpoint reward mapping mismatch.")

    config = _restore_config(payload["config"])
    completed_episodes = payload["completed_episodes"]

    if type(completed_episodes) is not int or completed_episodes < 0:
        raise ValueError("Stored completed_episodes must be non-negative.")

    network = build_q_network(config, seed=0)

    try:
        network.load_state_dict(payload["network_state"], strict=True)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError(
            "Checkpoint network state is incompatible."
        ) from error

    network.eval()
    network.requires_grad_(False)

    return LoadedEvaluationCheckpoint(
        config=config,
        network=network,
        completed_episodes=completed_episodes,
    )


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
