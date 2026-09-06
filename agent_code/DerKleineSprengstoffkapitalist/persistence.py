"""Schema-validated persistence for DerKleineSprengstoffkapitalist."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np

from .config import ACTIONS, EPSILON_DECAY, MINIMUM_EPSILON, REWARDS
from .features import (
    FEATURE_COUNT,
    FEATURE_SCHEMA_VERSION,
    StateFeatures,
    validate_features,
)
from .migration import (
    EXPECTED_PARENT_SHA256,
    PARENT_MODEL_PATH,
    load_parent_prior,
)
from .model import BOMB_PRIOR_MARGIN, QTable

MODEL_SCHEMA_VERSION = 3
MODEL_PATH = Path(__file__).resolve().parent / "model.npz"


@dataclass(frozen=True)
class LoadedModel:
    """A loaded Task 2 model and its training state."""

    q_table: QTable
    epsilon: float
    completed_episodes: int
    parent_model_sha256: str


def save_model(
    q_table: QTable,
    *,
    epsilon: float,
    completed_episodes: int,
    path: Path = MODEL_PATH,
    parent_path: Path = PARENT_MODEL_PATH,
) -> Path:
    """Save a sparse Task 2 Q-table as an atomic non-pickle archive."""

    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("Epsilon must be in [0, 1].")

    if completed_episodes < 0:
        raise ValueError("Completed episodes must be non-negative.")

    parent_prior = load_parent_prior(parent_path)
    states, q_values = _serialize_q_table(q_table)

    metadata = {
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_count": FEATURE_COUNT,
        "actions": list(ACTIONS),
        "learning_rate": q_table.learning_rate,
        "discount_factor": q_table.discount_factor,
        "epsilon": epsilon,
        "epsilon_decay": EPSILON_DECAY,
        "minimum_epsilon": MINIMUM_EPSILON,
        "completed_episodes": completed_episodes,
        "rewards": REWARDS,
        "parent_model_sha256": parent_prior.sha256,
        "bomb_prior_margin": BOMB_PRIOR_MARGIN,
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

            np.savez_compressed(
                temporary_file,
                states=states,
                q_values=q_values,
                metadata=np.array(json.dumps(metadata, sort_keys=True)),
            )

            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        os.replace(temporary_path, path)

    except Exception:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

        raise

    return path


def load_model(
    path: Path = MODEL_PATH,
    *,
    parent_path: Path = PARENT_MODEL_PATH,
) -> LoadedModel:
    """Load a Task 2 model and attach its validated Task 1 prior."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Model file does not exist: {path}")

    try:
        with np.load(path, allow_pickle=False) as archive:
            required_entries = {
                "states",
                "q_values",
                "metadata",
            }

            if set(archive.files) != required_entries:
                raise ValueError("Model archive has unexpected entries")

            raw_states = np.asarray(archive["states"]).copy()

            raw_q_values = np.asarray(archive["q_values"]).copy()

            metadata_text = str(archive["metadata"].item())

        metadata = json.loads(metadata_text)

    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        json.JSONDecodeError,
    ) as error:
        raise ValueError(f"Could not load model archive: {path}") from error

    _validate_metadata(metadata)
    _validate_raw_arrays(raw_states, raw_q_values)

    parent_prior = load_parent_prior(
        parent_path,
        expected_sha256=metadata["parent_model_sha256"],
    )

    states = raw_states.astype(np.int64, copy=True)
    q_values = raw_q_values.astype(np.float64, copy=True)

    q_table = QTable(
        learning_rate=float(metadata["learning_rate"]),
        discount_factor=float(metadata["discount_factor"]),
        parent_values=parent_prior.values,
    )

    for state_row, value_row in zip(
        states,
        q_values,
        strict=True,
    ):
        state: StateFeatures = tuple(int(value) for value in state_row)

        if state in q_table.values:
            raise ValueError("Model contains a duplicate Task 2 state")

        q_table.values[state] = value_row.copy()

    return LoadedModel(
        q_table=q_table,
        epsilon=float(metadata["epsilon"]),
        completed_episodes=int(metadata["completed_episodes"]),
        parent_model_sha256=parent_prior.sha256,
    )


def _serialize_q_table(
    q_table: QTable,
) -> tuple[np.ndarray, np.ndarray]:
    """Serialize only materialized Task 2 states."""

    ordered_entries = sorted(q_table.values.items())

    if not ordered_entries:
        return (
            np.empty(
                (0, FEATURE_COUNT),
                dtype=np.int64,
            ),
            np.empty(
                (0, len(ACTIONS)),
                dtype=np.float64,
            ),
        )

    states: list[StateFeatures] = []
    q_values: list[np.ndarray] = []

    for state, values in ordered_entries:
        validate_features(state)

        value_array = np.asarray(values, dtype=np.float64)

        if value_array.shape != (len(ACTIONS),):
            raise ValueError("Q-values have an incompatible action count")

        if not np.all(np.isfinite(value_array)):
            raise ValueError("Q-values must be finite")

        states.append(state)
        q_values.append(value_array)

    return (
        np.asarray(states, dtype=np.int64),
        np.asarray(q_values, dtype=np.float64),
    )


def _validate_raw_arrays(
    states: np.ndarray,
    q_values: np.ndarray,
) -> None:
    """Validate arrays before converting their dtypes."""

    if states.ndim != 2:
        raise ValueError("Model states must be two-dimensional")

    if states.shape[1] != FEATURE_COUNT:
        raise ValueError("Model states have an incompatible feature count")

    if q_values.ndim != 2:
        raise ValueError("Model Q-values must be two-dimensional")

    expected_q_shape = (
        states.shape[0],
        len(ACTIONS),
    )

    if q_values.shape != expected_q_shape:
        raise ValueError("Model Q-values have an incompatible shape")

    if not np.issubdtype(states.dtype, np.number):
        raise ValueError("Model states must be numeric")

    if not np.all(np.isfinite(states)):
        raise ValueError("Model states must be finite")

    if not np.all(states == np.floor(states)):
        raise ValueError("Model states must contain integral values")

    if not np.all(np.isfinite(q_values)):
        raise ValueError("Model Q-values must be finite")

    integer_states = states.astype(np.int64)

    for state_row in integer_states:
        state: StateFeatures = tuple(int(value) for value in state_row)
        validate_features(state)


def _validate_metadata(metadata: Any) -> None:
    """Validate the complete Task 2 model metadata."""

    if not isinstance(metadata, dict):
        raise ValueError("Model metadata must be a dictionary")

    if metadata["model_schema_version"] != MODEL_SCHEMA_VERSION:
        raise ValueError("Model schema version mismatch")

    required_fields = {
        "model_schema_version",
        "feature_schema_version",
        "feature_count",
        "actions",
        "learning_rate",
        "discount_factor",
        "epsilon",
        "epsilon_decay",
        "minimum_epsilon",
        "completed_episodes",
        "rewards",
        "parent_model_sha256",
        "bomb_prior_margin",
    }

    if set(metadata) != required_fields:
        raise ValueError("Model metadata has unexpected fields")

    if metadata["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
        raise ValueError("Feature schema version mismatch")

    if metadata["feature_count"] != FEATURE_COUNT:
        raise ValueError("Feature count mismatch")

    if metadata["actions"] != list(ACTIONS):
        raise ValueError("Action order mismatch")

    if metadata["rewards"] != REWARDS:
        raise ValueError("Reward configuration mismatch")

    if metadata["parent_model_sha256"] != EXPECTED_PARENT_SHA256:
        raise ValueError("Parent model checksum metadata mismatch")

    if metadata["bomb_prior_margin"] != BOMB_PRIOR_MARGIN:
        raise ValueError("Bomb prior margin mismatch")

    learning_rate = metadata["learning_rate"]
    discount_factor = metadata["discount_factor"]
    epsilon = metadata["epsilon"]
    completed_episodes = metadata["completed_episodes"]

    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not 0.0 < float(learning_rate) <= 1.0
    ):
        raise ValueError("Stored learning rate is invalid")

    if (
        isinstance(discount_factor, bool)
        or not isinstance(
            discount_factor,
            (int, float),
        )
        or not 0.0 <= float(discount_factor) <= 1.0
    ):
        raise ValueError("Stored discount factor is invalid")

    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float))
        or not 0.0 <= float(epsilon) <= 1.0
    ):
        raise ValueError("Stored epsilon is invalid")

    if type(completed_episodes) is not int or completed_episodes < 0:
        raise ValueError("Stored completed_episodes is invalid")

    if metadata["epsilon_decay"] != EPSILON_DECAY:
        raise ValueError("Epsilon decay mismatch")

    if metadata["minimum_epsilon"] != MINIMUM_EPSILON:
        raise ValueError("Minimum epsilon mismatch")
