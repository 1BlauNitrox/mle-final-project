"""anatomical persistence module for DerKleineVermoegensumverteiler"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np

from .config import ACTIONS, REWARDS
from .features import FEATURE_COUNT, FEATURE_SCHEMA_VERSION
from .model import QTable

MODEL_SCHEMA_VERSION = 1
MODEL_PATH = Path(__file__).resolve().parent / "model.npz"


@dataclass(frozen=True)
class LoadedModel:
    """A loaded model from disk."""

    q_table: QTable
    epsilon: float
    completed_episodes: int


def save_model(
    q_table: QTable,
    *,
    epsilon: float,
    completed_episodes: int,
    path: Path = MODEL_PATH,
) -> Path:
    """Save a Q-table model as non-pickle NumPy archive."""

    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("Epsilon must be in [0, 1].")

    if completed_episodes < 0:
        raise ValueError("Completed episodes must be non-negative.")

    states, q_values = _serialize_q_table(q_table)

    metadata = {
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_count": FEATURE_COUNT,
        "actions": list(ACTIONS),
        "learning_rate": q_table.learning_rate,
        "discount_factor": q_table.discount_factor,
        "epsilon": epsilon,
        "completed_episodes": completed_episodes,
        "rewards": REWARDS,
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None

    try:
        with NamedTemporaryFile(
            mode="w+b", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
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


def load_model(path: Path = MODEL_PATH) -> LoadedModel:
    """Load a Q-table model"""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Model file does not exist: {path}")

    try:
        with np.load(
            path,
            allow_pickle=False,
        ) as archive:
            required_entries = {
                "states",
                "q_values",
                "metadata",
            }

            if set(archive.files) != required_entries:
                raise ValueError("Model archive has unexpected entries")

            states = np.asarray(
                archive["states"],
                dtype=np.int64,
            ).copy()

            q_values = np.asarray(
                archive["q_values"],
                dtype=np.float64,
            ).copy()

            metadata_text = str(archive["metadata"].item())

        metadata = json.loads(metadata_text)

    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
    ) as error:
        raise ValueError(f"Could not load model archive: {path}") from error

    _validate_metadata(metadata)
    _validate_arrays(states, q_values)

    q_table = QTable(
        learning_rate=float(metadata["learning_rate"]),
        discount_factor=float(metadata["discount_factor"]),
    )

    for state_row, value_row in zip(
        states,
        q_values,
        strict=True,
    ):
        state = tuple(int(value) for value in state_row)

        if state in q_table.values:
            raise ValueError("Model contains a duplicate state")

        q_table.values[state] = value_row.copy()

    return LoadedModel(
        q_table=q_table,
        epsilon=float(metadata["epsilon"]),
        completed_episodes=int(metadata["completed_episodes"]),
    )


def _serialize_q_table(q_table: QTable) -> tuple[np.ndarray, np.ndarray]:
    """Serialize the Q-table into two NumPy arrays."""

    ordered_entries = sorted(q_table.values.items())

    if not ordered_entries:
        return np.empty((0, FEATURE_COUNT), dtype=np.int64), np.empty(
            (0, len(ACTIONS)), dtype=np.float64
        )

    states = np.asarray([state for state, _values in ordered_entries], dtype=np.int64)
    q_values = np.asarray([values for _state, values in ordered_entries], dtype=np.float64)

    _validate_arrays(states, q_values)

    return states, q_values


def _validate_arrays(
    states: np.ndarray,
    q_values: np.ndarray,
) -> None:
    """Validate serialized state and Q-value arrays."""
    if states.ndim != 2:
        raise ValueError("Model states must be a two-dimensional array")

    if q_values.ndim != 2:
        raise ValueError("Model Q-values must be a two-dimensional array")

    expected_state_shape = (
        states.shape[0],
        FEATURE_COUNT,
    )

    if states.shape != expected_state_shape:
        raise ValueError("Model states have an incompatible shape")

    expected_q_shape = (
        q_values.shape[0],
        len(ACTIONS),
    )

    if q_values.shape != expected_q_shape:
        raise ValueError("Model Q-values have an incompatible shape")

    if states.shape[0] != q_values.shape[0]:
        raise ValueError("State and Q-value row counts differ")

    if not np.all(np.isfinite(q_values)):
        raise ValueError("Model Q-values must all be finite")


def _validate_metadata(metadata: Any) -> None:
    """Validate the metadata."""

    if not isinstance(metadata, dict):
        raise ValueError("Model metadata must be a dictionary")

    required_fields = {
        "model_schema_version",
        "feature_schema_version",
        "feature_count",
        "actions",
        "learning_rate",
        "discount_factor",
        "epsilon",
        "completed_episodes",
        "rewards",
    }

    if set(metadata) != required_fields:
        raise ValueError("Model metadata has unexpected fields")

    if metadata["model_schema_version"] != MODEL_SCHEMA_VERSION:
        raise ValueError("Model schema version mismatch")

    if (
        metadata["feature_schema_version"] != FEATURE_SCHEMA_VERSION
        or metadata["feature_count"] != FEATURE_COUNT
    ):
        raise ValueError("Feature schema or count version mismatch")

    if metadata["actions"] != list(ACTIONS):
        raise ValueError("Actions mismatch")
    
    if metadata["rewards"] != REWARDS:
        raise ValueError("Reward configuration mismatch")

    epsilon = metadata["epsilon"]
    completed_episodes = metadata["completed_episodes"]

    if not isinstance(epsilon, (int, float)):
        raise ValueError("Stored epsilon must be a number")

    if not 0.0 <= float(epsilon) <= 1.0:
        raise ValueError("Stored epsilon must be in [0, 1]")

    if not isinstance(completed_episodes, int) or completed_episodes < 0:
        raise ValueError("Stored completed_episodes must be a non-negative integer")
