"""Load and validate the frozen Task 1 Q-table as a Task 2 prior."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PARENT_MODEL_PATH = Path(__file__).resolve().parent / "parent-model.npz"

EXPECTED_PARENT_SHA256 = "4e1da63a819ef8f51b112ffaf422ab251b853915375fe486538be8595b988307"

PARENT_MODEL_SCHEMA_VERSION = 2
PARENT_FEATURE_SCHEMA_VERSION = 1
PARENT_FEATURE_COUNT = 8

PARENT_ACTIONS = (
    "UP",
    "RIGHT",
    "DOWN",
    "LEFT",
    "WAIT",
)

PARENT_FEATURE_DOMAINS: tuple[frozenset[int], ...] = (
    frozenset({0, 1}),
    frozenset({0, 1}),
    frozenset({0, 1}),
    frozenset({0, 1}),
    frozenset({0, 1}),
    frozenset({-1, 0, 1}),
    frozenset({-1, 0, 1}),
    frozenset({0, 1, 2, 3}),
)

Task1State = tuple[int, ...]


@dataclass(frozen=True)
class ParentModelPrior:
    """Validated values and training state from the frozen parent."""

    values: dict[Task1State, np.ndarray]
    epsilon: float
    completed_episodes: int
    sha256: str


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 checksum of a file."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def load_parent_prior(
    path: Path = PARENT_MODEL_PATH,
    *,
    expected_sha256: str = EXPECTED_PARENT_SHA256,
) -> ParentModelPrior:
    """Load a checksum- and schema-validated Task 1 Q-table."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Parent model does not exist: {path}")

    actual_sha256 = sha256_file(path)

    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Parent model checksum mismatch: expected {expected_sha256}, got {actual_sha256}"
        )

    try:
        with np.load(path, allow_pickle=False) as archive:
            required_entries = {
                "states",
                "q_values",
                "metadata",
            }

            if set(archive.files) != required_entries:
                raise ValueError("Parent archive has unexpected entries")

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
        raise ValueError(f"Could not load parent model archive: {path}") from error

    _validate_parent_metadata(metadata)
    _validate_parent_arrays(raw_states, raw_q_values)

    states = raw_states.astype(np.int64, copy=True)
    q_values = raw_q_values.astype(np.float64, copy=True)

    values: dict[Task1State, np.ndarray] = {}

    for state_row, value_row in zip(
        states,
        q_values,
        strict=True,
    ):
        state = tuple(int(value) for value in state_row)

        if state in values:
            raise ValueError("Parent model contains a duplicate state")

        values[state] = value_row.copy()

    return ParentModelPrior(
        values=values,
        epsilon=float(metadata["epsilon"]),
        completed_episodes=int(metadata["completed_episodes"]),
        sha256=actual_sha256,
    )


def _validate_parent_metadata(metadata: Any) -> None:
    """Validate the parent metadata required for migration."""

    if not isinstance(metadata, dict):
        raise ValueError("Parent metadata must be a dictionary")

    if metadata.get("model_schema_version") != (PARENT_MODEL_SCHEMA_VERSION):
        raise ValueError("Parent model schema version mismatch")

    if metadata.get("feature_schema_version") != (PARENT_FEATURE_SCHEMA_VERSION):
        raise ValueError("Parent feature schema version mismatch")

    if metadata.get("feature_count") != PARENT_FEATURE_COUNT:
        raise ValueError("Parent feature count mismatch")

    if metadata.get("actions") != list(PARENT_ACTIONS):
        raise ValueError("Parent action order mismatch")

    epsilon = metadata.get("epsilon")
    completed_episodes = metadata.get("completed_episodes")

    if isinstance(epsilon, bool) or not isinstance(
        epsilon,
        (int, float),
    ):
        raise ValueError("Parent epsilon must be numeric")

    if not 0.0 <= float(epsilon) <= 1.0:
        raise ValueError("Parent epsilon must be in [0, 1]")

    if type(completed_episodes) is not int or completed_episodes < 0:
        raise ValueError("Parent completed_episodes must be a non-negative integer")


def _validate_parent_arrays(
    states: np.ndarray,
    q_values: np.ndarray,
) -> None:
    """Validate parent state and Q-value arrays."""

    if states.ndim != 2:
        raise ValueError("Parent states must be two-dimensional")

    if states.shape[1] != PARENT_FEATURE_COUNT:
        raise ValueError("Parent states have an incompatible shape")

    if q_values.ndim != 2:
        raise ValueError("Parent Q-values must be two-dimensional")

    expected_q_shape = (
        states.shape[0],
        len(PARENT_ACTIONS),
    )

    if q_values.shape != expected_q_shape:
        raise ValueError("Parent Q-values have an incompatible shape")

    if not np.issubdtype(states.dtype, np.number):
        raise ValueError("Parent states must be numeric")

    if not np.all(np.isfinite(states)):
        raise ValueError("Parent states must be finite")

    if not np.all(states == np.floor(states)):
        raise ValueError("Parent states must contain integral values")

    if not np.all(np.isfinite(q_values)):
        raise ValueError("Parent Q-values must be finite")

    integer_states = states.astype(np.int64)

    for index, domain in enumerate(PARENT_FEATURE_DOMAINS):
        present_values = set(integer_states[:, index])

        if not present_values.issubset(domain):
            raise ValueError(f"Parent feature {index} contains invalid values")
