"""Load the checksum-pinned frozen tabular Task 2 model as a Task 3 prior."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PARENT_MODEL_PATH = Path(__file__).resolve().parent / "parent-model.npz"
EXPECTED_PARENT_SHA256 = "93470f92082597b1848f2fa65265c7288dbb1b50ac90370e5747e085e775ea3e"
PARENT_FEATURE_COUNT = 5
PARENT_ACTION_COUNT = 6


@dataclass(frozen=True)
class ParentModelPrior:
    """Validated frozen Task 2 values and identity."""

    values: dict[tuple[int, ...], np.ndarray]
    epsilon: float
    completed_episodes: int
    sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_parent_prior(
    path: Path = PARENT_MODEL_PATH,
    *,
    expected_sha256: str = EXPECTED_PARENT_SHA256,
) -> ParentModelPrior:
    """Return the frozen compact Task 2 table after strict validation."""
    path = Path(path)
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Parent model checksum mismatch: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )
    try:
        with np.load(path, allow_pickle=False) as archive:
            expected = {"states", "q_values", "metadata", "visit_counts"}
            if set(archive.files) != expected:
                raise ValueError("Parent archive has unexpected entries")
            states = np.asarray(archive["states"], dtype=np.int64)
            q_values = np.asarray(archive["q_values"], dtype=np.float64)
            metadata = json.loads(str(archive["metadata"].item()))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not load parent model archive: {path}") from error

    if metadata.get("model_schema_version") != 6:
        raise ValueError("Parent model schema version mismatch")
    if metadata.get("state_representation") != "compact_decision":
        raise ValueError("Parent state representation mismatch")
    if metadata.get("feature_count") != PARENT_FEATURE_COUNT:
        raise ValueError("Parent feature count mismatch")
    if states.ndim != 2 or states.shape[1] != PARENT_FEATURE_COUNT:
        raise ValueError("Parent states have an incompatible shape")
    if q_values.shape != (states.shape[0], PARENT_ACTION_COUNT):
        raise ValueError("Parent Q-values have an incompatible shape")

    values = {
        tuple(int(value) for value in state): row.copy()
        for state, row in zip(states, q_values, strict=True)
    }
    if len(values) != len(states):
        raise ValueError("Parent model contains duplicate states")
    return ParentModelPrior(
        values=values,
        epsilon=float(metadata["epsilon"]),
        completed_episodes=int(metadata["completed_episodes"]),
        sha256=actual_sha256,
    )
