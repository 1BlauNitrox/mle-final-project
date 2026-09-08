# ruff: noqa: E402
"""Build the Issue #87 checkpoint from the corrected Issue #85 artifact.

The source is intentionally the versioned zero-suffix migration, never the
historical ``checkpoint.pt``.  The destination uses the common 26-input
architecture for both campaign treatments; the five new input columns are
zero-initialized so the feature-off control starts with the inherited Task 2
function unchanged.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from agent_code.DagobertDuckDQNTask2.config import DEFAULT_CONFIG
from agent_code.DagobertDuckDQNTask2.migration import (
    migrate_escape_continuation_network,
)
from agent_code.DagobertDuckDQNTask2.model import DQNLearner
from agent_code.DagobertDuckDQNTask2.persistence import (
    CHECKPOINT_PATH,
    load_evaluation_checkpoint,
    save_checkpoint,
)
from agent_code.DagobertDuckDQNTask2.replay import ReplayBuffer

SOURCE_CHECKPOINT_PATH = CHECKPOINT_PATH.with_name(
    "checkpoint-issue85-zero-suffix.pt"
)
MIGRATION_SEED = 87


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    source = load_evaluation_checkpoint(SOURCE_CHECKPOINT_PATH)
    migrated_network = migrate_escape_continuation_network(
        source.network,
        config=DEFAULT_CONFIG,
        seed=MIGRATION_SEED,
    )

    learner = DQNLearner(config=DEFAULT_CONFIG, seed=MIGRATION_SEED)
    learner.online_network.load_state_dict(migrated_network.state_dict())
    learner.target_network.load_state_dict(migrated_network.state_dict())

    save_checkpoint(
        learner=learner,
        replay_buffer=ReplayBuffer(
            capacity=DEFAULT_CONFIG.replay_capacity,
            seed=MIGRATION_SEED,
        ),
        action_rng=np.random.default_rng(MIGRATION_SEED),
        epsilon=DEFAULT_CONFIG.initial_epsilon,
        completed_episodes=0,
        agent_seed=MIGRATION_SEED,
        path=CHECKPOINT_PATH,
    )

    print(f"Source checkpoint:    {SOURCE_CHECKPOINT_PATH}")
    print(f"Source SHA-256:       {sha256_file(SOURCE_CHECKPOINT_PATH)}")
    print(f"Migrated checkpoint:  {CHECKPOINT_PATH}")
    print(f"Migrated SHA-256:     {sha256_file(CHECKPOINT_PATH)}")
    print(f"Migrated size:        {CHECKPOINT_PATH.stat().st_size} bytes")


if __name__ == "__main__":
    main()
