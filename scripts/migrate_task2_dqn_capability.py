# ruff: noqa: E402
"""Migrate the frozen Task 1 DQN network into a versioned Task 2 artifact.

Supersedes the earlier byte-copy placeholder from issue #43: the successor's
checkpoint now has a different shape (21 inputs, 6 outputs) than the frozen
parent (8 inputs, 5 outputs), so it can no longer be a byte-identical copy.
Run once, locally; the result is what gets committed.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from agent_code.DagobertDuckDQN.persistence import (
    CHECKPOINT_PATH as PARENT_CHECKPOINT_PATH,
)
from agent_code.DagobertDuckDQN.persistence import load_evaluation_checkpoint
from agent_code.DagobertDuckDQNTask2.config import DEFAULT_CONFIG
from agent_code.DagobertDuckDQNTask2.migration import migrate_online_network
from agent_code.DagobertDuckDQNTask2.model import DQNLearner
from agent_code.DagobertDuckDQNTask2.persistence import (
    CHECKPOINT_PATH as SUCCESSOR_CHECKPOINT_PATH,
)
from agent_code.DagobertDuckDQNTask2.persistence import save_checkpoint
from agent_code.DagobertDuckDQNTask2.replay import ReplayBuffer

MIGRATION_SEED = 44
CORRECTED_CHECKPOINT_PATH = SUCCESSOR_CHECKPOINT_PATH.with_name(
    "checkpoint-issue85-zero-suffix.pt"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def main() -> None:
    parent = load_evaluation_checkpoint(PARENT_CHECKPOINT_PATH)
    migrated_network = migrate_online_network(
        parent.network,
        config=DEFAULT_CONFIG,
        seed=MIGRATION_SEED,
    )

    learner = DQNLearner(config=DEFAULT_CONFIG, seed=MIGRATION_SEED)
    learner.online_network.load_state_dict(migrated_network.state_dict())
    learner.target_network.load_state_dict(migrated_network.state_dict())

    empty_replay = ReplayBuffer(
        capacity=DEFAULT_CONFIG.replay_capacity,
        seed=MIGRATION_SEED,
    )
    fresh_action_rng = np.random.default_rng(MIGRATION_SEED)

    save_checkpoint(
        learner=learner,
        replay_buffer=empty_replay,
        action_rng=fresh_action_rng,
        epsilon=DEFAULT_CONFIG.initial_epsilon,
        completed_episodes=0,
        agent_seed=MIGRATION_SEED,
        path=CORRECTED_CHECKPOINT_PATH,
    )

    print(f"Parent checkpoint:    {PARENT_CHECKPOINT_PATH}")
    print(f"Parent SHA-256:       {sha256_file(PARENT_CHECKPOINT_PATH)}")
    print(f"Migrated checkpoint:  {CORRECTED_CHECKPOINT_PATH}")
    print(f"Migrated SHA-256:     {sha256_file(CORRECTED_CHECKPOINT_PATH)}")
    print(f"Migrated size:        {CORRECTED_CHECKPOINT_PATH.stat().st_size} bytes")


if __name__ == "__main__":
    main()
