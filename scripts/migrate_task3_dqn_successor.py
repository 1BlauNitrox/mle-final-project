"""Build the provisional Task 3 DQN checkpoint from corrected Task 2."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from agent_code.DagobertDuckDQNTask2.persistence import (  # noqa: E402
    CHECKPOINT_PATH as TASK2_CHECKPOINT_PATH,
)
from agent_code.DagobertDuckDQNTask2.persistence import (  # noqa: E402
    load_evaluation_checkpoint,
)
from agent_code.DagobertDuckDQNTask3.config import DEFAULT_CONFIG  # noqa: E402
from agent_code.DagobertDuckDQNTask3.migration import (  # noqa: E402
    MIGRATION_INIT_SEED,
    migrate_online_network,
)
from agent_code.DagobertDuckDQNTask3.model import DQNLearner  # noqa: E402
from agent_code.DagobertDuckDQNTask3.persistence import (  # noqa: E402
    CHECKPOINT_PATH as SUCCESSOR_CHECKPOINT_PATH,
)
from agent_code.DagobertDuckDQNTask3.persistence import (  # noqa: E402
    save_checkpoint,
)
from agent_code.DagobertDuckDQNTask3.replay import ReplayBuffer  # noqa: E402

PARENT_CHECKPOINT_PATH = TASK2_CHECKPOINT_PATH.with_name(
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
        seed=MIGRATION_INIT_SEED,
    )

    learner = DQNLearner(config=DEFAULT_CONFIG, seed=MIGRATION_INIT_SEED)
    learner.online_network.load_state_dict(migrated_network.state_dict())
    learner.target_network.load_state_dict(migrated_network.state_dict())

    save_checkpoint(
        learner=learner,
        replay_buffer=ReplayBuffer(
            capacity=DEFAULT_CONFIG.replay_capacity,
            seed=MIGRATION_INIT_SEED,
        ),
        action_rng=np.random.default_rng(MIGRATION_INIT_SEED),
        epsilon=DEFAULT_CONFIG.initial_epsilon,
        completed_episodes=0,
        agent_seed=MIGRATION_INIT_SEED,
        path=SUCCESSOR_CHECKPOINT_PATH,
    )

    print(f"Parent checkpoint:    {PARENT_CHECKPOINT_PATH}")
    print(f"Parent SHA-256:       {sha256_file(PARENT_CHECKPOINT_PATH)}")
    print(f"Migrated checkpoint:  {SUCCESSOR_CHECKPOINT_PATH}")
    print(f"Migrated SHA-256:      {sha256_file(SUCCESSOR_CHECKPOINT_PATH)}")
    print(f"Migrated size:         {SUCCESSOR_CHECKPOINT_PATH.stat().st_size} bytes")


if __name__ == "__main__":
    main()
