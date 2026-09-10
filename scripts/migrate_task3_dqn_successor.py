"""Build the provisional Task 3 DQN checkpoint from corrected Task 2."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from agent_code.DagobertDuckDQNTask2.model import (  # noqa: E402
    build_q_network as build_parent_network,
)
from agent_code.DagobertDuckDQNTask2.persistence import (  # noqa: E402
    CHECKPOINT_PATH as TASK2_CHECKPOINT_PATH,
)
from agent_code.DagobertDuckDQNTask2.persistence import (  # noqa: E402
    load_evaluation_checkpoint,
)
from agent_code.DagobertDuckDQNTask3.migration import (  # noqa: E402
    MIGRATION_INIT_SEED,
    migrate_online_network,
    successor_config,
)
from agent_code.DagobertDuckDQNTask3.model import DQNLearner  # noqa: E402
from agent_code.DagobertDuckDQNTask3.persistence import (  # noqa: E402
    CHECKPOINT_PATH as SUCCESSOR_CHECKPOINT_PATH,
)
from agent_code.DagobertDuckDQNTask3.persistence import (  # noqa: E402
    save_checkpoint,
)
from agent_code.DagobertDuckDQNTask3.replay import ReplayBuffer  # noqa: E402

PARENT_CHECKPOINT_PATH = TASK2_CHECKPOINT_PATH.with_name("checkpoint-issue85-zero-suffix.pt")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=PARENT_CHECKPOINT_PATH)
    parser.add_argument("--output", type=Path, default=SUCCESSOR_CHECKPOINT_PATH)
    parser.add_argument("--parent-sha256", required=True)
    args = parser.parse_args()
    if args.parent.resolve() == args.output.resolve():
        parser.error("Parent and output must be different files")
    if sha256_file(args.parent) != args.parent_sha256:
        parser.error("Parent SHA-256 mismatch")
    parent = load_evaluation_checkpoint(args.parent)
    payload = torch.load(args.parent, map_location="cpu", weights_only=True)
    target = build_parent_network(parent.config, seed=MIGRATION_INIT_SEED)
    target.load_state_dict(payload["learner_state"]["target_network"], strict=True)
    config = successor_config(parent.config)
    migrated_network = migrate_online_network(
        parent.network,
        config=config,
        seed=MIGRATION_INIT_SEED,
    )

    learner = DQNLearner(config=config, seed=MIGRATION_INIT_SEED)
    learner.online_network.load_state_dict(migrated_network.state_dict())
    learner.target_network.load_state_dict(
        migrate_online_network(target, config=config, seed=MIGRATION_INIT_SEED).state_dict()
    )

    save_checkpoint(
        learner=learner,
        replay_buffer=ReplayBuffer(
            capacity=config.replay_capacity,
            seed=MIGRATION_INIT_SEED,
        ),
        action_rng=np.random.default_rng(MIGRATION_INIT_SEED),
        epsilon=config.initial_epsilon,
        completed_episodes=0,
        agent_seed=MIGRATION_INIT_SEED,
        path=args.output,
    )

    print(f"Parent checkpoint:    {args.parent}")
    print(f"Parent SHA-256:       {sha256_file(args.parent)}")
    print(f"Migrated checkpoint:  {args.output}")
    print(f"Migrated SHA-256:      {sha256_file(args.output)}")
    print(f"Migrated size:         {args.output.stat().st_size} bytes")


if __name__ == "__main__":
    main()
