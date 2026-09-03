"""Export the frozen Task 1 DagobertDuckDQN artifact selected under issue #42.

Issue #58's registered stopping rule ("no third Task 1 DQN tuning experiment")
and the prospective candidate-selection rule posted to #42 together identify
`run-02` of the #58 series as the candidate. This script reads that run's full
resumable checkpoint and re-saves only what evaluation needs, through the
existing, already-tested checkpoint schema: the learned network weights,
config, rewards, and completed-episode count. The replay buffer and optimizer
state are training-only and reset to fresh/empty, which is what actually
shrinks the artifact -- the schema itself is unchanged.

Run once, locally, against the retained training output. The result is what
gets committed; this script is kept for reviewer transparency and
reproducibility, not for repeated execution.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from agent_code.DagobertDuckDQN.model import DQNLearner
from agent_code.DagobertDuckDQN.persistence import (
    CHECKPOINT_PATH,
    load_evaluation_checkpoint,
    load_training_checkpoint,
    save_checkpoint,
)
from agent_code.DagobertDuckDQN.replay import ReplayBuffer

EXPECTED_WORLD_SEED = 15002
EXPECTED_AGENT_SEED = 25002
EXPECTED_COMPLETED_EPISODES = 10_000


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 checksum of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def build_compact_checkpoint(
    source: Path, *, agent_seed: int
) -> tuple[DQNLearner, ReplayBuffer, np.random.Generator, int]:
    """Load a resumable checkpoint and strip everything but the policy."""
    loaded = load_training_checkpoint(source)

    if loaded.agent_seed != agent_seed:
        raise ValueError(
            f"Source checkpoint agent_seed {loaded.agent_seed} does not match "
            f"the expected run-02 agent_seed {agent_seed}."
        )

    if loaded.completed_episodes != EXPECTED_COMPLETED_EPISODES:
        raise ValueError(
            f"Source checkpoint completed {loaded.completed_episodes} episodes, "
            f"expected {EXPECTED_COMPLETED_EPISODES}."
        )

    compact_learner = DQNLearner(config=loaded.config, seed=agent_seed)
    compact_learner.online_network.load_state_dict(
        loaded.learner.online_network.state_dict()
    )
    compact_learner.target_network.load_state_dict(
        loaded.learner.target_network.state_dict()
    )

    empty_replay = ReplayBuffer(
        capacity=loaded.config.replay_capacity,
        seed=agent_seed,
    )
    fresh_action_rng = np.random.default_rng(agent_seed)

    return compact_learner, empty_replay, fresh_action_rng, loaded.completed_episodes


def verify_policy_preserved(source: Path, frozen: Path) -> None:
    """Confirm the frozen artifact selects identically to the source checkpoint."""
    source_network = load_evaluation_checkpoint(source).network
    frozen_network = load_evaluation_checkpoint(frozen).network

    probe_states = torch.randn(64, source_network.config.input_dim)

    with torch.no_grad():
        source_values = source_network(probe_states)
        frozen_values = frozen_network(probe_states)

    if not torch.equal(source_values, frozen_values):
        raise RuntimeError(
            "Compact artifact does not reproduce the source network's Q-values."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to run-02's retained full checkpoint "
        "(training_outputs/issue-58-.../artifacts/run-02-final-checkpoint.pt).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CHECKPOINT_PATH,
        help="Destination for the frozen, compact checkpoint.",
    )
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing frozen artifact: {args.output}"
        )

    learner, replay_buffer, action_rng, completed_episodes = build_compact_checkpoint(
        args.source,
        agent_seed=EXPECTED_AGENT_SEED,
    )

    save_checkpoint(
        learner=learner,
        replay_buffer=replay_buffer,
        action_rng=action_rng,
        epsilon=0.0,
        completed_episodes=completed_episodes,
        agent_seed=EXPECTED_AGENT_SEED,
        path=args.output,
    )

    verify_policy_preserved(args.source, args.output)

    source_size = args.source.stat().st_size
    frozen_size = args.output.stat().st_size

    print(f"Source checkpoint:  {args.source} ({source_size} bytes)")
    print(f"Frozen artifact:    {args.output} ({frozen_size} bytes)")
    print(f"Frozen SHA-256:     {sha256_file(args.output)}")
    print(f"Reduction:          {source_size - frozen_size} bytes")


if __name__ == "__main__":
    main()
