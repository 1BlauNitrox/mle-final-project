"""Framework setup and action callbacks for DagobertDuckDQN."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from .config import DEFAULT_CONFIG
from .features import normalize_features, state_to_features
from .model import select_action
from .persistence import CHECKPOINT_PATH, load_evaluation_checkpoint

EVALUATION_CHECKPOINT_ENV = "BOMBERMAN_EVALUATION_CHECKPOINT"


def setup(self) -> None:
    """Load the frozen evaluation network.

    There is no resumable training state to restore: DagobertDuckDQN is a
    frozen Task 1 baseline, and the committed artifact is evaluation-only
    (see persistence.py). This runs the same way regardless of `self.train`
    -- train.py's setup_training rejects training immediately afterward, but
    that rejection must not depend on this function having already loaded a
    (nonexistent) resumable checkpoint first.
    """
    agent_seed = _read_agent_seed()
    self.agent_seed = agent_seed

    loaded = load_evaluation_checkpoint(_evaluation_checkpoint_path())

    self.config = loaded.config
    self.policy_network = loaded.network
    self.action_rng, _unused_replay_seed = _initial_random_streams(agent_seed)
    self.epsilon = 0.0
    self.completed_episodes = loaded.completed_episodes

    self.logger.info(
        "Loaded frozen DQN policy after %d training episodes",
        self.completed_episodes,
    )

    torch.set_num_threads(self.config.torch_num_threads)


def act(self, game_state: dict | None) -> str:
    """Select one seeded greedy Task 1 action."""
    features = state_to_features(game_state)

    if features is None:
        return "WAIT"

    state = normalize_features(features)

    return select_action(
        network=self.policy_network,
        state=state,
        epsilon=self.epsilon,
        rng=self.action_rng,
    )


def _evaluation_checkpoint_path() -> Path:
    """Resolve an optional repository-managed evaluation artifact locally."""
    file_name = os.environ.get(EVALUATION_CHECKPOINT_ENV)
    if file_name is None:
        return CHECKPOINT_PATH
    if not file_name or file_name != os.path.basename(file_name):
        raise ValueError(
            f"{EVALUATION_CHECKPOINT_ENV} must contain one file name."
        )
    return CHECKPOINT_PATH.with_name(file_name)


def _initial_random_streams(
    agent_seed: int,
) -> tuple[np.random.Generator, int]:
    """Derive independent action and replay streams from one seed."""
    if agent_seed < 0:
        raise ValueError("BOMBERMAN_AGENT_SEED must be non-negative.")

    root_sequence = np.random.SeedSequence(agent_seed)
    action_sequence, replay_sequence = root_sequence.spawn(2)

    action_rng = np.random.default_rng(action_sequence)
    replay_seed = int(
        replay_sequence.generate_state(
            1,
            dtype=np.uint64,
        )[0]
    )

    return action_rng, replay_seed


def _read_agent_seed() -> int:
    """Read the explicit agent seed from the environment."""
    raw_seed = os.environ.get("BOMBERMAN_AGENT_SEED")

    if raw_seed is None:
        return DEFAULT_CONFIG.default_seed

    try:
        seed = int(raw_seed)
    except ValueError as error:
        raise ValueError("BOMBERMAN_AGENT_SEED must be an integer.") from error

    if seed < 0:
        raise ValueError("BOMBERMAN_AGENT_SEED must be non-negative.")

    return seed
