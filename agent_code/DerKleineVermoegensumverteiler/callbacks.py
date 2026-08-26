"""Runnable starting point for a new learned agent.

Copy and rename this directory before implementing a real project agent. The
included policy is an intentionally weak action-value baseline, not a tournament
candidate.
"""

from __future__ import annotations

import os

import numpy as np

from .config import DEFAULT_SEED, INITIAL_EPSILON
from .features import state_to_features
from .model import QTable


def setup(self) -> None:
    """Initialize the Qtable, random generator and exploration state."""

    agent_seed = _read_agent_seed()

    self.rng = np.random.default_rng(agent_seed)
    self.q_table = QTable()
    self.epsilon = INITIAL_EPSILON if self.train else 0.0

    self.logger.info(
        "Initialized task 1 baseline agent with seed %d",
        agent_seed
    )


def act(self, game_state: dict) -> str:
    """Choose an action using epsilon-greedy exploration during training."""
    
    state = state_to_features(game_state)

    if state is None:
        return "WAIT"
    
    epsilon = self.epsilon if self.train else 0.0

    return self.q_table.select_action(
        state,
        epsilon=epsilon,
        rng=self.rng,
    )

def _read_agent_seed() -> int:
    """Read the agent seed from the environment."""
    
    raw_seed = os.environ.get("BOMBERMAN_AGENT_SEED")

    if raw_seed is None:
        return DEFAULT_SEED
    
    try:
        return int(raw_seed)
    except ValueError as error:
        raise ValueError(
            "BOMBERMAN_AGENT_SEED must be an integer"
        ) from error
