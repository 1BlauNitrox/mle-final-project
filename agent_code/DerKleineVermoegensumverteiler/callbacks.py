"""Runnable starting point for a new learned agent.

Copy and rename this directory before implementing a real project agent. The
included policy is an intentionally weak action-value baseline, not a tournament
candidate.
"""

from pathlib import Path

import numpy as np

from agent_code.DerKleineVermoegensumverteiler.config import ACTIONS

MODEL_PATH = Path(__file__).with_name("model.npz")


def setup(self) -> None:
    """Load learned action values or initialize an untrained baseline."""
    if MODEL_PATH.is_file():
        with np.load(MODEL_PATH) as data:
            self.action_values = data["action_values"]
            self.action_counts = data["action_counts"]
        self.logger.info("Loaded model from %s", MODEL_PATH.name)
    else:
        self.action_values = np.zeros(len(ACTIONS), dtype=float)
        self.action_counts = np.zeros(len(ACTIONS), dtype=np.int64)
        self.logger.info("No model found; initialized the template baseline")


def act(self, game_state: dict) -> str:
    """Choose an action using epsilon-greedy exploration during training."""
    del game_state
    if self.train and np.random.random() < 0.20:
        return str(np.random.choice(ACTIONS))

    best_indices = np.flatnonzero(self.action_values == self.action_values.max())
    return ACTIONS[int(np.random.choice(best_indices))]
