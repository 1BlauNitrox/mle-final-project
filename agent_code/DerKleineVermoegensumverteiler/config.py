"""actions, hyperparameters, rewards and seeds"""

from __future__ import annotations

# actions
ACTIONS: tuple[str, ...] = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT")

ACTION_TO_INDEX: dict[str, int] = {action: index for index, action in enumerate(ACTIONS)}

# hyperparameters
LEARNING_RATE: float = 0.1
DISCOUNT_FACTOR: float = 0.9

INITIAL_EPSILON: float = 1.0
EPSILON_DECAY: float = 0.99
MINIMUM_EPSILON: float = 0.1

# rewards
REWARDS: dict[str, float] = {
    "INVALID_ACTION": -0.5,
    "WAITED": -0.1,
    "COIN_COLLECTED": 10.0,
    # "MOVED_TOWARDS_COIN": 1.0,
    # "MOVED_AWAY_FROM_COIN": -1.0,
}

# seeds
DEFAULT_SEED: int = 0
