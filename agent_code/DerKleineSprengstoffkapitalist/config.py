"""actions, hyperparameters, rewards and seeds"""

from __future__ import annotations

# actions
ACTIONS: tuple[str, ...] = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")

ACTION_TO_INDEX: dict[str, int] = {action: index for index, action in enumerate(ACTIONS)}

# hyperparameters
LEARNING_RATE: float = 0.05
DISCOUNT_FACTOR: float = 0.9

INITIAL_EPSILON: float = 1.0
EPSILON_DECAY: float = 0.99
MINIMUM_EPSILON: float = 0.1

# rewards
# Task 1 rewards are inherited unchanged from the frozen parent.
#
# Task 2 native-event rewards are implementation defaults for issue #45.
# They have not yet been validated by a prospective experiment.
REWARDS: dict[str, float] = {
    "COIN_COLLECTED": 10.0,
    "INVALID_ACTION": -0.5,
    "WAITED": -0.1,
    "MOVED_TOWARDS_COIN": 0.1,
    "MOVED_AWAY_FROM_COIN": -0.1,
    "CRATE_DESTROYED": 1.0,
    "COIN_FOUND": 2.0,
    "KILLED_SELF": -10.0,
    "GOT_KILLED": -10.0,
    "SURVIVED_ROUND": 5.0,
}

# seeds
DEFAULT_SEED: int = 0
