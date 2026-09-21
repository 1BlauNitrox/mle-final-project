"""Configuration for the opponent-aware tabular agent."""

from __future__ import annotations

ACTIONS: tuple[str, ...] = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")

ACTION_TO_INDEX: dict[str, int] = {action: index for index, action in enumerate(ACTIONS)}

LEARNING_RATE: float = 0.05
DISCOUNT_FACTOR: float = 0.9

INITIAL_EPSILON: float = 1.0
EPSILON_DECAY: float = 0.99
MINIMUM_EPSILON: float = 0.1

# Earlier rewards stay unchanged so the agent keeps its coin and crate behavior.
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
    "KILLED_OPPONENT": 5.0,
}

DEFAULT_SEED: int = 0
