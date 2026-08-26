"""Q-tabel and policy"""

from __future__ import annotations

import numpy as np

from .config import ACTIONS, DISCOUNT_FACTOR, LEARNING_RATE
from .features import StateFeatures


class QTable:
    """A simple Q-table for tabular reinforcement learning."""

    def __init__(
        self, *, learning_rate: float = LEARNING_RATE, discount_factor: float = DISCOUNT_FACTOR
    ) -> None:

        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("Learning rate must be in (0, 1].")

        if not 0.0 <= discount_factor <= 1.0:
            raise ValueError("Discount factor must be in [0, 1].")

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.values: dict[StateFeatures, np.ndarray] = {}

    def q_values(self, state: StateFeatures) -> np.ndarray:
        """Get the Q-values for a given state."""

        values = self.values.get(state)

        if values is None:
            return np.zeros(len(ACTIONS), dtype=float)

        return values.copy()

    def select_action(
        self, state: StateFeatures, *, epsilon: float, rng: np.random.Generator
    ) -> str:
        """Select an action using epsilon-greedy exploration."""

        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("Epsilon must be in [0, 1].")

        if rng.random() < epsilon:
            action_index = int(rng.integers(len(ACTIONS)))
            return ACTIONS[action_index]

        values = self.q_values(state)
        maximum = np.max(values)

        best_indices = np.flatnonzero(np.isclose(values, maximum))

        select_index = int(rng.choice(best_indices))
        return ACTIONS[select_index]

    def update(
        self,
        *,
        state: StateFeatures,
        action: str,
        reward: float,
        next_state: StateFeatures | None,
        terminal: bool,
    ) -> float:
        """Update the Q-value and return the TD error"""
        if action not in ACTIONS:
            raise ValueError(f"Invalid action: {action}")

        if not terminal and next_state is None:
            raise ValueError("Next state must be provided for non-terminal updates.")

        current_values = self._get_or_create(state)
        action_index = ACTIONS.index(action)
        current_value = current_values[action_index]

        if terminal:
            target = reward
        else:
            assert next_state is not None
            target = reward + self.discount_factor * float(np.max(self.q_values(next_state)))

        td_error = target - current_value
        current_values[action_index] += self.learning_rate * td_error

        return float(td_error)

    def __len__(self) -> int:
        """Return the number of states stored in the Q-table."""
        return len(self.values)

    def _get_or_create(self, state: StateFeatures) -> np.ndarray:
        """Get the Q-values for a state, creating them if they don't exist."""
        if state not in self.values:
            self.values[state] = np.zeros(len(ACTIONS), dtype=float)

        return self.values[state]
