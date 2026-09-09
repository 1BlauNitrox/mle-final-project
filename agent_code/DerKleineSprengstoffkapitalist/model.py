"""Q-tabel and policy"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .config import ACTIONS, DISCOUNT_FACTOR, LEARNING_RATE
from .features import FEATURE_COUNT, StateFeatures

PARENT_FEATURE_COUNT = 8
PARENT_ACTION_COUNT = 5

BOMB_PRIOR_MARGIN = 1.0

PARENT_PRIOR_INITIALIZATION = "parent_prior"
ZERO_INITIALIZATION = "zeros"

VALID_INITIALIZATIONS = (
    PARENT_PRIOR_INITIALIZATION,
    ZERO_INITIALIZATION,
)

Task1State = tuple[int, ...]


class QTable:
    """Sparse Task 2 Q-table initialized from frozen Task 1 values."""

    def __init__(
        self,
        *,
        learning_rate: float = LEARNING_RATE,
        discount_factor: float = DISCOUNT_FACTOR,
        parent_values: Mapping[Task1State, np.ndarray] | None = None,
        feature_count: int = FEATURE_COUNT,
        initialization: str = PARENT_PRIOR_INITIALIZATION,
    ) -> None:
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("Learning rate must be in (0, 1].")

        if not 0.0 <= discount_factor <= 1.0:
            raise ValueError("Discount factor must be in [0, 1].")

        if type(feature_count) is not int or feature_count < 1:
            raise ValueError("Feature count must be a positive integer.")

        if initialization not in VALID_INITIALIZATIONS:
            raise ValueError(
                f"Initialization must be one of {list(VALID_INITIALIZATIONS)}."
            )

        if initialization == PARENT_PRIOR_INITIALIZATION:
            if feature_count < PARENT_FEATURE_COUNT:
                raise ValueError(
                    "Parent-prior initialization requires at least "
                    f"{PARENT_FEATURE_COUNT} features."
                )
        elif parent_values:
            raise ValueError(
                "Zero initialization cannot receive parent Q-values."
            )

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.feature_count = feature_count
        self.initialization = initialization
        self.parent_values = self._copy_parent_values(parent_values or {})
        self.values: dict[StateFeatures, np.ndarray] = {}

    def q_values(self, state: StateFeatures) -> np.ndarray:
        """Return Q-values without creating a sparse-table entry."""

        self._validate_state(state)

        values = self.values.get(state)

        if values is None:
            return self._initial_values(state)

        return values.copy()

    def select_action(
        self,
        state: StateFeatures,
        *,
        epsilon: float,
        rng: np.random.Generator,
        action_mask: np.ndarray | None = None,
    ) -> str:
        """Select an action using epsilon-greedy exploration."""

        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("Epsilon must be in [0, 1].")

        legal = _validate_action_mask(action_mask)
        legal_indices = np.flatnonzero(legal)

        if rng.random() < epsilon:
            selected_index = int(rng.choice(legal_indices))
            return ACTIONS[selected_index]

        values = self.q_values(state)
        masked_values = np.where(legal, values, -np.inf)
        maximum = np.max(masked_values)
        best_indices = np.flatnonzero(masked_values == maximum)

        selected_index = int(rng.choice(best_indices))
        return ACTIONS[selected_index]

    def update(
        self,
        *,
        state: StateFeatures,
        action: str,
        reward: float,
        next_state: StateFeatures | None,
        terminal: bool,
        next_action_mask: np.ndarray | None = None,
    ) -> float:
        """Update one Q-value and return its temporal difference."""

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

            legal = _validate_action_mask(next_action_mask)
            next_values = self.q_values(next_state)
            masked_next_values = np.where(legal, next_values, -np.inf)
            maximum_next_value = float(np.max(masked_next_values))

            target = reward + self.discount_factor * maximum_next_value

        td_error = target - current_value

        current_values[action_index] += self.learning_rate * td_error

        return float(td_error)

    def __len__(self) -> int:
        """Return the number of materialized Task 2 states."""

        return len(self.values)

    def _initial_values(
        self,
        state: StateFeatures,
    ) -> np.ndarray:
        """Return the configured initial Q-values for an unseen state."""

        if self.initialization == ZERO_INITIALIZATION:
            return np.zeros(len(ACTIONS), dtype=float)

        parent_state = tuple(state[:PARENT_FEATURE_COUNT])

        parent_q_values = self.parent_values.get(parent_state)

        if parent_q_values is None:
            parent_q_values = np.zeros(
                PARENT_ACTION_COUNT,
                dtype=float,
            )
        else:
            parent_q_values = parent_q_values.copy()

        bomb_prior = float(np.min(parent_q_values) - BOMB_PRIOR_MARGIN)

        return np.concatenate(
            (
                parent_q_values,
                np.array([bomb_prior], dtype=float),
            )
        )

    def _get_or_create(
        self,
        state: StateFeatures,
    ) -> np.ndarray:
        """Materialize a Task 2 state only during an update."""

        self._validate_state(state)

        if state not in self.values:
            self.values[state] = self._initial_values(state)

        return self.values[state]

    def _validate_state(self, state: StateFeatures) -> None:
        """Validate the minimum structural Task 2 state contract."""

        if not isinstance(state, tuple):
            raise ValueError("State must be a tuple.")

        if len(state) != self.feature_count:
            raise ValueError(
                f"Expected {self.feature_count} state values, got {len(state)}."
            )

    @staticmethod
    def _copy_parent_values(
        parent_values: Mapping[Task1State, np.ndarray],
    ) -> dict[Task1State, np.ndarray]:
        """Validate and defensively copy parent Q-values."""

        copied: dict[Task1State, np.ndarray] = {}

        for raw_state, raw_values in parent_values.items():
            state = tuple(raw_state)
            values = np.asarray(raw_values, dtype=float)

            if len(state) != PARENT_FEATURE_COUNT:
                raise ValueError("Parent state has an incompatible feature count.")

            if values.shape != (PARENT_ACTION_COUNT,):
                raise ValueError("Parent Q-values have an incompatible shape.")

            if not np.all(np.isfinite(values)):
                raise ValueError("Parent Q-values must be finite.")

            copied[state] = values.copy()

        return copied


def _validate_action_mask(
    action_mask: np.ndarray | None,
) -> np.ndarray:
    """Validate an optional action mask in ACTIONS order."""

    if action_mask is None:
        return np.ones(len(ACTIONS), dtype=np.bool_)

    mask = np.asarray(action_mask)

    if mask.shape != (len(ACTIONS),) or mask.dtype != np.bool_:
        raise ValueError("action_mask must be a boolean vector in ACTIONS order.")

    if not np.any(mask):
        raise ValueError("action_mask must retain at least one legal action.")

    return mask.copy()
