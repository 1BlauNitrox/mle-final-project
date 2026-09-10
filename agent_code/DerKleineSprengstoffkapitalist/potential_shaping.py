"""Potential-based safety shaping for the compact Task 2 state."""

from __future__ import annotations

from .config import DISCOUNT_FACTOR
from .features import StateFeatures
from .features.compact import validate_compact_features

NO_POTENTIAL_SHAPING = "none"
COMPACT_SAFETY_POTENTIAL_SHAPING = "compact_safety"

VALID_POTENTIAL_SHAPING_MODES = (
    NO_POTENTIAL_SHAPING,
    COMPACT_SAFETY_POTENTIAL_SHAPING,
)


def safety_potential(state: StateFeatures) -> float:
    """Return the registered safety potential for a compact state."""

    validate_compact_features(state)

    danger_level = state[0]
    safe_directions_mask = state[1]

    if danger_level == 0:
        return 0.0

    if safe_directions_mask != 0:
        return -1.0

    return -2.0


def potential_safety_reward(
    state: StateFeatures,
    next_state: StateFeatures | None,
    *,
    terminal: bool,
    discount_factor: float = DISCOUNT_FACTOR,
) -> float:
    """Return gamma * Phi(next_state) - Phi(state)."""

    if not 0.0 <= discount_factor <= 1.0:
        raise ValueError("Discount factor must be in [0, 1].")

    current_potential = safety_potential(state)

    if terminal:
        if next_state is not None:
            raise ValueError(
                "Terminal shaping transitions cannot have a next state."
            )

        next_potential = 0.0
    else:
        if next_state is None:
            raise ValueError(
                "Non-terminal shaping transitions require a next state."
            )

        next_potential = safety_potential(next_state)

    return (
        discount_factor * next_potential
        - current_potential
    )


def apply_potential_shaping(
    reward: float,
    state: StateFeatures,
    next_state: StateFeatures | None,
    *,
    terminal: bool,
    mode: str,
    discount_factor: float = DISCOUNT_FACTOR,
) -> float:
    """Add the selected potential-based shaping term to a reward."""

    if mode not in VALID_POTENTIAL_SHAPING_MODES:
        raise ValueError(
            "Potential shaping mode must be one of "
            f"{list(VALID_POTENTIAL_SHAPING_MODES)}."
        )

    if mode == NO_POTENTIAL_SHAPING:
        return float(reward)

    return float(
        reward
        + potential_safety_reward(
            state,
            next_state,
            terminal=terminal,
            discount_factor=discount_factor,
        )
    )