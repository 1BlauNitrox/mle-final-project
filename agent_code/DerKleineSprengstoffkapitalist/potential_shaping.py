"""Potential-based safety shaping for the compact Task 2 state."""

from __future__ import annotations

from .config import DISCOUNT_FACTOR
from .features import StateFeatures
from .features.bombs_and_crates import (
    build_danger_map,
    shortest_safe_escape_distance,
)
from .features.compact import validate_compact_features
from .features.navigation import _blocked_positions

NO_POTENTIAL_SHAPING = "none"
COMPACT_SAFETY_POTENTIAL_SHAPING = "compact_safety"
ESCAPE_DISTANCE_POTENTIAL_SHAPING = "escape_distance"
HALF_ESCAPE_DISTANCE_POTENTIAL_SHAPING = "escape_distance_half"

VALID_POTENTIAL_SHAPING_MODES = (
    NO_POTENTIAL_SHAPING,
    COMPACT_SAFETY_POTENTIAL_SHAPING,
    ESCAPE_DISTANCE_POTENTIAL_SHAPING,
    HALF_ESCAPE_DISTANCE_POTENTIAL_SHAPING,
)

ESCAPE_DISTANCE_POTENTIAL_SCALES = {
    ESCAPE_DISTANCE_POTENTIAL_SHAPING: 1.0,
    HALF_ESCAPE_DISTANCE_POTENTIAL_SHAPING: 0.5,
}


def escape_distance_potential_from_distance(distance: int | None) -> float:
    """Map a shortest persistent-safety distance to the registered potential."""
    if distance is None:
        return -5.0

    if isinstance(distance, bool) or not isinstance(distance, int) or distance < 0:
        raise ValueError("Escape distance must be a non-negative integer or None.")

    if distance == 0:
        return 0.0

    if distance == 1:
        return -1.0

    if distance == 2:
        return -2.0

    return -3.0


def escape_distance_potential(game_state: dict, *, scale: float = 1.0) -> float:
    """Return the time-aware escape-distance potential for a game state."""
    if not 0.0 < scale <= 1.0:
        raise ValueError("Escape-distance potential scale must be in (0, 1].")

    field = game_state["field"]
    position = game_state["self"][3]
    bombs = game_state.get("bombs", [])
    danger_map = build_danger_map(
        field,
        bombs,
        game_state["explosion_map"],
    )
    distance = shortest_safe_escape_distance(
        field,
        danger_map,
        _blocked_positions(game_state),
        bombs,
        position,
    )
    return scale * escape_distance_potential_from_distance(distance)


def potential_reward_from_values(
    current_potential: float,
    next_potential: float | None,
    *,
    terminal: bool,
    discount_factor: float = DISCOUNT_FACTOR,
) -> float:
    """Return gamma * Phi(next) - Phi(current) for supplied potentials."""
    if not 0.0 <= discount_factor <= 1.0:
        raise ValueError("Discount factor must be in [0, 1].")

    if terminal:
        if next_potential is not None:
            raise ValueError("Terminal shaping transitions cannot have a next potential.")
        successor = 0.0
    else:
        if next_potential is None:
            raise ValueError("Non-terminal shaping transitions require a next potential.")
        successor = float(next_potential)

    return discount_factor * successor - float(current_potential)


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

    current_potential = safety_potential(state)

    if terminal:
        if next_state is not None:
            raise ValueError(
                "Terminal shaping transitions cannot have a next state."
            )

        next_potential = None
    else:
        if next_state is None:
            raise ValueError(
                "Non-terminal shaping transitions require a next state."
            )

        next_potential = safety_potential(next_state)

    return potential_reward_from_values(
        current_potential,
        next_potential,
        terminal=terminal,
        discount_factor=discount_factor,
    )


def apply_potential_shaping(
    reward: float,
    state: StateFeatures,
    next_state: StateFeatures | None,
    *,
    terminal: bool,
    mode: str,
    discount_factor: float = DISCOUNT_FACTOR,
    current_external_potential: float | None = None,
    next_external_potential: float | None = None,
) -> float:
    """Add the selected potential-based shaping term to a reward."""

    if mode not in VALID_POTENTIAL_SHAPING_MODES:
        raise ValueError(
            "Potential shaping mode must be one of "
            f"{list(VALID_POTENTIAL_SHAPING_MODES)}."
        )

    if mode == NO_POTENTIAL_SHAPING:
        return float(reward)

    if mode in ESCAPE_DISTANCE_POTENTIAL_SCALES:
        if current_external_potential is None:
            raise ValueError(
                "Escape-distance shaping requires the current potential."
            )
        return float(
            reward
            + potential_reward_from_values(
                current_external_potential,
                next_external_potential,
                terminal=terminal,
                discount_factor=discount_factor,
            )
        )

    return float(
        reward
        + potential_safety_reward(
            state,
            next_state,
            terminal=terminal,
            discount_factor=discount_factor,
        )
    )
