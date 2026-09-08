"""Descriptive opponent features for the Task 3 DQN.

Only public ``game_state['others']`` information is used. The helpers encode
geometry and occupancy; they never select or rank an action for the policy.
"""

from __future__ import annotations

import numpy as np

from .bombs_and_crates import (
    DangerMap,
    Position,
    blast_footprint,
    safe_escape_exists,
)
from .navigation import DIRECTIONS

OPPONENT_FEATURE_COUNT = 13


def opponent_features(
    *,
    game_state: dict,
    field: np.ndarray,
    blocked_positions: set[Position],
    position: Position,
    danger_map_with_hypothetical_bomb: DangerMap,
    bombs_with_hypothetical: list[tuple[Position, int]],
) -> tuple[int, ...]:
    """Return the version-four thirteen-value opponent feature suffix.

    The attack descriptors answer what a bomb at the current position would
    affect and whether the current geometry leaves a time-safe escape. The
    four per-direction occupancy flags and the second-nearest descriptors
    are what make "opponents as obstacles" and multi-opponent awareness
    (up to three in `classic`) explicit rather than collapsed into a single
    count. They are descriptive inputs, not a selected action or a safety
    mask.
    """
    ranked = _ranked_opponents(game_state, position)
    if not ranked:
        return (0,) * OPPONENT_FEATURE_COUNT

    x, y = position
    opponent_positions = set(ranked)
    nearest = ranked[0]
    nearest_distance = _manhattan_distance(position, nearest)

    footprint = set(blast_footprint(position, field))
    attack_opportunity = int(
        any(candidate in footprint for candidate in opponent_positions)
    )
    attack_escape_exists = int(
        attack_opportunity
        and safe_escape_exists(
            field,
            danger_map_with_hypothetical_bomb,
            blocked_positions,
            bombs_with_hypothetical,
            position,
        )
    )
    adjacency = tuple(
        int((x + dx, y + dy) in opponent_positions) for dx, dy in DIRECTIONS
    )

    if len(ranked) >= 2:
        second = ranked[1]
        second_sign_x = _sign(second[0] - x)
        second_sign_y = _sign(second[1] - y)
        second_distance_bin = _distance_bin(_manhattan_distance(position, second))
    else:
        second_sign_x = 0
        second_sign_y = 0
        second_distance_bin = 0

    return (
        1,
        _sign(nearest[0] - x),
        _sign(nearest[1] - y),
        _distance_bin(nearest_distance),
        attack_opportunity,
        attack_escape_exists,
        *adjacency,
        second_sign_x,
        second_sign_y,
        second_distance_bin,
    )


def _ranked_opponents(game_state: dict, position: Position) -> tuple[Position, ...]:
    """Return public opponent positions ordered nearest-first, ties broken by coordinate."""
    positions = _opponent_positions(game_state)
    return tuple(
        sorted(
            positions,
            key=lambda candidate: (
                _manhattan_distance(position, candidate),
                candidate[0],
                candidate[1],
            ),
        )
    )


def _opponent_positions(game_state: dict) -> set[Position]:
    """Extract only positions exposed by the public opponent state."""
    positions: set[Position] = set()
    for opponent in game_state.get("others", []):
        if len(opponent) < 4:
            raise ValueError("Opponent state must contain a public position.")
        positions.add(tuple(opponent[3]))
    return positions


def _manhattan_distance(first: Position, second: Position) -> int:
    return abs(first[0] - second[0]) + abs(first[1] - second[1])


def _sign(value: int) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _distance_bin(distance: int) -> int:
    if distance <= 1:
        return 1
    if distance <= 3:
        return 2
    return 3
