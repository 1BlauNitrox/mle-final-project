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

OPPONENT_FEATURE_COUNT = 8
MAX_OPPONENT_COUNT_BIN = 3


def opponent_features(
    *,
    game_state: dict,
    field: np.ndarray,
    blocked_positions: set[Position],
    position: Position,
    danger_map_with_hypothetical_bomb: DangerMap,
    bombs_with_hypothetical: list[tuple[Position, int]],
) -> tuple[int, ...]:
    """Return the version-three eight-value opponent feature suffix.

    The attack descriptors answer what a bomb at the current position would
    affect and whether the current geometry leaves a time-safe escape. They
    are descriptive inputs, not a selected action or a safety mask.
    """
    opponent_positions = _opponent_positions(game_state)
    if not opponent_positions:
        return (0,) * OPPONENT_FEATURE_COUNT

    nearest = min(
        opponent_positions,
        key=lambda candidate: (
            _manhattan_distance(position, candidate),
            candidate[0],
            candidate[1],
        ),
    )
    x, y = position
    nearest_x, nearest_y = nearest
    distance = _manhattan_distance(position, nearest)
    footprint = set(blast_footprint(position, field))
    blast_count = min(sum(candidate in footprint for candidate in opponent_positions), 3)
    attack_opportunity = int(blast_count > 0)
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
    adjacent_count = min(
        sum(
            (x + dx, y + dy) in opponent_positions
            for dx, dy in DIRECTIONS
        ),
        MAX_OPPONENT_COUNT_BIN,
    )

    return (
        1,
        _sign(nearest_x - x),
        _sign(nearest_y - y),
        _distance_bin(distance),
        attack_opportunity,
        attack_escape_exists,
        blast_count,
        adjacent_count,
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
