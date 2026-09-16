"""Compact decision-oriented Task 2 state representation."""

from __future__ import annotations

from collections import deque
from typing import TypeAlias

import numpy as np

from .bombs_and_crates import (
    BOMB_TIMER,
    build_danger_map,
    crates_destroyed_by_bomb_at,
    danger_countdown_bin,
    nearest_crate_features,
    safe_escape_exists,
)
from .navigation import DIRECTIONS, _blocked_positions, _is_free_tile

CompactStateFeatures: TypeAlias = tuple[int, ...]

COMPACT_FEATURE_SCHEMA_VERSION = 1

COMPACT_FEATURE_NAMES: tuple[str, ...] = (
    "danger_level",
    "safe_directions_mask",
    "coin_direction",
    "crate_direction",
    "bomb_status",
)

COMPACT_FEATURE_DOMAINS: tuple[frozenset[int], ...] = (
    frozenset({0, 1, 2}),
    frozenset(range(16)),
    frozenset(range(5)),
    frozenset(range(5)),
    frozenset(range(4)),
)

COMPACT_FEATURE_COUNT = len(COMPACT_FEATURE_NAMES)
COMPACT_STATE_SPACE_UPPER_BOUND = 3 * 16 * 5 * 5 * 4

NONE = 0
UP = 1
RIGHT = 2
DOWN = 3
LEFT = 4

_DIRECTION_CODES = {
    direction: index
    for index, direction in enumerate(DIRECTIONS, start=1)
}


def compact_state_to_features(
    game_state: dict | None,
) -> CompactStateFeatures | None:
    """Convert a framework game state to the compact Task 2 state."""

    if game_state is None:
        return None

    field = game_state["field"]
    _, _, bomb_available, position = game_state["self"]
    bombs = game_state.get("bombs", [])
    explosion_map = game_state["explosion_map"]
    blocked_positions = _blocked_positions(game_state)

    danger_map = build_danger_map(field, bombs, explosion_map)

    danger_level = _compact_danger_level(
        danger_countdown_bin(danger_map, position)
    )

    safe_directions_mask = _safe_directions_mask(
        field=field,
        danger_map=danger_map,
        blocked_positions=blocked_positions,
        bombs=bombs,
        position=position,
    )

    coin_direction = _nearest_coin_direction(
        position=position,
        coins=game_state["coins"],
        field=field,
        blocked_positions=blocked_positions,
    )

    crate_features = nearest_crate_features(
        position=position,
        field=field,
        blocked_positions=blocked_positions,
    )
    crate_direction = _direction_code(
        (crate_features[1], crate_features[2])
        if crate_features[0]
        else (0, 0)
    )

    useful_bomb = crates_destroyed_by_bomb_at(position, field) > 0

    if not bomb_available:
        bomb_status = 0
    elif not useful_bomb:
        bomb_status = 1
    else:
        hypothetical_bombs = [
            *bombs,
            (position, BOMB_TIMER - 1),
        ]
        hypothetical_danger_map = build_danger_map(
            field,
            hypothetical_bombs,
            explosion_map,
        )
        safe_escape = safe_escape_exists(
            field,
            hypothetical_danger_map,
            blocked_positions,
            hypothetical_bombs,
            position,
        )
        bomb_status = 3 if safe_escape else 2

    features: CompactStateFeatures = (
        danger_level,
        safe_directions_mask,
        coin_direction,
        crate_direction,
        bomb_status,
    )

    validate_compact_features(features)
    return features


def validate_compact_features(features: CompactStateFeatures) -> None:
    """Validate the compact feature tuple."""

    if len(features) != COMPACT_FEATURE_COUNT:
        raise ValueError(
            f"Expected {COMPACT_FEATURE_COUNT} compact features, "
            f"got {len(features)}."
        )

    for index, (value, domain) in enumerate(
        zip(features, COMPACT_FEATURE_DOMAINS, strict=True)
    ):
        if type(value) is not int:
            raise ValueError(
                f"Compact feature {COMPACT_FEATURE_NAMES[index]} "
                "must be an integer."
            )

        if value not in domain:
            raise ValueError(
                f"Compact feature {COMPACT_FEATURE_NAMES[index]} "
                f"has invalid value {value}."
            )


def _compact_danger_level(current_danger_bin: int) -> int:
    """Collapse the existing four danger bins into three levels."""

    if current_danger_bin == 0:
        return 0

    if current_danger_bin == 3:
        return 1

    return 2


def _direction_code(direction: tuple[int, int]) -> int:
    """Encode NONE, UP, RIGHT, DOWN and LEFT as integers."""

    if direction == (0, 0):
        return NONE

    return _DIRECTION_CODES[direction]


def _nearest_coin_direction(
    *,
    position: tuple[int, int],
    coins: list[tuple[int, int]],
    field: np.ndarray,
    blocked_positions: set[tuple[int, int]],
) -> int:
    """Return the first BFS step towards the nearest reachable coin."""

    if not coins:
        return NONE

    targets = set(coins)
    queue = deque([(position, (0, 0))])
    visited = {position}

    while queue:
        current, first_direction = queue.popleft()

        if current in targets:
            return _direction_code(first_direction)

        x, y = current

        for direction in DIRECTIONS:
            dx, dy = direction
            neighbor = (x + dx, y + dy)

            if neighbor in visited:
                continue

            if not _is_free_tile(
                field,
                neighbor[0],
                neighbor[1],
                blocked_positions,
            ):
                continue

            visited.add(neighbor)
            queue.append(
                (
                    neighbor,
                    direction if current == position else first_direction,
                )
            )

    return NONE


def _safe_directions_mask(
    *,
    field: np.ndarray,
    danger_map: dict,
    blocked_positions: set[tuple[int, int]],
    bombs: list[tuple[tuple[int, int], int]],
    position: tuple[int, int],
) -> int:
    """Encode directions that begin a complete time-aware escape route."""

    mask = 0

    for index, direction in enumerate(DIRECTIONS):
        if safe_escape_exists(
            field,
            danger_map,
            blocked_positions,
            bombs,
            position,
            required_first_direction=direction,
        ):
            mask |= 1 << index

    return mask