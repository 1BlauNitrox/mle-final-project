from __future__ import annotations

from typing import TypeAlias

from .bombs_and_crates import (
    BOMB_TIMER,
    build_danger_map,
    crates_destroyed_by_bomb_at,
    danger_countdown_bin,
    nearest_crate_features,
    safe_direction,
    safe_escape_exists,
)
from .navigation import (
    DIRECTIONS,
    _blocked_positions,
    _is_free_tile,
    _nearest_coin_features,
)

StateFeatures: TypeAlias = tuple[int, ...]

FEATURE_SCHEMA_VERSION = 2

FEATURE_NAMES: tuple[str, ...] = (
    "free_up",
    "free_right",
    "free_down",
    "free_left",
    "coin_visible",
    "coin_dx",
    "coin_dy",
    "coin_distance_bin",
    "bomb_available",
    "current_danger_bin",
    "safe_direction_mask",
    "escape_after_bomb",
    "crate_visible",
    "crate_dx",
    "crate_dy",
    "crate_distance_bin",
    "crates_in_current_blast_bin",
)

FEATURE_DOMAINS: tuple[frozenset[int], ...] = (
    frozenset({0, 1}),
    frozenset({0, 1}),
    frozenset({0, 1}),
    frozenset({0, 1}),
    frozenset({0, 1}),
    frozenset({-1, 0, 1}),
    frozenset({-1, 0, 1}),
    frozenset({0, 1, 2, 3}),
    frozenset({0, 1}),
    frozenset({0, 1, 2, 3}),
    frozenset(range(16)),
    frozenset({0, 1}),
    frozenset({0, 1}),
    frozenset({-1, 0, 1}),
    frozenset({-1, 0, 1}),
    frozenset({0, 1, 2, 3}),
    frozenset({0, 1, 2, 3}),
)

FEATURE_COUNT = len(FEATURE_NAMES)

THEORETICAL_STATE_SPACE_UPPER_BOUND = 84_934_656

MAX_CRATES_DESTROYED_BIN = 3

if len(FEATURE_DOMAINS) != FEATURE_COUNT:
    raise RuntimeError("Feature names and domains must have equal lengths.")


def validate_features(features: StateFeatures) -> None:
    """Validate feature count, types, domains, and dependencies."""

    if len(features) != FEATURE_COUNT:
        raise ValueError(f"Expected {FEATURE_COUNT} features, got {len(features)}.")

    for index, (value, domain) in enumerate(zip(features, FEATURE_DOMAINS, strict=True)):
        if type(value) is not int:
            raise ValueError(f"Feature {FEATURE_NAMES[index]} must be an integer.")

        if value not in domain:
            raise ValueError(
                f"Feature {FEATURE_NAMES[index]} has invalid value {value}; "
                f"expected one of {sorted(domain)}."
            )

    free_directions = features[:4]
    safe_direction_mask = features[10]

    for index, free in enumerate(free_directions):
        direction_is_safe = bool(safe_direction_mask & (1 << index))

        if direction_is_safe and free == 0:
            raise ValueError(f"Blocked direction {FEATURE_NAMES[index]} cannot be marked as safe.")

    coin_visible = features[4]
    coin_dx = features[5]
    coin_dy = features[6]
    coin_distance_bin = features[7]

    if coin_visible == 0 and (coin_dx != 0 or coin_dy != 0 or coin_distance_bin != 0):
        raise ValueError("Missing coins must have neutral direction and distance features.")

    crate_visible = features[12]
    crate_dx = features[13]
    crate_dy = features[14]
    crate_distance_bin = features[15]

    if crate_visible == 0 and (crate_dx != 0 or crate_dy != 0 or crate_distance_bin != 0):
        raise ValueError("Missing crate targets must have neutral direction and distance features.")


def _direction_mask(
    values: tuple[bool, bool, bool, bool],
) -> int:
    """Encode UP, RIGHT, DOWN and LEFT as a four-bit mask."""

    mask = 0

    for index, value in enumerate(values):
        if value:
            mask |= 1 << index

    return mask


def state_to_features(
    game_state: dict | None,
) -> StateFeatures | None:
    """Convert a framework game state to the Task 2 feature tuple."""

    if game_state is None:
        return None

    field = game_state["field"]
    _, _, bomb_available, position = game_state["self"]
    x, y = position

    bombs = game_state.get("bombs", [])
    explosion_map = game_state["explosion_map"]
    blocked_positions = _blocked_positions(game_state)

    # Indices 0-3: unchanged Task 1 movement features.
    free_directions = tuple(
        int(
            _is_free_tile(
                field,
                x + dx,
                y + dy,
                blocked_positions,
            )
        )
        for dx, dy in DIRECTIONS
    )

    # Indices 4-7: unchanged Task 1 coin features.
    coin_features = _nearest_coin_features(
        position=position,
        coins=game_state["coins"],
    )

    danger_map = build_danger_map(
        field,
        bombs,
        explosion_map,
    )

    safe_directions = tuple(
        safe_direction(
            field,
            danger_map,
            blocked_positions,
            position,
            direction,
        )
        for direction in DIRECTIONS
    )

    safe_direction_mask = _direction_mask(safe_directions)

    # A newly placed bomb is decremented by the framework before the next
    # observable state. Therefore its hypothetical visible timer is
    # BOMB_TIMER - 1.
    if bomb_available:
        hypothetical_bombs = [
            *bombs,
            (position, BOMB_TIMER - 1),
        ]

        hypothetical_danger_map = build_danger_map(
            field,
            hypothetical_bombs,
            explosion_map,
        )

        escape_after_bomb = int(
            safe_escape_exists(
                field,
                hypothetical_danger_map,
                blocked_positions,
                hypothetical_bombs,
                position,
            )
        )
    else:
        escape_after_bomb = 0

    crate_features = nearest_crate_features(
        position=position,
        field=field,
        blocked_positions=blocked_positions,
    )

    crates_in_current_blast_bin = min(
        crates_destroyed_by_bomb_at(position, field),
        MAX_CRATES_DESTROYED_BIN,
    )

    features: StateFeatures = (
        *free_directions,
        *coin_features,
        int(bomb_available),
        danger_countdown_bin(danger_map, position),
        safe_direction_mask,
        escape_after_bomb,
        *crate_features,
        crates_in_current_blast_bin,
    )

    validate_features(features)

    return features
