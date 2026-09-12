"""Compact Task 2 state with descriptive post-bomb escape status."""

from __future__ import annotations

from typing import TypeAlias

from .bombs_and_crates import (
    BOMB_TIMER,
    build_danger_map,
    crates_destroyed_by_bomb_at,
    safe_direction,
    safe_escape_exists,
)
from .compact import (
    COMPACT_FEATURE_DOMAINS,
    COMPACT_FEATURE_NAMES,
    COMPACT_STATE_SPACE_UPPER_BOUND,
    compact_state_to_features,
)
from .navigation import DIRECTIONS, _blocked_positions

CompactEscapeStateFeatures: TypeAlias = tuple[int, ...]

COMPACT_ESCAPE_FEATURE_SCHEMA_VERSION = 1
COMPACT_ESCAPE_FEATURE_NAMES = (
    *COMPACT_FEATURE_NAMES,
    "escape_after_bomb_status",
)
COMPACT_ESCAPE_FEATURE_DOMAINS = (
    *COMPACT_FEATURE_DOMAINS,
    frozenset(range(4)),
)
COMPACT_ESCAPE_FEATURE_COUNT = len(COMPACT_ESCAPE_FEATURE_NAMES)
COMPACT_ESCAPE_STATE_SPACE_UPPER_BOUND = COMPACT_STATE_SPACE_UPPER_BOUND * 4

NOT_APPLICABLE = 0
COMPLETE_ROUTE = 1
TEMPORARY_SAFETY_ONLY = 2
NO_ROUTE = 3


def compact_escape_state_to_features(
    game_state: dict | None,
) -> CompactEscapeStateFeatures | None:
    """Append a post-bomb escape category to the existing compact state."""
    compact = compact_state_to_features(game_state)
    if compact is None or game_state is None:
        return None

    features = (*compact, post_bomb_escape_status(game_state))
    validate_compact_escape_features(features)
    return features


def post_bomb_escape_status(game_state: dict) -> int:
    """Describe route quality after a useful hypothetical bomb placement."""
    field = game_state["field"]
    _, _, bomb_available, position = game_state["self"]
    if not bomb_available or crates_destroyed_by_bomb_at(position, field) == 0:
        return NOT_APPLICABLE

    bombs = [*game_state.get("bombs", []), (position, BOMB_TIMER - 1)]
    danger_map = build_danger_map(field, bombs, game_state["explosion_map"])
    blocked_positions = _blocked_positions(game_state)

    if safe_escape_exists(
        field,
        danger_map,
        blocked_positions,
        bombs,
        position,
    ):
        return COMPLETE_ROUTE

    if any(
        safe_direction(
            field,
            danger_map,
            blocked_positions,
            position,
            direction,
        )
        for direction in DIRECTIONS
    ):
        return TEMPORARY_SAFETY_ONLY

    return NO_ROUTE


def validate_compact_escape_features(
    features: CompactEscapeStateFeatures,
) -> None:
    """Validate the extended compact feature tuple."""
    if len(features) != COMPACT_ESCAPE_FEATURE_COUNT:
        raise ValueError(
            f"Expected {COMPACT_ESCAPE_FEATURE_COUNT} compact escape features, "
            f"got {len(features)}."
        )
    for index, (value, domain) in enumerate(
        zip(features, COMPACT_ESCAPE_FEATURE_DOMAINS, strict=True)
    ):
        if type(value) is not int:
            raise ValueError(
                f"Compact escape feature {COMPACT_ESCAPE_FEATURE_NAMES[index]} "
                "must be an integer."
            )
        if value not in domain:
            raise ValueError(
                f"Compact escape feature {COMPACT_ESCAPE_FEATURE_NAMES[index]} "
                f"has invalid value {value}."
            )
