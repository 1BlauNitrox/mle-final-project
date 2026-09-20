"""Compact state with a shared crate-and-opponent target abstraction."""

from __future__ import annotations

from .bombs_and_crates import (
    BOMB_TIMER,
    build_danger_map,
    crates_destroyed_by_bomb_at,
    safe_escape_exists,
)
from .compact import compact_state_to_features
from .navigation import _blocked_positions
from .opponents import NONE, _attack_status, _nearest_opponent_route

SHARED_TARGET_FEATURE_SCHEMA_VERSION = 1
SHARED_TARGET_FEATURE_COUNT = 6
SHARED_TARGET_STATE_REPRESENTATION = "compact_shared_target"

TARGET_NONE = 0
TARGET_CRATE = 1
TARGET_OPPONENT = 2


def shared_target_state_to_features(
    game_state: dict | None,
) -> tuple[int, ...] | None:
    """Encode one public bomb target without prescribing an action."""
    compact = compact_state_to_features(game_state)
    if compact is None or game_state is None:
        return None

    position = tuple(game_state["self"][3])
    opponents = [tuple(other[3]) for other in game_state.get("others", [])]
    opponent_direction, _ = _nearest_opponent_route(
        game_state,
        position,
        opponents,
    )

    if opponent_direction != NONE:
        target_direction = opponent_direction
        target_kind = TARGET_OPPONENT
    elif compact[3] != NONE or crates_destroyed_by_bomb_at(
        position,
        game_state["field"],
    ) > 0:
        target_direction = compact[3]
        target_kind = TARGET_CRATE
    else:
        target_direction = NONE
        target_kind = TARGET_NONE

    bomb_status = _shared_bomb_status(game_state, position, opponents)
    features = (
        compact[0],
        compact[1],
        compact[2],
        target_direction,
        target_kind,
        bomb_status,
    )
    validate_shared_target_features(features)
    return features


def validate_shared_target_features(features: tuple[int, ...]) -> None:
    """Validate the six-value shared-target representation."""
    if len(features) != SHARED_TARGET_FEATURE_COUNT:
        raise ValueError(
            f"Expected {SHARED_TARGET_FEATURE_COUNT} shared-target features, "
            f"got {len(features)}."
        )
    domains = (
        {0, 1, 2},
        set(range(16)),
        set(range(5)),
        set(range(5)),
        {TARGET_NONE, TARGET_CRATE, TARGET_OPPONENT},
        {0, 1, 2, 3},
    )
    for index, (value, domain) in enumerate(zip(features, domains, strict=True)):
        if type(value) is not int or value not in domain:
            raise ValueError(f"Invalid shared-target feature {index}: {value!r}")
    if features[4] == TARGET_NONE and features[3] != NONE:
        raise ValueError("A target direction requires a target kind")


def _shared_bomb_status(
    game_state: dict,
    position: tuple[int, int],
    opponents: list[tuple[int, int]],
) -> int:
    if not game_state["self"][2]:
        return 0

    field = game_state["field"]
    useful = (
        crates_destroyed_by_bomb_at(position, field) > 0
        or _attack_status(game_state, position, opponents) == 2
    )
    if not useful:
        return 1

    bombs = game_state.get("bombs", [])
    hypothetical_bombs = [*bombs, (position, BOMB_TIMER - 1)]
    danger_map = build_danger_map(
        field,
        hypothetical_bombs,
        game_state["explosion_map"],
    )
    safe_escape = safe_escape_exists(
        field,
        danger_map,
        _blocked_positions(game_state),
        hypothetical_bombs,
        position,
    )
    return 3 if safe_escape else 2
