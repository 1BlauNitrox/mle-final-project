"""Task 3 schema 4: unchanged 26-input Task 2 prefix plus 13 opponent inputs."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np

from ..config import FEATURE_COUNT
from .bombs_and_crates import (
    BOMB_TIMER,
    build_danger_map,
    crates_destroyed_by_bomb_at,
    danger_countdown_bin,
    nearest_crate_features,
    safe_direction,
    safe_escape_exists,
    surviving_continuation_after_action,
)
from .navigation import (
    DIRECTIONS,
    _blocked_positions,
    _is_free_tile,
    _nearest_coin_features,
)
from .opponents import opponent_features

StateFeatures: TypeAlias = tuple[int, ...]

MAX_CRATES_DESTROYED_BIN = 3


def state_to_features(
    game_state: dict | None, *, include_continuation_features: bool = True
) -> StateFeatures | None:
    """Encode a framework state as the 39-element Task 3 feature tuple."""
    if game_state is None:
        return None

    field = game_state["field"]
    _, _, bomb_available, position = game_state["self"]
    x, y = position

    blocked_positions = _blocked_positions(game_state)

    free_directions = tuple(
        int(_is_free_tile(field, x + dx, y + dy, blocked_positions)) for dx, dy in DIRECTIONS
    )

    coin_features = _nearest_coin_features(position=position, coins=game_state["coins"])

    danger_map = build_danger_map(
        field,
        game_state.get("bombs", []),
        game_state["explosion_map"],
    )

    safe_directions = tuple(
        int(safe_direction(field, danger_map, blocked_positions, position, direction))
        for direction in DIRECTIONS
    )

    if include_continuation_features:
        continuation_features = tuple(
            int(
                surviving_continuation_after_action(
                    field,
                    danger_map,
                    blocked_positions,
                    game_state.get("bombs", []),
                    position,
                    direction,
                )
            )
            for direction in (*DIRECTIONS, (0, 0))
        )
    else:
        continuation_features = (0, 0, 0, 0, 0)

    # A bomb placed by the current action is decremented once by the
    # framework before the next observable state.
    bombs_with_hypothetical = [*game_state.get("bombs", []), (position, BOMB_TIMER - 1)]

    danger_map_with_hypothetical_bomb = build_danger_map(
        field,
        bombs_with_hypothetical,
        game_state["explosion_map"],
    )

    escape_after_bomb = int(
        safe_escape_exists(
            field,
            danger_map_with_hypothetical_bomb,
            blocked_positions,
            bombs_with_hypothetical,
            position,
        )
    )

    crate_features = nearest_crate_features(
        position=position,
        field=field,
        blocked_positions=blocked_positions,
    )

    crates_here = min(
        crates_destroyed_by_bomb_at(position, field),
        MAX_CRATES_DESTROYED_BIN,
    )
    bomb_has_useful_target = int(crates_here > 0)

    opponent_suffix = opponent_features(
        game_state=game_state,
        field=field,
        blocked_positions=blocked_positions,
        position=position,
        danger_map_with_hypothetical_bomb=danger_map_with_hypothetical_bomb,
        bombs_with_hypothetical=bombs_with_hypothetical,
    )

    return (
        *free_directions,
        *coin_features,
        int(bomb_available),
        danger_countdown_bin(danger_map, position),
        *safe_directions,
        escape_after_bomb,
        *crate_features,
        crates_here,
        bomb_has_useful_target,
        *continuation_features,
        *opponent_suffix,
    )


def normalize_features(features: StateFeatures) -> np.ndarray:
    """Convert raw features to the float32 input expected by the network."""
    values = np.asarray(features, dtype=np.float32)

    if values.shape != (FEATURE_COUNT,):
        raise ValueError(f"Expected {FEATURE_COUNT} features, got shape {values.shape}")

    normalized = values.copy()

    for index in (7, 9, 18, 19, 29, 38):
        normalized[index] /= 3.0

    return normalized
