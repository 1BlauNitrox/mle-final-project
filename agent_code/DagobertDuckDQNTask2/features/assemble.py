"""Documented Task 2 feature ordering, typing, and normalization.

Indices 0-7 are the unchanged Task 1 navigation prefix (see `navigation.py`);
issue #43's differential tests pin their values identical to the frozen
parent on Task 1 states. Indices 8-20 are the Task 2 additions from
`bombs_and_crates.py`. Appending rather than interleaving keeps the parent
comparison exact and keeps this ordering documented in one place.
"""

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
)
from .navigation import DIRECTIONS, _blocked_positions, _is_free_tile, _nearest_coin_features

StateFeatures: TypeAlias = tuple[int, ...]

MAX_CRATES_DESTROYED_BIN = 3


def state_to_features(game_state: dict | None) -> StateFeatures | None:
    """Encode a framework game state as the 21-element Task 2 feature tuple."""
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

    danger_map_with_hypothetical_bomb = build_danger_map(
        field,
        # A bomb placed by the current action is decremented once by the
        # framework before the next observable state.
        [*game_state.get("bombs", []), (position, BOMB_TIMER - 1)],
        game_state["explosion_map"],
    )

    escape_after_bomb = int(
        safe_escape_exists(field, danger_map_with_hypothetical_bomb, blocked_positions, position)
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
    )


def normalize_features(features: StateFeatures) -> np.ndarray:
    """Convert raw features to the float32 input expected by the network."""
    values = np.asarray(features, dtype=np.float32)

    if values.shape != (FEATURE_COUNT,):
        raise ValueError(f"Expected {FEATURE_COUNT} features, got shape {values.shape}")

    normalized = values.copy()

    for index in (7, 9, 18, 19):
        normalized[index] /= 3.0

    return normalized
