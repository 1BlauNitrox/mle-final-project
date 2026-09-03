from __future__ import annotations

from typing import TypeAlias

from .navigation import (
    DIRECTIONS,
    _blocked_positions,
    _is_free_tile,
    _nearest_coin_features,
)

StateFeatures: TypeAlias = tuple[int, int, int, int, int, int, int, int]

FEATURE_SCHEMA_VERSION = 1
FEATURE_COUNT = 8


def state_to_features(
    game_state: dict | None,
) -> StateFeatures | None:
    """Convert the game state to a hashable feature representation."""

    if game_state is None:
        return None

    field = game_state["field"]
    _, _, _, position = game_state["self"]
    x, y = position

    blocked_positions = _blocked_positions(game_state)

    free_directions = tuple(
        int(_is_free_tile(field, x + dx, y + dy, blocked_positions)) for dx, dy in DIRECTIONS
    )

    coin_features = _nearest_coin_features(position=position, coins=game_state["coins"])

    return (*free_directions, *coin_features)
