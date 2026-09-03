"""Documented feature ordering, typing, and normalization for Task 2."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np

from ..config import FEATURE_COUNT
from .navigation import DIRECTIONS, _blocked_positions, _is_free_tile, _nearest_coin_features

StateFeatures: TypeAlias = tuple[int, int, int, int, int, int, int, int]


def state_to_features(game_state: dict | None) -> StateFeatures | None:
    """Encode a framework game state using the controlled eight features."""
    if game_state is None:
        return None

    field = game_state["field"]
    _, _, _, position = game_state["self"]
    x, y = position

    blocked_positions = _blocked_positions(game_state)

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

    coin_features = _nearest_coin_features(
        position=position,
        coins=game_state["coins"],
    )

    return (*free_directions, *coin_features)


def normalize_features(features: StateFeatures) -> np.ndarray:
    """Convert raw features to the float32 input expected by the network"""
    values = np.asarray(features, dtype=np.float32)

    if values.shape != (FEATURE_COUNT,):
        raise ValueError(
            f"Expected {FEATURE_COUNT} features, got shape {values.shape}"
        )

    normalized = values.copy()
    normalized[7] /= 3.0

    return normalized
