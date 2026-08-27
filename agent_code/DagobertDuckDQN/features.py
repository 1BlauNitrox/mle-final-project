"""Feature extraction and normalization for the Task 1 DQN"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np

from .config import FEATURE_COUNT

StateFeatures: TypeAlias = tuple[int, int, int, int, int, int, int, int]

DIRECTIONS: tuple[tuple[int, int], ...] = (
    (0, -1), #UP
    (1, 0), #RIGHT
    (0, 1), #DOWN
    (-1, 0), #LEFT
)

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

def _blocked_positions(game_state: dict) -> set[tuple[int, int]]:
    """Return positions that cannot be entered currently"""
    bomb_positions = {
        position for position, _timer in game_state.get("bombs", [])
    }
    opponent_positions = {
        opponent[3] for opponent in game_state.get("others", [])
    }

    return bomb_positions | opponent_positions

def _is_free_tile(
    field: np.ndarray,
    x: int,
    y: int,
    blocked_positions: set[tuple[int, int]],
) -> bool:
    """Return wether a position is inside the board and reachable"""
    if x < 0 or y < 0:
        return False
    
    if x >= field.shape[0] or y >= field.shape[1]:
        return False
    
    return bool(field[x, y] == 0 and (x, y) not in blocked_positions)

def _nearest_coin_features(
    *,
    position: tuple[int, int],
    coins: list[tuple[int, int]],
) -> tuple[int, int, int, int]:
    """Encode visibility, direction and distance of the nearest coin"""
    if not coins:
        return (0, 0, 0, 0)
    
    x, y = position
    nearest_coin = min(
        coins, 
        key=lambda coin: (
            _manhattan_distance(position, coin),
            coin[0],
            coin[1],
        ),
    )

    coin_x, coin_y = nearest_coin
    distance = _manhattan_distance(position, nearest_coin)

    return(
        1,
        _sign(coin_x - x),
        _sign(coin_y - y),
        _distance_bin(distance),
    )

def _manhattan_distance(
    first: tuple[int, int],
    second: tuple[int, int],
) -> int: 
    """Calculate Manhattan distance between two positions"""
    return abs(first[0] - second[0]) + abs(first[1] - second[1])

def _sign(value: int) -> int:
    """Return -1, 0 or 1 for a signed integer."""
    if value > 0:
        return 1
    
    if value < 0:
        return -1
    
    return 0

def _distance_bin(distance: int) -> int:
    """Map Manhattan distance to the controlled Task 1 bins"""
    if distance <= 1:
        return 1
    
    if distance <= 3:
        return 2
    
    return 3
