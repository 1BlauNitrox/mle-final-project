"""Local movement and nearest-visible-coin geometry, split out for Task 2."""

from __future__ import annotations

import numpy as np

DIRECTIONS: tuple[tuple[int, int], ...] = (
    (0, -1),  # UP
    (1, 0),  # RIGHT
    (0, 1),  # DOWN
    (-1, 0),  # LEFT
)


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

    return (
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
