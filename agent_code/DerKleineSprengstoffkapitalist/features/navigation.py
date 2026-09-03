from __future__ import annotations

import numpy as np

DIRECTIONS: tuple[tuple[int, int], ...] = (
    (0, -1),
    (1, 0),
    (0, 1),
    (-1, 0),
)  # UP, RIGHT, DOWN, LEFT


def _blocked_positions(game_state: dict) -> set[tuple[int, int]]:
    """Return a set of positions that can't be entered."""

    bomb_positions = {position for position, _timer in game_state.get("bombs", [])}

    opponent_positions = {opponent[3] for opponent in game_state.get("others", [])}

    return bomb_positions | opponent_positions


def _is_free_tile(
    field: np.ndarray, x: int, y: int, blocked_positions: set[tuple[int, int]]
) -> bool:
    """Check if a tile is free to move into."""

    if x < 0 or y < 0:
        return False

    if x >= field.shape[0] or y >= field.shape[1]:
        return False

    return field[x, y] == 0 and (x, y) not in blocked_positions


def _nearest_coin_features(
    *, position: tuple[int, int], coins: list[tuple[int, int]]
) -> tuple[int, int, int, int]:
    """Return features for the nearest coin."""

    if not coins:
        return (0, 0, 0, 0)

    x, y = position

    nearest_coin = min(
        coins, key=lambda coin: (_manhatten_distance(position, coin), coin[0], coin[1])
    )

    coin_x, coin_y = nearest_coin
    dx = _sign(coin_x - x)
    dy = _sign(coin_y - y)
    distance = _manhatten_distance(position, nearest_coin)

    return (1, dx, dy, _distance_bin(distance))


def _manhatten_distance(pos1: tuple[int, int], pos2: tuple[int, int]) -> int:
    """Calculate the Manhattan distance between two positions."""

    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def _sign(value: int) -> int:
    """Return the sign of a value."""

    if value > 0:
        return 1
    elif value < 0:
        return -1
    else:
        return 0


def _distance_bin(distance: int) -> int:
    """Return a binned representation of the distance."""

    if distance <= 1:
        return 1
    elif distance <= 3:
        return 2
    else:
        return 3
