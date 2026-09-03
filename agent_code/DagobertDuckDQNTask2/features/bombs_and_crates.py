"""Blast-danger, escape, and crate-targeting geometry for the Task 2 features.

Blast physics mirror items.Bomb.get_blast_coords and environment.py's
update_bombs/update_explosions exactly (stops only at walls, not crates;
observed bomb timers count down to explosion; an explosion then lingers for
EXPLOSION_TIMER additional steps), so the constants below must track
settings.py/items.py if the framework ever changes them.

There are no opponents in the Task 2 curriculum (see #44's problem
statement), so the danger model only accounts for bombs already on the
board -- it does not simulate an opponent placing a new one.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .navigation import DIRECTIONS, _is_free_tile

BOMB_POWER = 3
BOMB_TIMER = 4
EXPLOSION_LINGER_STEPS = 2

MAX_ESCAPE_SEARCH_STEPS = 10

Position = tuple[int, int]


def blast_footprint(position: Position, field: np.ndarray) -> list[Position]:
    """Reproduce items.Bomb.get_blast_coords: blocked only by walls."""
    x, y = position
    coords = [(x, y)]

    for dx, dy in DIRECTIONS:
        for step in range(1, BOMB_POWER + 1):
            nx, ny = x + dx * step, y + dy * step

            if not _in_bounds(field, nx, ny) or field[nx, ny] == -1:
                break

            coords.append((nx, ny))

    return coords


def build_danger_map(
    field: np.ndarray,
    bombs: list[tuple[Position, int]],
    explosion_map: np.ndarray,
) -> dict[Position, int]:
    """Map each threatened tile to the bomb timer that will detonate it.

    A missing key means the tile is not threatened by any bomb currently on
    the board. A value of `0` (or a negative reading from an active
    explosion) means the tile is already lethal.
    """
    danger: dict[Position, int] = {}

    threatened_now = np.argwhere(explosion_map > 0)
    for x, y in threatened_now:
        danger[(int(x), int(y))] = 0

    for position, timer in bombs:
        for coords in blast_footprint(position, field):
            existing = danger.get(coords)

            if existing is None or timer < existing:
                danger[coords] = timer

    return danger


def is_safe_at_arrival(
    danger_map: dict[Position, int],
    position: Position,
    arrival_time: int,
) -> bool:
    """Return whether `position` is not lethal at `arrival_time` steps from now."""
    countdown = danger_map.get(position)

    if countdown is None:
        return True

    return not (countdown <= arrival_time <= countdown + EXPLOSION_LINGER_STEPS)


def danger_countdown_bin(danger_map: dict[Position, int], position: Position) -> int:
    """Bin the remaining safety countdown at `position` into four categories."""
    countdown = danger_map.get(position)

    if countdown is None:
        return 0

    if countdown <= 0:
        return 1

    if countdown <= 2:
        return 2

    return 3


def safe_direction(
    field: np.ndarray,
    danger_map: dict[Position, int],
    blocked_positions: set[Position],
    position: Position,
    direction: tuple[int, int],
) -> bool:
    """Return whether moving one step in `direction` is enterable and safe next turn.

    "Enterable" matches `navigation._is_free_tile` exactly (field openness and
    the current bomb/opponent occupancy), so this reduces to the Task 1
    free-direction check whenever nothing is dangerous.
    """
    x, y = position
    dx, dy = direction
    nx, ny = x + dx, y + dy

    if not _is_free_tile(field, nx, ny, blocked_positions):
        return False

    return is_safe_at_arrival(danger_map, (nx, ny), 1)


def safe_escape_exists(
    field: np.ndarray,
    danger_map: dict[Position, int],
    blocked_positions: set[Position],
    start: Position,
) -> bool:
    """Return whether a genuinely safe tile is reachable without ever stepping
    onto a tile that is lethal at the time of arrival.

    `blocked_positions` (current bombs and opponents) is only applied to the
    first step: it reflects a snapshot that itself changes as bombs tick down
    and, unlike blast danger, is not modeled forward in time.
    """
    moves = (*DIRECTIONS, (0, 0))
    visited = {(start, 0)}
    queue = deque([(start, 0)])

    while queue:
        (x, y), elapsed = queue.popleft()

        if elapsed > 0 and (x, y) not in danger_map:
            return True

        if elapsed >= MAX_ESCAPE_SEARCH_STEPS:
            continue

        for dx, dy in moves:
            nx, ny = x + dx, y + dy

            if elapsed == 0 and (dx, dy) != (0, 0):
                if not _is_free_tile(field, nx, ny, blocked_positions):
                    continue
            elif not _is_open_tile(field, nx, ny):
                continue

            next_time = elapsed + 1

            if not is_safe_at_arrival(danger_map, (nx, ny), next_time):
                continue

            state = ((nx, ny), next_time)

            if state in visited:
                continue

            visited.add(state)
            queue.append(state)

    return False


def crates_destroyed_by_bomb_at(position: Position, field: np.ndarray) -> int:
    """Count crates that a bomb placed at `position` would destroy."""
    return sum(1 for coords in blast_footprint(position, field) if field[coords] == 1)


def nearest_crate_features(
    *,
    position: Position,
    field: np.ndarray,
) -> tuple[int, int, int, int]:
    """Encode visibility, direction, and distance of the nearest crate."""
    crate_positions = list(zip(*np.where(field == 1), strict=True))

    if not crate_positions:
        return (0, 0, 0, 0)

    x, y = position
    nearest_crate = min(
        crate_positions,
        key=lambda crate: (_manhattan_distance(position, crate), crate[0], crate[1]),
    )

    crate_x, crate_y = nearest_crate
    distance = _manhattan_distance(position, nearest_crate)

    return (
        1,
        _sign(crate_x - x),
        _sign(crate_y - y),
        _distance_bin(distance),
    )


def _in_bounds(field: np.ndarray, x: int, y: int) -> bool:
    return 0 <= x < field.shape[0] and 0 <= y < field.shape[1]


def _is_open_tile(field: np.ndarray, x: int, y: int) -> bool:
    return _in_bounds(field, x, y) and bool(field[x, y] == 0)


def _manhattan_distance(first: Position, second: Position) -> int:
    return abs(first[0] - second[0]) + abs(first[1] - second[1])


def _sign(value: int) -> int:
    if value > 0:
        return 1

    if value < 0:
        return -1

    return 0


def _distance_bin(distance: int) -> int:
    if distance <= 1:
        return 1

    if distance <= 3:
        return 2

    return 3
