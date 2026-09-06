"""Blast danger, escape, and crate-targeting geometry for Task 2.

Blast physics mirror items.Bomb.get_blast_coords and environment.py's
update_bombs and update_explosions behavior. Blasts stop at stone walls but
continue through crates in the current framework revision.

Issue #45 covers Task 2 without opponents. The danger model therefore handles
bombs already visible in the game state but does not simulate opponents placing
new bombs.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .navigation import DIRECTIONS, _is_free_tile

BOMB_POWER = 3
BOMB_TIMER = 4
EXPLOSION_DANGEROUS_STEPS = 2

MAX_ESCAPE_SEARCH_STEPS = 10

Position = tuple[int, int]
DangerInterval = tuple[int, int]
DangerMap = dict[Position, tuple[DangerInterval, ...]]


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
) -> DangerMap:
    """Map threatened tiles to every future inclusive lethal interval.

    The framework exposes a bomb with timer ``t`` before the action after
    which it is decremented. A positive timer therefore detonates after
    ``t + 1`` arrivals; timer zero detonates after the next action. Active
    explosion-map values encode how many future arrivals remain dangerous.
    Multiple bombs may cover the same tile at different times, so retaining
    only the earliest timer would make later lethal windows disappear.
    """
    danger: DangerMap = {}

    threatened_now = np.argwhere(explosion_map > 0)
    for x, y in threatened_now:
        position = (int(x), int(y))
        _add_danger_interval(
            danger,
            position,
            start=0,
            end=int(explosion_map[position]),
        )

    for position, timer in bombs:
        detonation_time = max(int(timer), 0) + 1
        dangerous_until = detonation_time + EXPLOSION_DANGEROUS_STEPS - 1
        for coords in blast_footprint(position, field):
            _add_danger_interval(
                danger,
                coords,
                start=detonation_time,
                end=dangerous_until,
            )

    return danger


def is_safe_at_arrival(
    danger_map: DangerMap,
    position: Position,
    arrival_time: int,
) -> bool:
    """Return whether `position` is not lethal at `arrival_time` steps from now."""
    intervals = danger_map.get(position, ())
    return not any(start <= arrival_time <= end for start, end in intervals)


def danger_countdown_bin(danger_map: DangerMap, position: Position) -> int:
    """Bin the remaining safety countdown at `position` into four categories."""
    intervals = danger_map.get(position)

    if not intervals:
        return 0

    countdown = min(start for start, _end in intervals)

    if countdown <= 0:
        return 1

    if countdown <= 2:
        return 2

    return 3


def safe_direction(
    field: np.ndarray,
    danger_map: DangerMap,
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
    danger_map: DangerMap,
    blocked_positions: set[Position],
    bombs: list[tuple[Position, int]],
    start: Position,
) -> bool:
    """Return whether a genuinely safe tile is reachable without ever stepping
    onto a tile that is lethal at the time of arrival, or entering a tile
    still occupied by an undetonated bomb.

    `blocked_positions` (current bombs and opponents) is only applied to the
    first step for opponents, which are not modeled forward in time (the
    Task 2 curriculum has none, see the module docstring). Bomb occupancy is
    instead modeled for every step from `bombs` directly, matching whichever
    bomb list built `danger_map` (including a hypothetical bomb the caller
    added): a bomb tile stays non-enterable until the framework actually
    removes it, including after waits and detours, not just for one step.
    """
    bomb_occupied_until = _bomb_occupied_until(bombs)
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
            is_wait = (dx, dy) == (0, 0)

            if elapsed == 0 and not is_wait:
                if not _is_free_tile(field, nx, ny, blocked_positions):
                    continue
            elif not _is_open_tile(field, nx, ny):
                continue

            next_time = elapsed + 1

            # Waiting never "enters" a new tile, so it is exempt: an agent
            # is not evicted from the tile it just placed its own bomb on.
            if not is_wait and next_time < bomb_occupied_until.get((nx, ny), 0):
                continue

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
    blocked_positions: set[Position] | None = None,
) -> tuple[int, int, int, int]:
    """Encode the first step and path distance to a reachable bombing tile.

    Crates themselves are not traversable. The search therefore targets the
    closest reachable open tile from which the framework blast would destroy
    at least one crate. Direction is the first BFS step, not the sign of a raw
    coordinate delta through walls.
    """
    if not np.any(field == 1):
        return (0, 0, 0, 0)

    blocked = set() if blocked_positions is None else blocked_positions
    queue = deque([(position, 0, (0, 0))])
    visited = {position}

    while queue:
        current, distance, first_direction = queue.popleft()

        if crates_destroyed_by_bomb_at(current, field) > 0:
            return (
                1,
                first_direction[0],
                first_direction[1],
                _distance_bin(distance),
            )

        x, y = current
        for direction in DIRECTIONS:
            dx, dy = direction
            neighbor = (x + dx, y + dy)

            if neighbor in visited or not _is_free_tile(
                field,
                neighbor[0],
                neighbor[1],
                blocked,
            ):
                continue

            visited.add(neighbor)
            queue.append(
                (
                    neighbor,
                    distance + 1,
                    direction if distance == 0 else first_direction,
                )
            )

    return (0, 0, 0, 0)


def _add_danger_interval(
    danger_map: DangerMap,
    position: Position,
    *,
    start: int,
    end: int,
) -> None:
    """Add one interval and merge only overlapping/adjacent windows."""
    intervals = sorted((*danger_map.get(position, ()), (start, end)))
    merged: list[DangerInterval] = []

    for interval_start, interval_end in intervals:
        if merged and interval_start <= merged[-1][1] + 1:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, interval_end))
        else:
            merged.append((interval_start, interval_end))

    danger_map[position] = tuple(merged)


def _bomb_occupied_until(bombs: list[tuple[Position, int]]) -> dict[Position, int]:
    """Map each bomb's tile to the first arrival time it is no longer occupied.

    Mirrors `build_danger_map`'s detonation-time formula (a bomb with
    observed timer ``t`` is removed, starting its blast, after ``t + 1``
    arrivals): a tile is only enterable from that arrival time onward, not
    merely once its blast becomes safe.
    """
    occupied_until: dict[Position, int] = {}

    for position, timer in bombs:
        detonation_time = max(int(timer), 0) + 1
        occupied_until[position] = max(occupied_until.get(position, 0), detonation_time)

    return occupied_until


def _in_bounds(field: np.ndarray, x: int, y: int) -> bool:
    return 0 <= x < field.shape[0] and 0 <= y < field.shape[1]


def _is_open_tile(field: np.ndarray, x: int, y: int) -> bool:
    return _in_bounds(field, x, y) and bool(field[x, y] == 0)


def _distance_bin(distance: int) -> int:
    if distance <= 1:
        return 1

    if distance <= 3:
        return 2

    return 3
