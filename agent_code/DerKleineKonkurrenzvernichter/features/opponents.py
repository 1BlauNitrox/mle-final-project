"""Opponent-aware extension of the frozen compact Task 2 state."""

from __future__ import annotations

from collections import deque

from .compact import compact_state_to_features, validate_compact_features
from .navigation import DIRECTIONS, _blocked_positions, _is_free_tile

OPPONENT_FEATURE_SCHEMA_VERSION = 1
OPPONENT_FEATURE_COUNT = 8
OPPONENT_STATE_REPRESENTATION = "compact_opponent"

NONE = 0
UP = 1
RIGHT = 2
DOWN = 3
LEFT = 4


def opponent_state_to_features(game_state: dict | None) -> tuple[int, ...] | None:
    """Append learned hunting context to the five-value Task 2 state."""
    base = compact_state_to_features(game_state)
    if base is None or game_state is None:
        return None

    opponents = [tuple(other[3]) for other in game_state.get("others", [])]
    if not opponents:
        features = (*base, NONE, 0, 0)
        validate_opponent_features(features)
        return features

    position = tuple(game_state["self"][3])
    direction, distance = _nearest_opponent_route(game_state, position, opponents)
    attack_status = _attack_status(game_state, position, opponents)
    features = (*base, direction, _distance_bin(distance), attack_status)
    validate_opponent_features(features)
    return features


def validate_opponent_features(features: tuple[int, ...]) -> None:
    """Validate the compact Task 3 representation."""
    if len(features) != OPPONENT_FEATURE_COUNT:
        raise ValueError(
            f"Expected {OPPONENT_FEATURE_COUNT} opponent-aware features, "
            f"got {len(features)}."
        )
    validate_compact_features(features[:5])
    domains = ({0, 1, 2, 3, 4}, {0, 1, 2, 3}, {0, 1, 2})
    for index, (value, domain) in enumerate(zip(features[5:], domains, strict=True), 5):
        if type(value) is not int or value not in domain:
            raise ValueError(f"Invalid opponent feature {index}: {value!r}")


def _nearest_opponent_route(
    game_state: dict,
    start: tuple[int, int],
    opponents: list[tuple[int, int]],
) -> tuple[int, int | None]:
    field = game_state["field"]
    blocked = _blocked_positions(game_state) - set(opponents)
    queue = deque([(start, NONE, 0)])
    seen = {start}
    target_set = set(opponents)

    while queue:
        position, first_direction, distance = queue.popleft()
        if position in target_set:
            return first_direction, distance
        for code, (dx, dy) in enumerate(DIRECTIONS, start=1):
            candidate = (position[0] + dx, position[1] + dy)
            if candidate in seen:
                continue
            if candidate not in target_set and not _is_free_tile(
                field, candidate[0], candidate[1], blocked
            ):
                continue
            seen.add(candidate)
            queue.append((candidate, code if distance == 0 else first_direction, distance + 1))
    return NONE, None


def _distance_bin(distance: int | None) -> int:
    if distance is None:
        return 0
    if distance <= 1:
        return 1
    if distance <= 3:
        return 2
    return 3


def _attack_status(
    game_state: dict,
    position: tuple[int, int],
    opponents: list[tuple[int, int]],
) -> int:
    if not game_state["self"][2]:
        return 0
    field = game_state["field"]
    for opponent in opponents:
        aligned = opponent[0] == position[0] or opponent[1] == position[1]
        if aligned and _clear_blast_line(field, position, opponent):
            return 2
    return 1


def _clear_blast_line(
    field,
    source: tuple[int, int],
    target: tuple[int, int],
) -> bool:
    distance = abs(source[0] - target[0]) + abs(source[1] - target[1])
    if distance > 3:
        return False
    dx = 0 if source[0] == target[0] else (1 if target[0] > source[0] else -1)
    dy = 0 if source[1] == target[1] else (1 if target[1] > source[1] else -1)
    cursor = source
    for _ in range(distance - 1):
        cursor = (cursor[0] + dx, cursor[1] + dy)
        if field[cursor] != 0:
            return False
    return True
