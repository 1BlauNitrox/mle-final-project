"""Correctness and bounded-latency tests for the Issue #87 features."""

from __future__ import annotations

from collections import deque
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from agent_code.DagobertDuckDQNTask2.config import DQNConfig
from agent_code.DagobertDuckDQNTask2.features import state_to_features
from agent_code.DagobertDuckDQNTask2.features.bombs_and_crates import (
    DIRECTIONS,
    MAX_ESCAPE_SEARCH_STEPS,
    build_danger_map,
    is_safe_at_arrival,
    surviving_continuation_after_action,
)
from agent_code.DagobertDuckDQNTask2.model import build_q_network


def _field(open_tiles: set[tuple[int, int]], shape: tuple[int, int] = (9, 9)) -> np.ndarray:
    field = np.full(shape, -1, dtype=int)
    for position in open_tiles:
        field[position] = 0
    return field


def _oracle(
    field: np.ndarray,
    danger_map: dict,
    blocked_positions: set[tuple[int, int]],
    bombs: list[tuple[tuple[int, int], int]],
    start: tuple[int, int],
    direction: tuple[int, int],
) -> bool:
    """Small exhaustive trajectory oracle for the bounded continuation rule."""
    x, y = start
    wait = direction == (0, 0)
    candidate = start if wait else (x + direction[0], y + direction[1])

    def open_tile(position: tuple[int, int]) -> bool:
        px, py = position
        return (
            0 <= px < field.shape[0]
            and 0 <= py < field.shape[1]
            and field[position] == 0
        )

    if not wait and (
        not open_tile(candidate) or candidate in blocked_positions
    ):
        return False
    if not is_safe_at_arrival(danger_map, candidate, 1):
        return False

    occupied_until = {
        position: max(int(timer), 0) + 1 for position, timer in bombs
    }
    queue = deque([(candidate, 1)])
    visited = {(candidate, 1)}
    moves = (*DIRECTIONS, (0, 0))

    while queue:
        position, elapsed = queue.popleft()
        if elapsed >= MAX_ESCAPE_SEARCH_STEPS:
            return True
        for move in moves:
            is_wait = move == (0, 0)
            next_position = (position[0] + move[0], position[1] + move[1])
            next_time = elapsed + 1
            if not open_tile(next_position):
                continue
            if (
                not is_wait
                and next_time < occupied_until.get(next_position, 0)
            ):
                continue
            if not is_safe_at_arrival(danger_map, next_position, next_time):
                continue
            state = (next_position, next_time)
            if state not in visited:
                visited.add(state)
                queue.append(state)
    return False


def test_next_step_safe_dead_end_is_not_a_surviving_continuation() -> None:
    field = _field({(2, 2), (3, 2)})
    danger_map = {(2, 2): ((2, 10),), (3, 2): ((3, 10),)}

    assert is_safe_at_arrival(danger_map, (3, 2), 1) is True
    assert surviving_continuation_after_action(
        field, danger_map, set(), [], (2, 2), (1, 0)
    ) is False


def test_safe_first_action_with_a_surviving_branch_is_true() -> None:
    field = _field({(2, 2), (3, 2), (3, 3)})
    danger_map = {(3, 2): ((3, 10),)}

    assert surviving_continuation_after_action(
        field, danger_map, set(), [], (2, 2), (1, 0)
    ) is True


def test_wait_on_own_bomb_can_be_required_before_the_exit_opens() -> None:
    open_tiles = {(4, 4), (5, 4), (5, 3)}
    field = _field(open_tiles, shape=(7, 7))
    explosion_map = np.zeros_like(field)
    explosion_map[5, 4] = 1
    bombs = [((4, 4), 5)]
    danger_map = build_danger_map(field, bombs, explosion_map)

    assert surviving_continuation_after_action(
        field, danger_map, {(4, 4)}, bombs, (4, 4), (0, 0)
    ) is True


def test_live_bomb_blocks_reentry_even_when_its_blast_window_is_later() -> None:
    open_tiles = {(1, 1), (2, 1), (3, 1), (3, 2)}
    field = _field(open_tiles, shape=(6, 5))
    bombs = [((3, 1), 4)]
    danger_map = build_danger_map(field, bombs, np.zeros_like(field))

    # RIGHT is safe on arrival at (2,1), but the only eventual exit crosses
    # the still-live bomb and waiting at (2,1) reaches its later blast.
    assert surviving_continuation_after_action(
        field, danger_map, {(3, 1)}, bombs, (1, 1), (1, 0)
    ) is False


def test_overlapping_blast_windows_are_honored_by_continuation_search() -> None:
    open_tiles = {(5, 2), (5, 3), (5, 4)}
    field = _field(open_tiles, shape=(9, 7))
    bombs = [((5, 1), 1), ((5, 5), 4)]
    danger_map = build_danger_map(field, bombs, np.zeros_like(field))

    assert danger_map[(5, 3)] == ((2, 3), (5, 6))
    assert surviving_continuation_after_action(
        field, danger_map, set(), bombs, (5, 2), (1, 0)
    ) is False


def test_no_danger_marks_legal_actions_and_no_escape_marks_none() -> None:
    game_state = {
        "field": _field({(3, 3), (4, 3)}, shape=(7, 7)),
        "self": ("agent", 0, True, (3, 3)),
        "coins": [],
        "bombs": [],
        "others": [],
        "explosion_map": np.zeros((7, 7), dtype=int),
    }
    features = state_to_features(game_state)
    assert features is not None
    assert features[21:] == (0, 1, 0, 0, 1)

    game_state["explosion_map"][3, 3] = 10
    game_state["explosion_map"][4, 3] = 10
    features = state_to_features(game_state)
    assert features is not None
    assert features[21:] == (0, 0, 0, 0, 0)


def test_feature_off_treatment_uses_neutral_zero_columns() -> None:
    game_state = {
        "field": _field({(3, 3), (4, 3)}, shape=(7, 7)),
        "self": ("agent", 0, True, (3, 3)),
        "coins": [],
        "bombs": [],
        "others": [],
        "explosion_map": np.zeros((7, 7), dtype=int),
    }
    features = state_to_features(game_state, include_continuation_features=False)
    assert features is not None
    assert features[21:] == (0, 0, 0, 0, 0)


def test_feature_search_matches_exhaustive_trajectory_oracle() -> None:
    field = _field({(2, 2), (3, 2), (3, 3), (2, 3), (1, 2)}, shape=(6, 6))
    bombs = [((3, 3), 2), ((1, 4), 4)]
    danger_map = build_danger_map(field, bombs, np.zeros_like(field))
    start = (2, 2)
    blocked = {(3, 3)}

    for direction in (*DIRECTIONS, (0, 0)):
        expected = _oracle(field, danger_map, blocked, bombs, start, direction)
        actual = surviving_continuation_after_action(
            field, danger_map, blocked, bombs, start, direction
        )
        assert actual is expected


def test_blast_geometry_matches_the_unchanged_framework() -> None:
    from items import Bomb

    field = np.zeros((11, 11), dtype=int)
    field[5, 8] = -1
    expected = Bomb(
        (5, 5), SimpleNamespace(), 4, 3, None
    ).get_blast_coords(field)

    from agent_code.DagobertDuckDQNTask2.features.bombs_and_crates import blast_footprint

    assert set(blast_footprint((5, 5), field)) == set(expected)


def test_full_act_feature_latency_has_a_bounded_smoke_margin() -> None:
    from types import SimpleNamespace

    import agent_code.DagobertDuckDQNTask2.callbacks as callbacks

    config = DQNConfig(escape_continuation_features=True)
    agent = SimpleNamespace(
        config=config,
        policy_network=build_q_network(config, seed=87),
        action_rng=np.random.default_rng(87),
        epsilon=0.0,
        train=False,
    )
    game_state = {
        "field": _field({(3, 3), (4, 3), (3, 2), (3, 4)}, shape=(7, 7)),
        "self": ("agent", 0, True, (3, 3)),
        "coins": [],
        "bombs": [((4, 3), 3)],
        "others": [],
        "explosion_map": np.zeros((7, 7), dtype=int),
    }

    for _ in range(20):
        callbacks.act(agent, game_state)
    durations_ms = []
    for _ in range(100):
        start = perf_counter()
        callbacks.act(agent, game_state)
        durations_ms.append((perf_counter() - start) * 1000.0)

    p95 = float(np.percentile(durations_ms, 95))
    assert p95 < 50.0
    assert max(durations_ms) < 100.0
