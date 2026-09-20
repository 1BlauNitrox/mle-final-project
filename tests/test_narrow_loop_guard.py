"""Contracts for the conservative 24-step late-game loop guard."""

import numpy as np
import pytest

from agent_code.DagobertDuckDQNAntiLoop.callbacks import (
    DEFAULT_NARROW_LOOP_GUARD_MODE,
    _loop_guard_window,
)
from agent_code.DagobertDuckDQNAntiLoop.config import ACTIONS
from agent_code.DagobertDuckDQNAntiLoop.narrow_loop_guard import NarrowLoopGuard


def state(step, position, *, crates=False, coins=((7, 7),), others=(), bombs=(), score=0):
    field = np.zeros((9, 9), dtype=int)
    field[[0, -1], :] = -1
    field[:, [0, -1]] = -1
    if crates:
        field[2, 2] = 1
    return {
        "round": 1,
        "step": step,
        "field": field,
        "self": ("me", score, True, position),
        "others": list(others),
        "coins": list(coins),
        "bombs": list(bombs),
        "explosion_map": np.zeros_like(field),
    }


def q(**values):
    return lambda: np.asarray([values.get(action, 0.0) for action in ACTIONS])


def pacing(**kwargs):
    guard = NarrowLoopGuard()
    for index in range(1, 25):
        position = (4, 4) if index % 2 else (4, 5)
        guard.observe(state(index, position, **kwargs))
    return guard


LEGAL = np.ones(6, dtype=bool)


def test_frozen_evaluation_default_selects_confirmed_sixteen_step_mode():
    assert DEFAULT_NARROW_LOOP_GUARD_MODE == "mid_persistent"
    assert _loop_guard_window(DEFAULT_NARROW_LOOP_GUARD_MODE) == 16


def test_guard_requires_a_full_unchanged_hazard_free_late_game_window():
    guard = NarrowLoopGuard()
    for index in range(1, 24):
        guard.observe(state(index, (4, 4) if index % 2 else (4, 5)))
    current = state(23, (4, 4))
    assert not guard.stuck(current)
    assert not pacing(crates=True).stuck(state(24, (4, 5), crates=True))
    assert not pacing(bombs=[((2, 2), 3)]).stuck(state(24, (4, 5), bombs=[((2, 2), 3)]))


def test_guard_requires_a_remaining_objective():
    guard = pacing(coins=(), others=())
    assert not guard.stuck(state(24, (4, 5), coins=(), others=()))


def test_guard_uses_highest_q_safe_novel_uncontested_move():
    guard = pacing()
    current = state(24, (4, 5))
    chosen = guard.choose(current, "UP", q(UP=9, RIGHT=2, DOWN=5, LEFT=3), LEGAL)
    assert chosen == "DOWN"
    assert guard.snapshot()["overrides"] == 1


def test_guard_rejects_a_tile_an_opponent_can_take_next():
    opponent = ("them", 0, True, (5, 6))
    guard = pacing(others=(opponent,))
    current = state(24, (4, 5), others=(opponent,))
    chosen = guard.choose(current, "UP", q(UP=9, RIGHT=8, DOWN=7, LEFT=6), LEGAL)
    assert chosen == "LEFT"
    assert guard.snapshot()["rejected_contested"] == 2


def test_guard_leaves_wait_bomb_and_non_looping_move_unchanged():
    current = state(24, (4, 5))
    for chosen in ("WAIT", "BOMB", "DOWN"):
        guard = pacing()
        assert guard.choose(current, chosen, q(LEFT=9), LEGAL) == chosen
        assert guard.snapshot()["overrides"] == 0


def test_progress_change_breaks_the_window():
    guard = NarrowLoopGuard()
    for index in range(1, 25):
        score = 1 if index == 24 else 0
        guard.observe(state(index, (4, 4) if index % 2 else (4, 5), score=score))
    assert not guard.stuck(state(24, (4, 5), score=1))


def test_cooldown_only_redirects_a_contested_policy_move_after_override():
    opponent = ("them", 0, True, (6, 6))
    guard = pacing(others=(opponent,))
    current = state(24, (4, 5), others=(opponent,))
    assert guard.choose(current, "UP", q(UP=9, DOWN=8), LEGAL, followup_steps=4) == "DOWN"
    followup = state(25, (4, 6), others=(opponent,))
    assert guard.redirect_contested(followup, "RIGHT", q(RIGHT=9, LEFT=8), LEGAL) == "LEFT"
    assert guard.snapshot()["followup_redirects"] == 1
    assert guard.snapshot()["followup_steps_remaining"] == 3


def test_no_cooldown_leaves_a_contested_policy_move_unchanged():
    opponent = ("them", 0, True, (6, 6))
    guard = NarrowLoopGuard()
    current = state(25, (4, 6), others=(opponent,))
    assert guard.redirect_contested(current, "RIGHT", q(LEFT=9), LEGAL) == "RIGHT"


def test_persistent_followup_remains_active_after_four_decisions():
    opponent = ("them", 0, True, (6, 6))
    guard = pacing(others=(opponent,))
    current = state(24, (4, 5), others=(opponent,))
    guard.choose(current, "UP", q(DOWN=8), LEGAL, followup_steps=400)
    for step_number in range(25, 29):
        guard.redirect_contested(state(step_number, (4, 6), others=(opponent,)), "WAIT", q(), LEGAL)
    followup = state(29, (4, 6), others=(opponent,))
    assert guard.redirect_contested(followup, "RIGHT", q(LEFT=8), LEGAL) == "LEFT"
    assert guard.snapshot()["followup_steps_remaining"] == 395


@pytest.mark.parametrize("window", [12, 16])
def test_configurable_window_triggers_only_after_its_full_history(window):
    guard = NarrowLoopGuard(window=window)
    for index in range(1, window):
        guard.observe(state(index, (4, 4) if index % 2 else (4, 5)))
    assert not guard.stuck(state(window - 1, (4, 4) if (window - 1) % 2 else (4, 5)))

    current = state(window, (4, 4) if window % 2 else (4, 5))
    guard.observe(current)
    assert guard.stuck(current)
    assert guard.choose(
        current, "UP", q(UP=9, DOWN=8), LEGAL, followup_steps=400
    ) == "DOWN"
