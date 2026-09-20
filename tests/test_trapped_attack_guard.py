"""Contracts for the conservative trapped-opponent bomb override."""

import numpy as np

from agent_code.DagobertDuckDQNAntiLoop.config import ACTIONS
from agent_code.DagobertDuckDQNAntiLoop.trapped_attack_guard import (
    TrappedAttackGuard,
    opponent_can_escape,
)


def state(*, trapped=True, self_trapped=False, others=1):
    field = np.zeros((9, 9), dtype=int)
    field[[0, -1], :] = -1
    field[:, [0, -1]] = -1
    if trapped:
        field[3, 5] = field[5, 5] = field[4, 6] = -1
    if self_trapped:
        field[3, 4] = field[5, 4] = field[4, 3] = -1
    opponents = [("them", 0, True, (4, 5))]
    if others == 2:
        opponents.append(("other", 0, True, (6, 6)))
    return {
        "round": 1,
        "step": 20,
        "field": field,
        "self": ("me", 0, True, (4, 4)),
        "others": opponents,
        "coins": [],
        "bombs": [],
        "explosion_map": np.zeros_like(field),
    }


LEGAL = np.ones(len(ACTIONS), dtype=bool)


def q_values():
    return np.zeros(len(ACTIONS), dtype=float)


def test_opponent_escape_search_distinguishes_open_and_sealed_routes():
    open_state = state(trapped=False)
    sealed_state = state(trapped=True)
    assert opponent_can_escape(open_state["field"], (4, 4), (4, 5))
    assert not opponent_can_escape(sealed_state["field"], (4, 4), (4, 5))


def test_guard_forces_one_bomb_when_sole_opponent_is_trapped_and_self_can_escape():
    guard = TrappedAttackGuard()
    game_state = state()
    assert guard.choose(game_state, "UP", q_values, LEGAL) == "BOMB"
    assert guard.choose(game_state, "UP", q_values, LEGAL) == "UP"
    assert guard.snapshot()["overrides"] == 1


def test_guard_rejects_escape_routes_multiple_opponents_and_unsafe_self():
    cases = (state(trapped=False), state(others=2), state(self_trapped=True))
    for game_state in cases:
        guard = TrappedAttackGuard()
        assert guard.choose(game_state, "UP", q_values, LEGAL) == "UP"
        assert guard.snapshot()["overrides"] == 0
