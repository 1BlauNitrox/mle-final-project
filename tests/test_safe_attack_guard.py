"""Contracts for the narrow learned-policy attack tie-break."""

import numpy as np

from agent_code.DagobertDuckDQNAntiLoop.config import ACTIONS
from agent_code.DagobertDuckDQNAntiLoop.safe_attack_guard import SafeAttackGuard


def state(*, bomb_ready=True, bombs=(), opponents=(("other", 0, True, (4, 6)),)):
    field = np.zeros((9, 9), dtype=int)
    field[[0, -1], :] = -1
    field[:, [0, -1]] = -1
    return {
        "round": 1,
        "step": 1,
        "field": field,
        "self": ("me", 0, bomb_ready, (4, 4)),
        "others": list(opponents),
        "coins": [],
        "bombs": list(bombs),
        "explosion_map": np.zeros_like(field),
    }


def q(**values):
    return lambda: np.asarray([values.get(action, 0.0) for action in ACTIONS])


LEGAL = np.ones(len(ACTIONS), dtype=bool)


def test_second_ranked_safe_bomb_is_forced_at_most_once_per_round():
    guard = SafeAttackGuard()
    assert guard.choose(state(), "WAIT", q(WAIT=9, BOMB=8, LEFT=7), LEGAL) == "BOMB"
    assert guard.choose(state(), "WAIT", q(WAIT=9, BOMB=8, LEFT=7), LEGAL) == "WAIT"
    assert guard.snapshot() == {"eligible": 1, "overrides": 1, "rejected_rank": 0}


def test_bomb_below_second_rank_remains_under_learned_policy_control():
    guard = SafeAttackGuard()
    assert guard.choose(state(), "WAIT", q(WAIT=9, LEFT=8, BOMB=7), LEGAL) == "WAIT"
    assert guard.snapshot() == {"eligible": 1, "overrides": 0, "rejected_rank": 1}


def test_guard_requires_legal_ready_hazard_free_opponent_facing_bomb():
    unavailable = LEGAL.copy()
    unavailable[ACTIONS.index("BOMB")] = False
    cases = (
        (state(bomb_ready=False), unavailable),
        (state(bombs=(((2, 2), 3),)), LEGAL),
        (state(opponents=(("other", 0, True, (6, 6)),)), LEGAL),
    )
    for game_state, legal in cases:
        guard = SafeAttackGuard()
        assert guard.choose(game_state, "WAIT", q(WAIT=9, BOMB=8), legal) == "WAIT"
        assert guard.snapshot()["overrides"] == 0
