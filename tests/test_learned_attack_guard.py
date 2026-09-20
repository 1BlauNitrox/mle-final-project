"""Contracts for the learned high-confidence attack gate."""

import numpy as np

from agent_code.DagobertDuckDQNAntiLoop.config import ACTIONS, DEFAULT_CONFIG
from agent_code.DagobertDuckDQNAntiLoop.learned_attack_guard import LearnedAttackGuard
from agent_code.DagobertDuckDQNAntiLoop.model import build_q_network


def state(*, opponents=(("other", 0, True, (4, 6)),), bombs=()):
    field = np.zeros((9, 9), dtype=int)
    field[[0, -1], :] = -1
    field[:, [0, -1]] = -1
    return {
        "round": 1,
        "step": 1,
        "field": field,
        "self": ("me", 0, True, (4, 4)),
        "others": list(opponents),
        "coins": [],
        "bombs": list(bombs),
        "explosion_map": np.zeros_like(field),
    }


def guard(*, bias, threshold=0.8, maximum=1):
    network = build_q_network(DEFAULT_CONFIG, seed=2)
    hidden_dim = DEFAULT_CONFIG.hidden_sizes[-1]
    return LearnedAttackGuard(
        network=network,
        mean=np.zeros(hidden_dim, dtype=np.float32),
        scale=np.ones(hidden_dim, dtype=np.float32),
        weight=np.zeros(hidden_dim, dtype=np.float32),
        bias=bias,
        threshold=threshold,
        max_better_actions=maximum,
    )


LEGAL = np.ones(len(ACTIONS), dtype=bool)
FEATURES = np.zeros(DEFAULT_CONFIG.input_dim, dtype=np.float32)


def values(**overrides):
    return lambda: np.asarray([overrides.get(action, 0.0) for action in ACTIONS])


def test_confident_second_ranked_attack_overrides_only_once():
    learned = guard(bias=10)
    q_values = values(WAIT=2, BOMB=1)
    assert learned.choose(state(), "WAIT", q_values, LEGAL, FEATURES) == "BOMB"
    assert learned.choose(state(), "WAIT", q_values, LEGAL, FEATURES) == "WAIT"
    assert learned.snapshot()["overrides"] == 1


def test_gate_preserves_action_when_confidence_rank_or_geometry_fails():
    cases = (
        (guard(bias=-10), state(), values(WAIT=2, BOMB=1)),
        (guard(bias=10), state(), values(WAIT=3, LEFT=2, BOMB=1)),
        (guard(bias=10), state(opponents=(("other", 0, True, (6, 6)),)), values(WAIT=2, BOMB=1)),
    )
    for learned, game_state, q_values in cases:
        assert learned.choose(game_state, "WAIT", q_values, LEGAL, FEATURES) == "WAIT"
        assert learned.snapshot()["overrides"] == 0
