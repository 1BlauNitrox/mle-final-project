"""Contracts for learned endgame pursuit and its training teacher."""

import numpy as np
import torch

from agent_code.DagobertDuckDQNAntiLoop.config import ACTIONS
from agent_code.DagobertDuckDQNAntiLoop.endgame_pursuit_guard import EndgamePursuitGuard
from training.endgame_pursuit import (
    BOMB_INDEX,
    PursuitDataset,
    pursuit_teacher_action,
    train_grouped_pursuit,
)


def state(*, crates=False, opponents=(("other", 0, True, (4, 6)),)):
    field = np.zeros((9, 9), dtype=int)
    field[[0, -1], :] = -1
    field[:, [0, -1]] = -1
    if crates:
        field[2, 3] = 1
    return {
        "round": 1,
        "step": 100,
        "field": field,
        "self": ("me", 0, True, (4, 4)),
        "others": list(opponents),
        "coins": [],
        "bombs": [],
        "explosion_map": np.zeros_like(field),
    }


LEGAL = np.ones(len(ACTIONS), dtype=bool)
FEATURES = np.zeros(56, dtype=np.float32)


def movement_guard():
    bias2 = np.full(len(ACTIONS), -10.0, dtype=np.float32)
    bias2[ACTIONS.index("RIGHT")] = 10.0
    return EndgamePursuitGuard(
        mean=np.zeros(74, dtype=np.float32),
        scale=np.ones(74, dtype=np.float32),
        weight1=np.zeros((32, 74), dtype=np.float32),
        bias1=np.zeros(32, dtype=np.float32),
        weight2=np.zeros((len(ACTIONS), 32), dtype=np.float32),
        bias2=bias2,
        threshold=0.8,
    )


def bomb_guard(*, allow):
    value = movement_guard()
    value.bias2[:] = -10
    value.bias2[ACTIONS.index("BOMB")] = 10
    value.allow_bomb_override = allow
    return value


def test_guard_changes_only_eligible_empty_board_states():
    guard = movement_guard()

    def q_values():
        return np.zeros(len(ACTIONS))

    assert guard.choose(state(), "WAIT", q_values, LEGAL, FEATURES) == "RIGHT"
    assert guard.choose(state(crates=True), "WAIT", q_values, LEGAL, FEATURES) == "WAIT"
    assert guard.snapshot()["overrides"] == 1


def test_teacher_bombs_from_a_safe_attack_tile():
    assert pursuit_teacher_action(state(), LEGAL) == "BOMB"
    assert pursuit_teacher_action(state(crates=True), LEGAL) is None


def test_movement_only_mode_preserves_fallback_bomb_decision():
    q_values = lambda: np.zeros(len(ACTIONS))  # noqa: E731
    assert bomb_guard(allow=False).choose(state(), "WAIT", q_values, LEGAL, FEATURES) == "WAIT"
    assert bomb_guard(allow=True).choose(state(), "WAIT", q_values, LEGAL, FEATURES) == "BOMB"


def test_grouped_pursuit_training_accepts_separable_teacher_actions():
    values, labels, groups = [], [], []
    for group in range(5):
        for _ in range(10):
            positive = np.zeros(74, dtype=np.float32)
            positive[0] = 2
            negative = np.zeros(74, dtype=np.float32)
            negative[0] = -2
            values.extend((positive, negative))
            labels.extend((BOMB_INDEX, ACTIONS.index("UP")))
            groups.extend((group, group))
    dataset = PursuitDataset(
        inputs=torch.from_numpy(np.asarray(values)),
        labels=torch.tensor(labels),
        groups=np.asarray(groups),
    )
    parameters, report = train_grouped_pursuit(
        dataset,
        folds=5,
        epochs=100,
        learning_rate=0.02,
        l2_weight=0.001,
        seed=8,
        thresholds=[0.5],
        minimum_coverage=0.5,
        minimum_accuracy=0.9,
        minimum_bomb_precision=0.9,
        minimum_bomb_recall=0.9,
    )
    assert parameters is not None
    assert report["accepted_thresholds"] == [0.5]
