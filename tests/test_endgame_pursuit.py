"""Contracts for learned endgame pursuit and its training teacher."""

import numpy as np
import torch

from agent_code.DagobertDuckDQNAntiLoop.callbacks import (
    _active_loop_guard_window,
    _loop_guard_window,
    _pursuit_guard_kwargs,
)
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


def test_bomb_only_mode_preserves_learned_movement_decision():
    guard = movement_guard()
    guard.allow_movement_override = False
    q_values = lambda: np.zeros(len(ACTIONS))  # noqa: E731

    assert guard.choose(state(), "WAIT", q_values, LEGAL, FEATURES) == "WAIT"
    assert guard.snapshot()["rejected_movement_override"] == 1


def test_bomb_only_mode_requires_dqn_agreement_and_enforces_round_limit():
    guard = bomb_guard(allow=True)
    guard.allow_movement_override = False
    guard.maximum_better_bomb_actions = 2
    guard.maximum_bomb_overrides_per_round = 1
    values = np.zeros(len(ACTIONS))
    values[ACTIONS.index("UP")] = 3
    values[BOMB_INDEX] = 2
    q_values = lambda: values  # noqa: E731

    first = state()
    assert guard.choose(first, "UP", q_values, LEGAL, FEATURES) == "BOMB"
    first["step"] += 1
    assert guard.choose(first, "UP", q_values, LEGAL, FEATURES) == "UP"

    next_round = state()
    next_round["round"] = 2
    assert guard.choose(next_round, "UP", q_values, LEGAL, FEATURES) == "BOMB"

    too_low = bomb_guard(allow=True)
    too_low.maximum_better_bomb_actions = 2
    values[ACTIONS.index("RIGHT")] = 4
    values[ACTIONS.index("DOWN")] = 5
    assert too_low.choose(state(), "DOWN", q_values, LEGAL, FEATURES) == "DOWN"
    assert too_low.snapshot()["rejected_bomb_rank"] == 1


def test_escape_followup_redirects_only_an_unsafe_frozen_action():
    guard = bomb_guard(allow=True)
    guard.escape_steps_remaining = 7
    guard.escape_armed_step = 100
    game_state = state()
    game_state["step"] = 101
    game_state["self"] = ("me", 0, False, (4, 5))
    game_state["bombs"] = [((4, 4), 0)]
    legal = LEGAL.copy()
    legal[BOMB_INDEX] = False
    values = np.zeros(len(ACTIONS))
    values[ACTIONS.index("WAIT")] = 10
    values[ACTIONS.index("RIGHT")] = 5
    q_values = lambda: values  # noqa: E731

    assert guard.redirect_unsafe_escape(game_state, "WAIT", q_values, legal) == "RIGHT"
    assert guard.snapshot()["escape_redirects"] == 1


def test_corridor_gate_counts_immediate_target_escapes():
    guard = bomb_guard(allow=True)
    guard.allow_movement_override = False
    guard.maximum_immediate_target_escapes = 1
    values = np.zeros(len(ACTIONS))
    q_values = lambda: values  # noqa: E731

    open_state = state()
    assert guard._target_escape_options(open_state) == [2]
    assert guard.choose(open_state, "WAIT", q_values, LEGAL, FEATURES) == "WAIT"
    assert guard.snapshot()["rejected_target_mobility"] == 1

    corridor_state = state()
    corridor_state["field"][3, 6] = -1
    assert guard._target_escape_options(corridor_state) == [1]
    assert guard.choose(corridor_state, "WAIT", q_values, LEGAL, FEATURES) == "BOMB"


def test_corridor_cap_modes_change_only_the_registered_bomb_limit():
    cap1 = _pursuit_guard_kwargs("broad_learned_pursuit_corridor1_cap1")
    cap2 = _pursuit_guard_kwargs("broad_learned_pursuit_corridor1_cap2")

    assert cap1 == {
        "allow_bomb_override": True,
        "allow_movement_override": False,
        "maximum_better_bomb_actions": None,
        "maximum_bomb_overrides_per_round": 1,
        "maximum_immediate_target_escapes": 1,
        "escape_followup_steps": 7,
    }
    assert cap2 == {**cap1, "maximum_bomb_overrides_per_round": 2}


def test_earlier_loop_modes_change_only_the_registered_history_window():
    cap2 = _pursuit_guard_kwargs("broad_learned_pursuit_corridor1_cap2")
    for mode, window in (
        ("broad_learned_pursuit_corridor1_cap2_w12", 12),
        ("broad_learned_pursuit_corridor1_cap2_w8", 8),
    ):
        assert _pursuit_guard_kwargs(mode) == cap2
        assert _loop_guard_window(mode) == window


def test_post_kill_cap_preserves_second_attempt_until_an_opponent_is_eliminated():
    values = np.zeros(len(ACTIONS))
    q_values = lambda: values  # noqa: E731

    guard = bomb_guard(allow=True)
    guard.maximum_bomb_overrides_per_round = 2
    guard.stop_after_opponent_elimination = True
    first = state(opponents=(("a", 0, True, (4, 6)), ("b", 0, True, (6, 4))))
    assert guard.choose(first, "WAIT", q_values, LEGAL, FEATURES) == "BOMB"
    after_kill = state(opponents=(("a", 0, True, (4, 6)),))
    after_kill["step"] += 10
    assert guard.choose(after_kill, "WAIT", q_values, LEGAL, FEATURES) == "WAIT"
    assert guard.snapshot()["rejected_post_kill"] == 1

    retry_guard = bomb_guard(allow=True)
    retry_guard.maximum_bomb_overrides_per_round = 2
    retry_guard.stop_after_opponent_elimination = True
    assert retry_guard.choose(first, "WAIT", q_values, LEGAL, FEATURES) == "BOMB"
    retry = state(opponents=(("a", 0, True, (4, 6)), ("b", 0, True, (6, 4))))
    retry["step"] += 10
    assert retry_guard.choose(retry, "WAIT", q_values, LEGAL, FEATURES) == "BOMB"


def test_post_kill_mode_changes_only_the_registered_kill_cap():
    control = _pursuit_guard_kwargs("broad_learned_pursuit_corridor1_cap2_w12")
    candidate = _pursuit_guard_kwargs(
        "broad_learned_pursuit_corridor1_cap2_w12_postkill"
    )
    assert candidate == {**control, "stop_after_opponent_elimination": True}
    assert _loop_guard_window("broad_learned_pursuit_corridor1_cap2_w12_postkill") == 12


def test_targeted_loop_mode_shortens_history_only_after_a_pursuit_bomb():
    mode = "broad_learned_pursuit_corridor1_cap2_targeted12"
    guard = bomb_guard(allow=True)
    assert _pursuit_guard_kwargs(mode) == _pursuit_guard_kwargs(
        "broad_learned_pursuit_corridor1_cap2"
    )
    assert _loop_guard_window(mode) == 16
    assert _active_loop_guard_window(mode, guard) == 16
    guard.bomb_overrides_this_round = 1
    assert _active_loop_guard_window(mode, guard) == 12

    postkill = "broad_learned_pursuit_corridor1_cap2_targeted12_postkill"
    assert _pursuit_guard_kwargs(postkill) == {
        **_pursuit_guard_kwargs(mode),
        "stop_after_opponent_elimination": True,
    }
    assert _active_loop_guard_window(postkill, guard) == 12


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
