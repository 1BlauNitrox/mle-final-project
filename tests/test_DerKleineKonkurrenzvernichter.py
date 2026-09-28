"""Contracts for the tabular Task 3 successor."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from agent_code.DerKleineKonkurrenzvernichter import callbacks, train
from agent_code.DerKleineKonkurrenzvernichter.features.opponents import (
    opponent_state_to_features,
)
from agent_code.DerKleineKonkurrenzvernichter.features.shared_targets import (
    SHARED_TARGET_STATE_REPRESENTATION,
    TARGET_CRATE,
    TARGET_OPPONENT,
    shared_target_state_to_features,
)
from agent_code.DerKleineKonkurrenzvernichter.migration import load_parent_prior
from agent_code.DerKleineKonkurrenzvernichter.model import (
    DOUBLE_Q_LEARNING,
    Q_LEARNING,
    QTable,
)
from agent_code.DerKleineKonkurrenzvernichter.persistence import load_model, save_model


class FixedUpdateRng:
    """Choose a fixed Double-Q estimator in an update."""

    def __init__(self, estimator: int) -> None:
        self.estimator = estimator

    def integers(self, low: int, high: int) -> int:
        assert (low, high) == (0, 2)
        return self.estimator


def game_state(*, others: list[tuple] | None = None) -> dict:
    field = np.zeros((7, 7), dtype=int)
    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    field[:, -1] = -1
    return {
        "round": 1,
        "step": 1,
        "field": field,
        "self": ("agent", 0, True, (3, 3)),
        "coins": [],
        "bombs": [],
        "others": [] if others is None else others,
        "explosion_map": np.zeros_like(field),
    }


def test_opponent_features_are_neutral_without_opponents() -> None:
    features = opponent_state_to_features(game_state())
    assert features is not None
    assert len(features) == 8
    assert features[5:] == (0, 0, 0)


def test_opponent_features_encode_route_distance_and_attack() -> None:
    features = opponent_state_to_features(
        game_state(others=[("peaceful", 0, True, (5, 3))])
    )
    assert features is not None
    assert features[5:] == (2, 2, 2)


def test_task2_prior_is_preserved_for_unseen_opponent_state() -> None:
    parent = load_parent_prior()
    loaded = load_model()
    unseen_state = None
    for parent_state, expected in parent.values.items():
        for suffix in ((0, 0, 0), (1, 3, 1), (4, 2, 2)):
            candidate = (*parent_state, *suffix)
            if not loaded.q_table.contains_state(candidate):
                unseen_state = candidate
                np.testing.assert_array_equal(
                    loaded.q_table.q_values(candidate),
                    expected,
                )
                break
        if unseen_state is not None:
            break

    assert unseen_state is not None


def test_shared_target_prefers_reachable_opponent_over_crate() -> None:
    state = game_state(others=[("peaceful", 0, True, (5, 3))])
    state["field"][3, 4] = 1

    features = shared_target_state_to_features(state)

    assert features is not None
    assert features[3] == 2
    assert features[4] == TARGET_OPPONENT


def test_shared_target_uses_crate_without_opponent() -> None:
    state = game_state()
    state["field"][3, 4] = 1

    features = shared_target_state_to_features(state)

    assert features is not None
    assert features[3] == 0
    assert features[4] == TARGET_CRATE


def test_shared_target_marks_safe_useful_opponent_bomb() -> None:
    features = shared_target_state_to_features(
        game_state(others=[("peaceful", 0, True, (5, 3))])
    )

    assert features is not None
    assert features[5] == 3


def test_shared_target_prior_projects_target_kind_away() -> None:
    parent = load_parent_prior()
    parent_state = next(iter(parent.values))
    q_table = QTable(
        parent_values=parent.values,
        feature_count=6,
        initialization="task2_prior",
    )
    expected = parent.values[parent_state]
    shared_state = (
        parent_state[0],
        parent_state[1],
        parent_state[2],
        parent_state[3],
        TARGET_OPPONENT,
        parent_state[4],
    )

    np.testing.assert_array_equal(q_table.q_values(shared_state), expected)


def test_fresh_training_receives_task2_prior(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(callbacks, "MODEL_PATH", tmp_path / "missing.npz")
    monkeypatch.setenv(callbacks.STATE_REPRESENTATION_ENV, "compact_opponent")
    monkeypatch.setenv(callbacks.INITIALIZATION_ENV, "task2_prior")
    agent = SimpleNamespace(train=True, logger=Mock())

    callbacks.setup(agent)

    parent = load_parent_prior()
    parent_state = next(iter(parent.values))
    np.testing.assert_array_equal(
        agent.q_table.q_values((*parent_state, 0, 0, 0)),
        parent.values[parent_state],
    )


def test_shared_target_model_round_trip(tmp_path) -> None:
    parent = load_parent_prior()
    q_table = QTable(
        parent_values=parent.values,
        feature_count=6,
        initialization="task2_prior",
    )
    state = (0, 15, 2, 1, TARGET_OPPONENT, 3)
    q_table.update(
        state=state,
        action="BOMB",
        reward=5.0,
        next_state=None,
        terminal=True,
    )
    path = tmp_path / "shared-target.npz"

    save_model(
        q_table,
        epsilon=0.5,
        completed_episodes=1,
        state_representation=SHARED_TARGET_STATE_REPRESENTATION,
        initialization="task2_prior",
        path=path,
    )
    loaded = load_model(path)

    assert loaded.state_representation == SHARED_TARGET_STATE_REPRESENTATION
    assert loaded.initialization == "task2_prior"
    np.testing.assert_array_equal(
        loaded.q_table.q_values(state),
        q_table.q_values(state),
    )


def test_final_deadline_freeze_rejects_training() -> None:
    agent = SimpleNamespace(train=True, logger=Mock())
    callbacks.setup(agent)
    with pytest.raises(RuntimeError, match="frozen final tabular agent"):
        train.setup_training(agent)
    assert agent.completed_episodes == 10000


def test_double_q_update_selects_with_one_table_and_evaluates_with_other() -> None:
    table = QTable(
        learning_rate=1.0,
        discount_factor=0.9,
        feature_count=8,
        initialization="zeros",
        learning_algorithm=DOUBLE_Q_LEARNING,
    )
    state = (0,) * 8
    next_state = (1,) + (0,) * 7
    table.values[next_state] = np.array([1.0, 5.0, 0.0, 0.0, 0.0, 0.0])
    table.secondary_values[next_state] = np.array(
        [7.0, 2.0, 0.0, 0.0, 0.0, 0.0]
    )

    td_error = table.update(
        state=state,
        action="UP",
        reward=1.0,
        next_state=next_state,
        terminal=False,
        rng=FixedUpdateRng(0),
    )

    np.testing.assert_allclose(td_error, 2.8)
    np.testing.assert_allclose(table.values[state][0], 2.8)
    assert state not in table.secondary_values


def test_double_q_update_honors_legal_action_mask() -> None:
    table = QTable(
        learning_rate=1.0,
        discount_factor=1.0,
        feature_count=8,
        initialization="zeros",
        learning_algorithm=DOUBLE_Q_LEARNING,
    )
    state = (0,) * 8
    next_state = (1,) + (0,) * 7
    table.values[next_state] = np.array([9.0, 5.0, 0.0, 0.0, 0.0, 0.0])
    table.secondary_values[next_state] = np.array(
        [8.0, 3.0, 0.0, 0.0, 0.0, 0.0]
    )
    legal = np.array([False, True, True, True, True, True])

    table.update(
        state=state,
        action="WAIT",
        reward=0.0,
        next_state=next_state,
        terminal=False,
        next_action_mask=legal,
        rng=FixedUpdateRng(0),
    )

    assert table.values[state][4] == 3.0


def test_double_q_terminal_update_has_no_bootstrap() -> None:
    table = QTable(
        learning_rate=1.0,
        feature_count=8,
        initialization="zeros",
        learning_algorithm=DOUBLE_Q_LEARNING,
    )
    state = (0,) * 8

    table.update(
        state=state,
        action="BOMB",
        reward=-4.0,
        next_state=None,
        terminal=True,
        rng=FixedUpdateRng(1),
    )

    assert state not in table.values
    assert table.secondary_values[state][5] == -4.0


def test_double_q_action_selection_combines_both_tables() -> None:
    table = QTable(
        feature_count=8,
        initialization="zeros",
        learning_algorithm=DOUBLE_Q_LEARNING,
    )
    state = (0,) * 8
    table.values[state] = np.array([4.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    table.secondary_values[state] = np.array([0.0, 6.0, 0.0, 0.0, 0.0, 0.0])

    action = table.select_action(
        state,
        epsilon=0.0,
        rng=np.random.default_rng(220),
    )

    assert action == "RIGHT"


def test_double_q_updates_are_reproducible_for_a_fixed_seed() -> None:
    first = QTable(
        feature_count=8,
        initialization="zeros",
        learning_algorithm=DOUBLE_Q_LEARNING,
    )
    second = QTable(
        feature_count=8,
        initialization="zeros",
        learning_algorithm=DOUBLE_Q_LEARNING,
    )
    first_rng = np.random.default_rng(220)
    second_rng = np.random.default_rng(220)
    state = (0,) * 8

    for reward in (1.0, -0.5, 2.0, 0.25):
        for table, rng in ((first, first_rng), (second, second_rng)):
            table.update(
                state=state,
                action="UP",
                reward=reward,
                next_state=None,
                terminal=True,
                rng=rng,
            )

    np.testing.assert_array_equal(first.values[state], second.values[state])
    np.testing.assert_array_equal(
        first.secondary_values[state],
        second.secondary_values[state],
    )


def test_double_q_model_round_trip(tmp_path: Path) -> None:
    parent = load_parent_prior()
    table = QTable(
        parent_values=parent.values,
        feature_count=8,
        initialization="task2_prior",
        learning_algorithm=DOUBLE_Q_LEARNING,
    )
    state = (*next(iter(parent.values)), 0, 0, 0)
    table.values[state] = np.arange(6, dtype=float)
    table.secondary_values[state] = np.arange(6, dtype=float) + 10.0
    table.visit_counts[state] = 3
    path = tmp_path / "double-q.npz"

    save_model(
        table,
        epsilon=0.3,
        completed_episodes=4,
        state_representation="compact_opponent",
        initialization="task2_prior",
        path=path,
    )
    loaded = load_model(path)

    assert loaded.learning_algorithm == DOUBLE_Q_LEARNING
    np.testing.assert_array_equal(loaded.q_table.values[state], table.values[state])
    np.testing.assert_array_equal(
        loaded.q_table.secondary_values[state],
        table.secondary_values[state],
    )


def test_existing_model_migrates_as_single_q_learning() -> None:
    assert load_model().learning_algorithm == Q_LEARNING


def test_one_step_wrapper_matches_general_n_step_update() -> None:
    state = (0,) * 8
    next_state = (1,) + (0,) * 7
    one_step = QTable(
        learning_rate=0.5,
        discount_factor=0.75,
        feature_count=8,
        initialization="zeros",
    )
    general = QTable(
        learning_rate=0.5,
        discount_factor=0.75,
        feature_count=8,
        initialization="zeros",
    )

    first_error = one_step.update(
        state=state,
        action="UP",
        reward=2.0,
        next_state=next_state,
        terminal=False,
    )
    second_error = general.update_n_step(
        state=state,
        action="UP",
        discounted_return=2.0,
        next_state=next_state,
        terminal=False,
        bootstrap_steps=1,
    )

    assert first_error == pytest.approx(second_error)
    np.testing.assert_array_equal(one_step.q_values(state), general.q_values(state))


def test_five_step_queue_uses_discounted_terminal_return() -> None:
    agent = SimpleNamespace(
        q_table=QTable(
            learning_rate=1.0,
            discount_factor=0.5,
            feature_count=8,
            initialization="zeros",
        ),
        update_horizon=5,
        potential_shaping="none",
        transition_queue=[],
        episode_reward=0.0,
        absolute_td_errors=[],
    )
    states = [tuple([index] + [0] * 7) for index in range(5)]

    for index, state in enumerate(states):
        train._record_transition(
            agent,
            state=state,
            action="RIGHT",
            reward=1.0,
            next_state=None if index == 4 else states[index + 1],
            terminal=index == 4,
        )

    assert agent.q_table.q_values(states[0])[1] == pytest.approx(1.9375)
    assert len(agent.transition_queue) == 4
    train._drain_transition_queue(agent, force=True)
    assert agent.transition_queue == []


def test_forced_flush_uses_available_steps_and_bootstraps() -> None:
    q_table = QTable(
        learning_rate=1.0,
        discount_factor=0.5,
        feature_count=8,
        initialization="zeros",
    )
    states = [tuple([index] + [0] * 7) for index in range(4)]
    q_table.update(
        state=states[3],
        action="UP",
        reward=4.0,
        next_state=None,
        terminal=True,
    )
    agent = SimpleNamespace(
        q_table=q_table,
        update_horizon=5,
        potential_shaping="none",
        transition_queue=[],
        episode_reward=0.0,
        absolute_td_errors=[],
    )
    for index in range(3):
        train._record_transition(
            agent,
            state=states[index],
            action="RIGHT",
            reward=1.0,
            next_state=states[index + 1],
            terminal=False,
        )

    train._drain_transition_queue(agent, force=True)

    assert q_table.q_values(states[0])[1] == pytest.approx(2.25)


def test_update_horizon_is_read_and_persisted(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(callbacks.UPDATE_HORIZON_ENV, "5")
    assert callbacks._read_update_horizon() == 5

    parent = load_parent_prior()
    q_table = QTable(
        parent_values=parent.values,
        feature_count=8,
        initialization="task2_prior",
    )
    path = tmp_path / "model.npz"
    save_model(
        q_table,
        epsilon=0.2,
        completed_episodes=3,
        state_representation="compact_opponent",
        initialization="task2_prior",
        update_horizon=5,
        path=path,
    )

    assert load_model(path).update_horizon == 5
