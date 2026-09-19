"""Contracts for the tabular Task 3 successor."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from agent_code.DerKleineKonkurrenzvernichter import callbacks, train
from agent_code.DerKleineKonkurrenzvernichter.features.opponents import (
    opponent_state_to_features,
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


def test_task2_prior_is_preserved_for_every_opponent_suffix() -> None:
    parent = load_parent_prior()
    loaded = load_model()
    parent_state = next(iter(parent.values))
    expected = parent.values[parent_state]

    for suffix in ((0, 0, 0), (1, 3, 1), (4, 2, 2)):
        np.testing.assert_array_equal(
            loaded.q_table.q_values((*parent_state, *suffix)),
            expected,
        )


def test_training_is_enabled_only_in_successor() -> None:
    agent = SimpleNamespace(train=True, logger=Mock())
    callbacks.setup(agent)
    train.setup_training(agent)
    assert agent.completed_episodes == 0
    assert agent.pending_transition is None


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
