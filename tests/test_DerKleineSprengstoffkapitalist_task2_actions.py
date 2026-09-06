"""Tests for the tabular Task 2 action contract."""

from __future__ import annotations

import numpy as np
import pytest

from agent_code.DerKleineSprengstoffkapitalist.config import (
    ACTION_TO_INDEX,
    ACTIONS,
)
from agent_code.DerKleineSprengstoffkapitalist.model import QTable

TEST_STATE = (
    1,
    1,
    1,
    1,
    0,
    0,
    0,
    0,
    1,
    0,
    15,
    1,
    0,
    0,
    0,
    0,
    0,
)


def test_task2_action_order_preserves_task1_indices() -> None:
    assert ACTIONS == (
        "UP",
        "RIGHT",
        "DOWN",
        "LEFT",
        "WAIT",
        "BOMB",
    )

    assert ACTIONS[:5] == (
        "UP",
        "RIGHT",
        "DOWN",
        "LEFT",
        "WAIT",
    )

    assert ACTION_TO_INDEX == {
        "UP": 0,
        "RIGHT": 1,
        "DOWN": 2,
        "LEFT": 3,
        "WAIT": 4,
        "BOMB": 5,
    }


def test_unseen_state_has_six_q_values_without_creating_row() -> None:
    q_table = QTable()

    values = q_table.q_values(TEST_STATE)

    np.testing.assert_array_equal(
        values,
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),
    )

    assert len(q_table) == 0


def test_greedy_policy_can_select_bomb() -> None:
    q_table = QTable()

    q_table.values[TEST_STATE] = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
    )

    action = q_table.select_action(
        TEST_STATE,
        epsilon=0.0,
        rng=np.random.default_rng(45),
    )

    assert action == "BOMB"


def test_bomb_update_uses_sixth_q_value() -> None:
    q_table = QTable(
        learning_rate=0.5,
        discount_factor=0.9,
    )

    values_before = q_table.q_values(TEST_STATE)

    assert values_before[ACTION_TO_INDEX["BOMB"]] == pytest.approx(-1.0)

    td_error = q_table.update(
        state=TEST_STATE,
        action="BOMB",
        reward=2.0,
        next_state=None,
        terminal=True,
    )

    values_after = q_table.q_values(TEST_STATE)

    assert td_error == pytest.approx(3.0)
    assert values_after.shape == (6,)
    assert values_after[ACTION_TO_INDEX["BOMB"]] == pytest.approx(0.5)

    np.testing.assert_array_equal(
        values_after[:5],
        np.zeros(5),
    )


def test_all_actions_are_accepted_by_q_update() -> None:
    for action in ACTIONS:
        q_table = QTable(
            learning_rate=0.05,
            discount_factor=0.9,
        )

        action_index = ACTION_TO_INDEX[action]
        values_before = q_table.q_values(TEST_STATE)
        value_before = values_before[action_index]

        td_error = q_table.update(
            state=TEST_STATE,
            action=action,
            reward=1.0,
            next_state=None,
            terminal=True,
        )

        values_after = q_table.q_values(TEST_STATE)

        expected_td_error = 1.0 - value_before
        expected_value = value_before + q_table.learning_rate * expected_td_error

        assert td_error == pytest.approx(expected_td_error)
        assert values_after[action_index] == pytest.approx(expected_value)
        assert len(q_table) == 1

        unchanged_indices = [index for index in range(len(ACTIONS)) if index != action_index]

        np.testing.assert_array_equal(
            values_after[unchanged_indices],
            values_before[unchanged_indices],
        )


def test_unknown_action_is_rejected() -> None:
    q_table = QTable()

    with pytest.raises(
        ValueError,
        match="Invalid action",
    ):
        q_table.update(
            state=TEST_STATE,
            action="UNKNOWN",
            reward=1.0,
            next_state=None,
            terminal=True,
        )
