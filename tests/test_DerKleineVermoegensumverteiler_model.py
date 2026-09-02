"""Tests for the Task 1 tabular Q-learning model."""

import numpy as np
import pytest

from agent_code.DerKleineVermoegensumverteiler.config import ACTIONS
from agent_code.DerKleineVermoegensumverteiler.model import QTable

STATE = (1, 1, 1, 1, 1, 1, 0, 2)
NEXT_STATE = (1, 0, 1, 1, 1, 0, 1, 1)


def test_unseen_state_has_zero_q_values() -> None:
    model = QTable()

    values = model.q_values(STATE)

    np.testing.assert_array_equal(
        values,
        np.zeros(len(ACTIONS)),
    )
    assert len(model) == 0


def test_returned_q_values_do_not_mutate_model() -> None:
    model = QTable()
    model.update(
        state=STATE,
        action="UP",
        reward=2.0,
        next_state=None,
        terminal=True,
    )

    returned_values = model.q_values(STATE)
    returned_values[0] = 999.0

    assert model.q_values(STATE)[0] != 999.0


def test_unique_greedy_action_is_selected() -> None:
    model = QTable(learning_rate=1.0)
    model.update(
        state=STATE,
        action="RIGHT",
        reward=5.0,
        next_state=None,
        terminal=True,
    )

    action = model.select_action(
        STATE,
        epsilon=0.0,
        rng=np.random.default_rng(1),
    )

    assert action == "RIGHT"


def test_seeded_tie_breaking_is_reproducible() -> None:
    first_model = QTable()
    second_model = QTable()

    first_rng = np.random.default_rng(42)
    second_rng = np.random.default_rng(42)

    first_actions = [
        first_model.select_action(
            STATE,
            epsilon=0.0,
            rng=first_rng,
        )
        for _ in range(20)
    ]

    second_actions = [
        second_model.select_action(
            STATE,
            epsilon=0.0,
            rng=second_rng,
        )
        for _ in range(20)
    ]

    assert first_actions == second_actions


def test_full_exploration_returns_only_task1_actions() -> None:
    model = QTable()
    rng = np.random.default_rng(7)

    selected_actions = {
        model.select_action(
            STATE,
            epsilon=1.0,
            rng=rng,
        )
        for _ in range(200)
    }

    assert selected_actions <= set(ACTIONS)
    assert "BOMB" not in selected_actions


def test_ordinary_q_learning_update() -> None:
    model = QTable(
        learning_rate=0.5,
        discount_factor=0.9,
    )

    model.update(
        state=NEXT_STATE,
        action="LEFT",
        reward=4.0,
        next_state=None,
        terminal=True,
    )

    td_error = model.update(
        state=STATE,
        action="UP",
        reward=1.0,
        next_state=NEXT_STATE,
        terminal=False,
    )

    # target = 1 + 0.9 * 2 = 2.8
    # updated Q = 0 + 0.5 * 2.8 = 1.4
    assert td_error == pytest.approx(2.8)
    assert model.q_values(STATE)[0] == pytest.approx(1.4)


def test_terminal_update_does_not_bootstrap() -> None:
    model = QTable(
        learning_rate=0.5,
        discount_factor=0.9,
    )

    model.update(
        state=NEXT_STATE,
        action="RIGHT",
        reward=100.0,
        next_state=None,
        terminal=True,
    )

    td_error = model.update(
        state=STATE,
        action="UP",
        reward=2.0,
        next_state=NEXT_STATE,
        terminal=True,
    )

    # The value 50 in NEXT_STATE must not influence the terminal target.
    assert td_error == pytest.approx(2.0)
    assert model.q_values(STATE)[0] == pytest.approx(1.0)


def test_update_rejects_bomb_action() -> None:
    model = QTable()

    with pytest.raises(ValueError, match="Invalid action: BOMB"):
        model.update(
            state=STATE,
            action="BOMB",
            reward=0.0,
            next_state=None,
            terminal=True,
        )


def test_non_terminal_update_requires_next_state() -> None:
    model = QTable()

    with pytest.raises(
        ValueError,
        match="Next state must be provided for non-terminal updates.",
    ):
        model.update(
            state=STATE,
            action="WAIT",
            reward=0.0,
            next_state=None,
            terminal=False,
        )


@pytest.mark.parametrize("epsilon", [-0.1, 1.1])
def test_invalid_epsilon_is_rejected(
    epsilon: float,
) -> None:
    model = QTable()

    with pytest.raises(
        ValueError,
        match="Epsilon must be in \\[0, 1\\]\\.",
    ):
        model.select_action(
            STATE,
            epsilon=epsilon,
            rng=np.random.default_rng(1),
        )

def test_greedy_selection_does_not_treat_close_values_as_tie() -> None:
    model = QTable()

    model.values[STATE] = np.array(
        [
            1.0,
            1.0 - 1e-12,
            0.0,
            0.0,
            0.0,
        ]
    )

    rng = np.random.default_rng(42)

    selected_actions = {
        model.select_action(
            STATE,
            epsilon=0.0,
            rng=rng,
        )
        for _ in range(100)
    }

    assert selected_actions == {"UP"}