"""Tests for Task 2 tabular framework-legal action masking."""

from __future__ import annotations

import numpy as np
import pytest

from agent_code.DerKleineSprengstoffkapitalist.legality import (
    framework_legal_action_mask,
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


def make_game_state() -> dict:
    field = np.zeros((7, 7), dtype=int)
    field[[0, -1], :] = -1
    field[:, [0, -1]] = -1

    field[3, 2] = 1  # crate above

    return {
        "field": field,
        "self": ("agent", 0, False, (3, 3)),
        "bombs": [((4, 3), 3)],
        "others": [("other", 0, True, (2, 3))],
    }


def test_mask_matches_framework_legality() -> None:
    mask = framework_legal_action_mask(make_game_state())

    assert mask.tolist() == [
        False,  # UP: crate
        False,  # RIGHT: bomb
        True,  # DOWN: free
        False,  # LEFT: opponent
        True,  # WAIT
        False,  # BOMB: unavailable
    ]


def test_bomb_is_legal_when_available() -> None:
    game_state = make_game_state()
    game_state["self"] = ("agent", 0, True, (3, 3))

    mask = framework_legal_action_mask(game_state)

    assert bool(mask[5]) is True


def test_masked_greedy_selection_ignores_illegal_high_value() -> None:
    q_table = QTable()
    q_table.values[TEST_STATE] = np.array([1.0, 100.0, 2.0, 3.0, 4.0, 99.0])

    mask = np.array(
        [True, False, True, True, True, False],
        dtype=np.bool_,
    )

    action = q_table.select_action(
        TEST_STATE,
        epsilon=0.0,
        rng=np.random.default_rng(1),
        action_mask=mask,
    )

    assert action == "WAIT"


def test_masked_exploration_only_uses_legal_actions() -> None:
    q_table = QTable()

    mask = np.array(
        [True, False, False, False, True, False],
        dtype=np.bool_,
    )

    selected = {
        q_table.select_action(
            TEST_STATE,
            epsilon=1.0,
            rng=np.random.default_rng(seed),
            action_mask=mask,
        )
        for seed in range(50)
    }

    assert selected <= {"UP", "WAIT"}
    assert selected == {"UP", "WAIT"}


def test_masked_tie_breaking_only_uses_legal_actions() -> None:
    q_table = QTable()
    q_table.values[TEST_STATE] = np.ones(6)

    mask = np.array(
        [False, True, False, True, False, False],
        dtype=np.bool_,
    )

    selected = {
        q_table.select_action(
            TEST_STATE,
            epsilon=0.0,
            rng=np.random.default_rng(seed),
            action_mask=mask,
        )
        for seed in range(50)
    }

    assert selected == {"RIGHT", "LEFT"}


def test_masked_bellman_update_excludes_illegal_high_value() -> None:
    q_table = QTable(
        learning_rate=1.0,
        discount_factor=0.5,
    )
    q_table.values[TEST_STATE] = np.array([1.0, 100.0, 2.0, 3.0, 4.0, 5.0])

    mask = np.array(
        [True, False, True, True, True, True],
        dtype=np.bool_,
    )

    td_error = q_table.update(
        state=TEST_STATE,
        action="UP",
        reward=1.0,
        next_state=TEST_STATE,
        terminal=False,
        next_action_mask=mask,
    )

    # Target: 1.0 + 0.5 * 5.0 = 3.5
    # Previous Q(UP): 1.0
    assert td_error == pytest.approx(2.5)
    assert q_table.q_values(TEST_STATE)[0] == pytest.approx(3.5)


def test_empty_action_mask_is_rejected() -> None:
    q_table = QTable()

    with pytest.raises(ValueError, match="at least one legal action"):
        q_table.select_action(
            TEST_STATE,
            epsilon=0.0,
            rng=np.random.default_rng(1),
            action_mask=np.zeros(6, dtype=np.bool_),
        )


def test_none_mask_preserves_previous_behavior() -> None:
    q_table = QTable()
    q_table.values[TEST_STATE] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 10.0])

    action = q_table.select_action(
        TEST_STATE,
        epsilon=0.0,
        rng=np.random.default_rng(1),
        action_mask=None,
    )

    assert action == "BOMB"
