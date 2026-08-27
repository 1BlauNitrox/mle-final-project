"""Tests for DerKleineVermögensumverteiler configuration."""

from agent_code.DerKleineVermoegensumverteiler.config import (
    ACTION_TO_INDEX,
    ACTIONS,
    DISCOUNT_FACTOR,
    EPSILON_DECAY,
    INITIAL_EPSILON,
    LEARNING_RATE,
    MINIMUM_EPSILON,
    REWARDS,
)


def test_task1_action_order() -> None:
    assert ACTIONS == (
        "UP",
        "RIGHT",
        "DOWN",
        "LEFT",
        "WAIT",
    )


def test_bomb_is_not_part_of_action_space() -> None:
    assert "BOMB" not in ACTIONS
    assert "BOMB" not in ACTION_TO_INDEX


def test_action_indices_match_action_order() -> None:
    assert ACTION_TO_INDEX == {
        "UP": 0,
        "RIGHT": 1,
        "DOWN": 2,
        "LEFT": 3,
        "WAIT": 4,
    }


def test_learning_parameters_are_in_valid_ranges() -> None:
    assert 0.0 < LEARNING_RATE <= 1.0
    assert 0.0 <= DISCOUNT_FACTOR <= 1.0


def test_epsilon_schedule_is_valid() -> None:
    assert 0.0 <= MINIMUM_EPSILON <= INITIAL_EPSILON <= 1.0
    assert 0.0 < EPSILON_DECAY <= 1.0


def test_minimal_reward_mapping_is_defined() -> None:
    assert REWARDS["COIN_COLLECTED"] > 0.0
    assert REWARDS["INVALID_ACTION"] < 0.0
    assert REWARDS["WAITED"] < 0.0
