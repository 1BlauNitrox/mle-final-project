"""Tests for the DagobertDuckDQN configuration"""

from dataclasses import replace

import pytest

from agent_code.DagobertDuckDQN.config import (
    ACTION_TO_INDEX,
    ACTIONS,
    DEFAULT_CONFIG,
    FEATURE_COUNT,
    REWARDS,
)


def test_task1_action_order_is_fixed() -> None:
    assert ACTIONS == (
        "UP",
        "RIGHT",
        "DOWN",
        "LEFT",
        "WAIT",
    )

def test_bomb_is_impossible_by_construction() -> None:
    assert "BOMB" not in ACTIONS
    assert "BOMB" not in ACTION_TO_INDEX

def test_action_indices_match_network_outputs() -> None:
    assert ACTION_TO_INDEX == {
        "UP": 0,
        "RIGHT": 1,
        "DOWN": 2,
        "LEFT": 3,
        "WAIT": 4,
    }

def test_initial_reward_mapping_matches_tabular_agent() -> None:
    assert REWARDS == {
        "COIN_COLLECTED": 10.0,
        "INVALID_ACTION": -0.5,
        "WAITED": -0.1,
    }

def test_network_dimensions_match_contract() -> None:
    assert FEATURE_COUNT == 8
    assert DEFAULT_CONFIG.input_dim == FEATURE_COUNT
    assert DEFAULT_CONFIG.output_dim == len(ACTIONS)
    assert all(size > 0 for size in DEFAULT_CONFIG.hidden_sizes)

def test_learning_parameters_are_valid() -> None:
    assert DEFAULT_CONFIG.learning_rate > 0.0
    assert 0.0 <= DEFAULT_CONFIG.discount_factor <= 1.0
    assert DEFAULT_CONFIG.gradient_clip_norm > 0.0

def test_replay_configuration_is_bounded_and_trainable() -> None:
    assert DEFAULT_CONFIG.batch_size > 0
    assert DEFAULT_CONFIG.batch_size <= DEFAULT_CONFIG.replay_warmup
    assert DEFAULT_CONFIG.replay_warmup <= DEFAULT_CONFIG.replay_capacity
    assert DEFAULT_CONFIG.target_update_interval > 0

def test_epsilon_schedule_is_valid() -> None:
    assert (
        0.0
        <= DEFAULT_CONFIG.minimum_epsilon
        <= DEFAULT_CONFIG.initial_epsilon
        <= 1.0
    )
    assert 0.0 < DEFAULT_CONFIG.epsilon_decay <= 1.0

def test_evaluation_defaults_to_one_torch_thread() -> None:
    assert DEFAULT_CONFIG.torch_num_threads == 1

@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("learning_rate", 0.0),
        ("discount_factor", -0.1),
        ("discount_factor", 1.1),
        ("batch_size", 0),
        ("replay_capacity", 0),
        ("replay_warmup", 0),
        ("target_update_interval", 0),
        ("initial_epsilon", 1.1),
        ("minimum_epsilon", -0.1),
        ("epsilon_decay", 0.0),
        ("gradient_clip_norm", 0.0),
        ("torch_num_threads", 0),
        ("torch_num_threads", 2),
    ],
)

def test_invalid_configuration_is_rejected(
    field: str,
    invalid_value: int | float,
) -> None:
    with pytest.raises(ValueError):
        replace(DEFAULT_CONFIG, **{field: invalid_value})
