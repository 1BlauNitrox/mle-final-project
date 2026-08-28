"""Tests for the DagobertDuckDQN neural-network model"""

from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from agent_code.DagobertDuckDQN.config import DEFAULT_CONFIG
from agent_code.DagobertDuckDQN.model import (
    DQNLearner,
    QNetwork,
    build_q_network,
    compute_bellman_targets,
    select_action,
)
from agent_code.DagobertDuckDQN.replay import ReplayBatch


def test_single_state_produces_one_q_value_per_action() -> None:
    network = build_q_network(DEFAULT_CONFIG, seed=1)
    state = torch.zeros(DEFAULT_CONFIG.input_dim, dtype=torch.float32)

    q_values = network(state)

    assert q_values.shape ==(DEFAULT_CONFIG.output_dim, )

def test_batch_produces_one_q_value_vector_per_state() -> None:
    network = build_q_network(DEFAULT_CONFIG, seed=1)
    states = torch.zeros((7, DEFAULT_CONFIG.input_dim), dtype=torch.float32)

    q_values = network(states)

    assert q_values.shape == (7, DEFAULT_CONFIG.output_dim)

def test_network_architecture_matches_configuration() -> None:
    network = build_q_network(DEFAULT_CONFIG, seed=1)

    linear_layers = [
        module 
        for module in network.modules()
        if isinstance(module, nn.Linear)
    ]

    dimensions = [
        (layer.in_features, layer.out_features)
        for layer in linear_layers
    ]

    assert dimensions == [
        (8, 64),
        (64, 64),
        (64, 5)
    ]

def test_equal_seeds_produce_equal_parameters() -> None:
    first = build_q_network(DEFAULT_CONFIG, seed=123)
    second = build_q_network(DEFAULT_CONFIG, seed=123)

    for first_parameter, second_parameter in zip(
        first.parameters(),
        second.parameters(),
        strict=True,
    ): 
        torch.testing.assert_close(
            first_parameter,
            second_parameter
        )

def test_different_seeds_change_initial_parameters() -> None:
    first = build_q_network(DEFAULT_CONFIG, seed=123)
    second = build_q_network(DEFAULT_CONFIG, seed=124)

    assert any(
        not torch.equal(first_parameter, second_parameter)
        for first_parameter, second_parameter in zip(
            first.parameters(),
            second.parameters(),
            strict=True,
        )
    )

def test_network_initialization_does_not_change_global_rng_state() -> None:
    torch.manual_seed(777)
    state_before = torch.random.get_rng_state()

    build_q_network(DEFAULT_CONFIG, seed=123)

    state_after = torch.random.get_rng_state()

    torch.testing.assert_close(state_before, state_after)

def test_network_is_cpu_only() -> None:
    network = build_q_network(DEFAULT_CONFIG, seed=1)

    assert all(
        parameter.device.type =="cpu"
        for parameter in network.parameters()
    )

def test_wrong_feature_count_is_rejected() -> None:
    network = build_q_network(DEFAULT_CONFIG, seed=1)
    invalid_state = torch.zeros(7, dtype=torch.float32)

    try:
        network(invalid_state)
    except ValueError as error:
        assert "feature" in str(error).lower()
    else:
        raise AssertionError("Expected a ValueError for invalid input shape")
    
def test_non_float32_input_is_rejected() -> None:
    network = build_q_network(DEFAULT_CONFIG, seed=1)
    invalid_state = torch.zeros(
        DEFAULT_CONFIG.input_dim,
        dtype=torch.float64
    )
    
    try:
        network(invalid_state)
    except ValueError as error:
        assert "float32" in str(error)
    else:
        raise AssertionError("Expected a ValueError for invalid input dtype")
    
def test_outputs_are_finite() -> None:
    network = build_q_network(DEFAULT_CONFIG, seed=1)
    state = torch.ones(DEFAULT_CONFIG.input_dim, dtype=torch.float32)

    q_values = network(state)

    assert torch.isfinite(q_values).all()

def test_q_network_is_a_torch_module() -> None:
    network = QNetwork(DEFAULT_CONFIG)

    assert isinstance(network, nn.Module)

def test_bellman_targets_bootstrap_only_non_terminal_transitions() -> None:
    rewards = torch.tensor([1.0, 2.0], dtype=torch.float32)
    next_q_values = torch.tensor(
        [
            [10.0, 4.0, 3.0, 2.0, 1.0],
            [100.0, 50.0, 20.0, 10.0, 5.0],
        ],
        dtype=torch.float32,
    )
    terminals = torch.tensor([False, True], dtype=torch.bool)

    targets = compute_bellman_targets(
        rewards=rewards,
        next_q_values=next_q_values,
        terminals=terminals,
        discount_factor=0.9,
    )

    torch.testing.assert_close(
        targets,
        torch.tensor(
            [
                1.0 + 0.9 * 10.0,
                2.0,
            ],
            dtype=torch.float32,
        ),
    )

def test_bellman_targets_do_not_track_gradients() -> None:
    rewards = torch.tensor([1.0], dtype=torch.float32)
    next_q_values = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0, 5.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    terminals = torch.tensor([False], dtype=torch.bool)

    targets = compute_bellman_targets(
        rewards=rewards,
        next_q_values=next_q_values,
        terminals=terminals,
        discount_factor=0.9,
    )

    assert not targets.requires_grad

def test_invalid_bellman_discount_factor_is_rejected() -> None:
    rewards = torch.tensor([1.0], dtype=torch.float32)
    next_q_values = torch.zeros((1, 5), dtype=torch.float32)
    terminals = torch.tensor([False], dtype=torch.bool)

    try:
        compute_bellman_targets(
            rewards=rewards,
            next_q_values=next_q_values,
            terminals=terminals,
            discount_factor=1.1,
        )
    except ValueError as error:
        assert "discount" in str(error).lower()
    else:
        raise AssertionError("Expected an invalid discoutn factor to fail")
    
def make_small_config(**changes):
    """Create a fast configuration for unit tests."""
    config = replace(
        DEFAULT_CONFIG,
        batch_size=2,
        replay_warmup=2,
        replay_capacity=8,
    )
    return replace(config, **changes)


def make_training_batch() -> ReplayBatch:
    """Create a deterministic two-transition training batch."""
    return ReplayBatch(
        states=np.array(
            [
                [0.0] * 8,
                [1.0] * 8,
            ],
            dtype=np.float32,
        ),
        action_indices=np.array([0, 1], dtype=np.int64),
        rewards=np.array([1.0, -1.0], dtype=np.float32),
        next_states=np.array(
            [
                [0.5] * 8,
                [0.0] * 8,
            ],
            dtype=np.float32,
        ),
        terminals=np.array([False, True], dtype=np.bool_),
    )


def copy_parameters(network: nn.Module) -> list[torch.Tensor]:
    """Return detached parameter copies."""
    return [
        parameter.detach().clone()
        for parameter in network.parameters()
    ]


def test_learner_starts_with_synchronized_networks() -> None:
    learner = DQNLearner(
        config=make_small_config(),
        seed=123,
    )

    for online_parameter, target_parameter in zip(
        learner.online_network.parameters(),
        learner.target_network.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(
            online_parameter,
            target_parameter,
        )


def test_target_network_is_frozen_and_in_evaluation_mode() -> None:
    learner = DQNLearner(
        config=make_small_config(),
        seed=123,
    )

    assert not learner.target_network.training
    assert all(
        not parameter.requires_grad
        for parameter in learner.target_network.parameters()
    )


def test_training_batch_updates_online_parameters() -> None:
    learner = DQNLearner(
        config=make_small_config(),
        seed=123,
    )
    before = copy_parameters(learner.online_network)

    result = learner.train_batch(make_training_batch())

    after = copy_parameters(learner.online_network)

    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, after, strict=True)
    )
    assert result.loss >= 0.0
    assert result.mean_abs_td_error >= 0.0
    assert learner.update_steps == 1


def test_adam_optimizer_accumulates_state() -> None:
    learner = DQNLearner(
        config=make_small_config(),
        seed=123,
    )

    assert not learner.optimizer.state

    learner.train_batch(make_training_batch())

    assert learner.optimizer.state


def test_target_network_remains_unchanged_before_update_interval() -> None:
    learner = DQNLearner(
        config=make_small_config(target_update_interval=2),
        seed=123,
    )
    target_before = copy_parameters(learner.target_network)

    result = learner.train_batch(make_training_batch())

    target_after = copy_parameters(learner.target_network)

    assert not result.target_synchronized

    for old, new in zip(target_before, target_after, strict=True):
        torch.testing.assert_close(old, new)


def test_target_network_synchronizes_at_update_interval() -> None:
    learner = DQNLearner(
        config=make_small_config(target_update_interval=1),
        seed=123,
    )

    result = learner.train_batch(make_training_batch())

    assert result.target_synchronized

    for online_parameter, target_parameter in zip(
        learner.online_network.parameters(),
        learner.target_network.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(
            online_parameter,
            target_parameter,
        )


def test_target_network_never_receives_gradients() -> None:
    learner = DQNLearner(
        config=make_small_config(),
        seed=123,
    )

    learner.train_batch(make_training_batch())

    assert all(
        parameter.grad is None
        for parameter in learner.target_network.parameters()
    )

def set_network_outputs(
    network: QNetwork,
    q_values: list[float],
) -> None:
    """Configure a network to return fixed Q-values."""
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()

        output_layer = network.layers[-1]
        assert isinstance(output_layer, nn.Linear)
        output_layer.bias.copy_(
            torch.tensor(q_values, dtype=torch.float32)
        )


def test_greedy_policy_selects_highest_q_value() -> None:
    network = build_q_network(DEFAULT_CONFIG, seed=1)
    set_network_outputs(
        network,
        [0.0, 1.0, 2.0, 3.0, 4.0],
    )
    state = np.zeros(8, dtype=np.float32)
    rng = np.random.default_rng(123)

    action = select_action(
        network=network,
        state=state,
        epsilon=0.0,
        rng=rng,
    )

    assert action == "WAIT"


def test_exploration_is_reproducible_for_equal_seeds() -> None:
    network = build_q_network(DEFAULT_CONFIG, seed=1)
    state = np.zeros(8, dtype=np.float32)
    first_rng = np.random.default_rng(123)
    second_rng = np.random.default_rng(123)

    first_actions = [
        select_action(
            network=network,
            state=state,
            epsilon=1.0,
            rng=first_rng,
        )
        for _ in range(20)
    ]
    second_actions = [
        select_action(
            network=network,
            state=state,
            epsilon=1.0,
            rng=second_rng,
        )
        for _ in range(20)
    ]

    assert first_actions == second_actions


def test_greedy_tie_breaking_is_seeded() -> None:
    network = build_q_network(DEFAULT_CONFIG, seed=1)
    set_network_outputs(
        network,
        [1.0, 1.0, 1.0, 1.0, 1.0],
    )
    state = np.zeros(8, dtype=np.float32)
    first_rng = np.random.default_rng(456)
    second_rng = np.random.default_rng(456)

    first_actions = [
        select_action(
            network=network,
            state=state,
            epsilon=0.0,
            rng=first_rng,
        )
        for _ in range(20)
    ]
    second_actions = [
        select_action(
            network=network,
            state=state,
            epsilon=0.0,
            rng=second_rng,
        )
        for _ in range(20)
    ]

    assert first_actions == second_actions
    assert len(set(first_actions)) > 1


def test_policy_never_returns_bomb() -> None:
    network = build_q_network(DEFAULT_CONFIG, seed=1)
    state = np.zeros(8, dtype=np.float32)
    rng = np.random.default_rng(123)

    actions = {
        select_action(
            network=network,
            state=state,
            epsilon=1.0,
            rng=rng,
        )
        for _ in range(100)
    }

    assert "BOMB" not in actions
    assert actions <= {
        "UP",
        "RIGHT",
        "DOWN",
        "LEFT",
        "WAIT",
    }


def test_action_selection_does_not_mutate_parameters() -> None:
    network = build_q_network(DEFAULT_CONFIG, seed=1)
    state = np.zeros(8, dtype=np.float32)
    rng = np.random.default_rng(123)
    before = copy_parameters(network)

    for _ in range(20):
        select_action(
            network=network,
            state=state,
            epsilon=0.5,
            rng=rng,
        )

    after = copy_parameters(network)

    for old, new in zip(before, after, strict=True):
        torch.testing.assert_close(old, new)


def test_action_selection_does_not_create_gradients() -> None:
    network = build_q_network(DEFAULT_CONFIG, seed=1)
    state = np.zeros(8, dtype=np.float32)

    select_action(
        network=network,
        state=state,
        epsilon=0.0,
        rng=np.random.default_rng(123),
    )

    assert all(
        parameter.grad is None
        for parameter in network.parameters()
    )


def test_invalid_epsilon_is_rejected() -> None:
    network = build_q_network(DEFAULT_CONFIG, seed=1)
    state = np.zeros(8, dtype=np.float32)

    try:
        select_action(
            network=network,
            state=state,
            epsilon=1.1,
            rng=np.random.default_rng(123),
        )
    except ValueError as error:
        assert "epsilon" in str(error).lower()
    else:
        raise AssertionError("Expected invalid epsilon to fail.")

def test_learner_state_round_trip_resumes_training_exactly() -> None:
    config = make_small_config(target_update_interval=10)
    original = DQNLearner(config=config, seed=123)
    batch = make_training_batch()

    original.train_batch(batch)
    saved_state = original.state_dict()

    restored = DQNLearner(config=config, seed=999)
    restored.load_state_dict(saved_state)

    assert restored.update_steps == original.update_steps

    original_result = original.train_batch(batch)
    restored_result = restored.train_batch(batch)

    assert restored_result.loss == pytest.approx(original_result.loss)
    assert restored_result.mean_abs_td_error == pytest.approx(
        original_result.mean_abs_td_error
    )
    assert (
        restored_result.target_synchronized
        == original_result.target_synchronized
    )

    for original_parameter, restored_parameter in zip(
        original.online_network.parameters(),
        restored.online_network.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(
            original_parameter,
            restored_parameter,
        )

    for original_parameter, restored_parameter in zip(
        original.target_network.parameters(),
        restored.target_network.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(
            original_parameter,
            restored_parameter,
        )


def test_learner_state_dict_is_a_defensive_copy() -> None:
    learner = DQNLearner(
        config=make_small_config(),
        seed=123,
    )
    parameters_before = copy_parameters(learner.online_network)

    saved_state = learner.state_dict()
    saved_online_state = saved_state["online_network"]

    assert isinstance(saved_online_state, dict)

    first_tensor = next(iter(saved_online_state.values()))
    first_tensor.add_(100.0)

    parameters_after = copy_parameters(learner.online_network)

    for before, after in zip(
        parameters_before,
        parameters_after,
        strict=True,
    ):
        torch.testing.assert_close(before, after)


def test_learner_state_with_missing_field_is_rejected() -> None:
    learner = DQNLearner(
        config=make_small_config(),
        seed=123,
    )
    saved_state = learner.state_dict()
    del saved_state["optimizer"]

    with pytest.raises(ValueError):
        learner.load_state_dict(saved_state)


def test_negative_update_count_is_rejected() -> None:
    learner = DQNLearner(
        config=make_small_config(),
        seed=123,
    )
    saved_state = learner.state_dict()
    saved_state["update_steps"] = -1

    with pytest.raises(ValueError):
        learner.load_state_dict(saved_state)