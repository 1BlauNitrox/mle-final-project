"""Tests for the Task 1 -> Task 2 checkpoint weight migration."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from agent_code.DagobertDuckDQN.config import DQNConfig as ParentConfig
from agent_code.DagobertDuckDQN.features import normalize_features as normalize_parent_features
from agent_code.DagobertDuckDQN.features import state_to_features as parent_state_to_features
from agent_code.DagobertDuckDQN.model import QNetwork
from agent_code.DagobertDuckDQN.model import build_q_network as build_parent_network
from agent_code.DagobertDuckDQN.model import select_action as select_parent_action
from agent_code.DagobertDuckDQNTask2.config import DEFAULT_CONFIG, DQNConfig
from agent_code.DagobertDuckDQNTask2.features import (
    normalize_features as normalize_task2_features,
)
from agent_code.DagobertDuckDQNTask2.features import (
    state_to_features as task2_state_to_features,
)
from agent_code.DagobertDuckDQNTask2.migration import (
    BOMB_OUTPUT_BIAS,
    PARENT_INPUT_DIM,
    PARENT_OUTPUT_DIM,
    migrate_online_network,
)
from agent_code.DagobertDuckDQNTask2.model import build_q_network, select_action

Q_VALUE_TOLERANCE = 1e-6


def task1_probe_state(*, crate: bool = False) -> dict:
    """Return a framework-compatible Task 1 state with a nonzero Task 2 suffix."""
    field = np.zeros((7, 7), dtype=int)
    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    if crate:
        field[3, 2] = 1
    field[:, -1] = -1
    return {
        "round": 1,
        "step": 1,
        "field": field,
        "self": ("test-agent", 0, True, (3, 3)),
        "coins": [(5, 3)],
        "bombs": [],
        "others": [],
        "explosion_map": np.zeros_like(field),
    }


def build_mismatched_network(
    *, input_dim: int, output_dim: int, hidden_sizes: tuple[int, ...]
) -> QNetwork:
    """Build a network with dimensions DQNConfig's own validation would reject."""
    fake_config = SimpleNamespace(
        input_dim=input_dim, hidden_sizes=hidden_sizes, output_dim=output_dim
    )
    return QNetwork(fake_config)


def test_migration_preserves_input_and_hidden_weights() -> None:
    parent = build_parent_network(ParentConfig(), seed=1)
    migrated = migrate_online_network(parent, config=DEFAULT_CONFIG, seed=44)

    parent_linear = [layer for layer in parent.layers if hasattr(layer, "weight")]
    migrated_linear = [layer for layer in migrated.layers if hasattr(layer, "weight")]

    assert torch.equal(
        parent_linear[0].weight, migrated_linear[0].weight[:, :PARENT_INPUT_DIM]
    )
    assert torch.equal(parent_linear[0].bias, migrated_linear[0].bias)

    hidden_pairs = zip(parent_linear[1:-1], migrated_linear[1:-1], strict=True)
    for parent_layer, migrated_layer in hidden_pairs:
        assert torch.equal(parent_layer.weight, migrated_layer.weight)
        assert torch.equal(parent_layer.bias, migrated_layer.bias)


def test_migration_preserves_task1_output_rows() -> None:
    parent = build_parent_network(ParentConfig(), seed=2)
    migrated = migrate_online_network(parent, config=DEFAULT_CONFIG, seed=44)

    parent_output = [layer for layer in parent.layers if hasattr(layer, "weight")][-1]
    migrated_output = [layer for layer in migrated.layers if hasattr(layer, "weight")][-1]

    assert torch.equal(parent_output.weight, migrated_output.weight[:PARENT_OUTPUT_DIM, :])
    assert torch.equal(parent_output.bias, migrated_output.bias[:PARENT_OUTPUT_DIM])


def test_migration_initializes_bomb_row_conservatively() -> None:
    parent = build_parent_network(ParentConfig(), seed=3)
    migrated = migrate_online_network(parent, config=DEFAULT_CONFIG, seed=44)

    migrated_output = [layer for layer in migrated.layers if hasattr(layer, "weight")][-1]

    assert torch.equal(
        migrated_output.weight[PARENT_OUTPUT_DIM, :],
        torch.zeros(migrated_output.weight.shape[1]),
    )
    assert migrated_output.bias[PARENT_OUTPUT_DIM].item() == BOMB_OUTPUT_BIAS


def test_migration_zeroes_new_input_columns() -> None:
    parent = build_parent_network(ParentConfig(), seed=4)
    migrated = migrate_online_network(parent, config=DEFAULT_CONFIG, seed=44)

    migrated_input = [layer for layer in migrated.layers if hasattr(layer, "weight")][0]

    assert torch.equal(
        migrated_input.weight[:, PARENT_INPUT_DIM:],
        torch.zeros_like(migrated_input.weight[:, PARENT_INPUT_DIM:]),
    )


def test_nonzero_task2_suffix_previously_changed_inherited_q_values() -> None:
    """Pin the regression with a valid, nonzero normalized Task 2 suffix."""
    parent = build_parent_network(ParentConfig(), seed=4)
    corrected = migrate_online_network(parent, config=DEFAULT_CONFIG, seed=44)
    old_style = migrate_online_network(parent, config=DEFAULT_CONFIG, seed=44)
    seeded = build_q_network(DEFAULT_CONFIG, seed=44)

    old_style_input = [layer for layer in old_style.layers if hasattr(layer, "weight")][0]
    seeded_input = [layer for layer in seeded.layers if hasattr(layer, "weight")][0]
    with torch.no_grad():
        old_style_input.weight[:, PARENT_INPUT_DIM:].copy_(
            seeded_input.weight[:, PARENT_INPUT_DIM:]
        )

    for game_state in (task1_probe_state(), task1_probe_state(crate=True)):
        parent_features = parent_state_to_features(game_state)
        task2_features = task2_state_to_features(game_state)
        assert parent_features is not None
        assert task2_features is not None
        assert any(task2_features[PARENT_INPUT_DIM:])

        parent_state = torch.from_numpy(normalize_parent_features(parent_features))
        task2_state = torch.from_numpy(normalize_task2_features(task2_features))

        with torch.no_grad():
            parent_q_values = parent(parent_state)
            corrected_q_values = corrected(task2_state)[:PARENT_OUTPUT_DIM]
            old_style_q_values = old_style(task2_state)[:PARENT_OUTPUT_DIM]

        torch.testing.assert_close(
            corrected_q_values,
            parent_q_values,
            rtol=0.0,
            atol=Q_VALUE_TOLERANCE,
        )
        assert not torch.allclose(
            old_style_q_values,
            parent_q_values,
            rtol=0.0,
            atol=Q_VALUE_TOLERANCE,
        )


def test_new_input_columns_receive_gradients() -> None:
    parent = build_parent_network(ParentConfig(), seed=7)
    migrated = migrate_online_network(parent, config=DEFAULT_CONFIG, seed=44)
    input_layer = [layer for layer in migrated.layers if hasattr(layer, "weight")][0]
    state = torch.ones(DEFAULT_CONFIG.input_dim, dtype=torch.float32)

    migrated(state).sum().backward()

    assert input_layer.weight.requires_grad
    assert input_layer.weight.grad is not None
    assert bool(input_layer.weight.grad[:, PARENT_INPUT_DIM:].abs().any())


def test_bomb_does_not_displace_parent_action_on_task1_probes() -> None:
    """The appended BOMB action stays below the inherited greedy action."""
    parent = build_parent_network(ParentConfig(), seed=4)
    migrated = migrate_online_network(parent, config=DEFAULT_CONFIG, seed=44)

    for game_state in (task1_probe_state(), task1_probe_state(crate=True)):
        parent_features = parent_state_to_features(game_state)
        task2_features = task2_state_to_features(game_state)
        assert parent_features is not None
        assert task2_features is not None

        parent_state = normalize_parent_features(parent_features)
        task2_state = normalize_task2_features(task2_features)

        with torch.no_grad():
            parent_q_values = parent(torch.from_numpy(parent_state))
            migrated_q_values = migrated(torch.from_numpy(task2_state))

        assert migrated_q_values[PARENT_OUTPUT_DIM] < parent_q_values.max()
        assert select_action(
            network=migrated,
            state=task2_state,
            epsilon=0.0,
            rng=np.random.default_rng(91),
        ) == select_parent_action(
            network=parent,
            state=parent_state,
            epsilon=0.0,
            rng=np.random.default_rng(91),
        )


def test_migration_is_deterministically_repeatable() -> None:
    parent = build_parent_network(ParentConfig(), seed=5)

    first = migrate_online_network(parent, config=DEFAULT_CONFIG, seed=44)
    second = migrate_online_network(parent, config=DEFAULT_CONFIG, seed=44)

    first_state = first.state_dict()
    second_state = second.state_dict()

    for key in first_state:
        assert torch.equal(first_state[key], second_state[key])


def test_migration_produces_a_working_six_action_network() -> None:
    parent = build_parent_network(ParentConfig(), seed=6)
    migrated = migrate_online_network(parent, config=DEFAULT_CONFIG, seed=44)

    probe = torch.zeros(DEFAULT_CONFIG.input_dim)

    with torch.no_grad():
        q_values = migrated(probe)

    assert q_values.shape == (DEFAULT_CONFIG.output_dim,)
    assert bool(torch.isfinite(q_values).all())


def test_migration_rejects_a_parent_with_the_wrong_input_dim() -> None:
    incompatible_parent = build_mismatched_network(
        input_dim=9, output_dim=5, hidden_sizes=(64, 64)
    )

    with pytest.raises(ValueError, match="input_dim"):
        migrate_online_network(incompatible_parent, config=DEFAULT_CONFIG, seed=44)


def test_migration_rejects_a_parent_with_the_wrong_output_dim() -> None:
    incompatible_parent = build_mismatched_network(
        input_dim=8, output_dim=4, hidden_sizes=(64, 64)
    )

    with pytest.raises(ValueError, match="output_dim"):
        migrate_online_network(incompatible_parent, config=DEFAULT_CONFIG, seed=44)


def test_migration_rejects_a_parent_with_incompatible_hidden_sizes() -> None:
    incompatible_parent = build_mismatched_network(
        input_dim=8, output_dim=5, hidden_sizes=(32, 32)
    )

    with pytest.raises(ValueError, match="hidden_sizes"):
        migrate_online_network(incompatible_parent, config=DEFAULT_CONFIG, seed=44)


def test_migration_rejects_a_successor_config_with_different_hidden_sizes() -> None:
    parent = build_parent_network(ParentConfig(), seed=10)
    incompatible_successor_config = DQNConfig(hidden_sizes=(32, 32))

    with pytest.raises(ValueError, match="hidden sizes"):
        migrate_online_network(parent, config=incompatible_successor_config, seed=44)
