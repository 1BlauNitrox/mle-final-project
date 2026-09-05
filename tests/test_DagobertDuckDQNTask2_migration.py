"""Tests for the Task 1 -> Task 2 checkpoint weight migration."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from agent_code.DagobertDuckDQN.config import DQNConfig as ParentConfig
from agent_code.DagobertDuckDQN.model import QNetwork
from agent_code.DagobertDuckDQN.model import build_q_network as build_parent_network
from agent_code.DagobertDuckDQNTask2.config import DEFAULT_CONFIG, DQNConfig
from agent_code.DagobertDuckDQNTask2.migration import (
    BOMB_OUTPUT_BIAS,
    PARENT_INPUT_DIM,
    PARENT_OUTPUT_DIM,
    migrate_online_network,
)
from agent_code.DagobertDuckDQNTask2.model import build_q_network


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


def test_migration_initializes_new_input_columns_from_a_fresh_seeded_network() -> None:
    parent = build_parent_network(ParentConfig(), seed=4)
    migrated = migrate_online_network(parent, config=DEFAULT_CONFIG, seed=44)
    reference = build_q_network(DEFAULT_CONFIG, seed=44)

    migrated_input = [layer for layer in migrated.layers if hasattr(layer, "weight")][0]
    reference_input = [layer for layer in reference.layers if hasattr(layer, "weight")][0]

    assert torch.equal(
        migrated_input.weight[:, PARENT_INPUT_DIM:],
        reference_input.weight[:, PARENT_INPUT_DIM:],
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
