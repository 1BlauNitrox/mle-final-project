"""Checkpoint migration and schema contracts for Issue #87."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from agent_code.DagobertDuckDQNTask2.config import (
    DEFAULT_CONFIG,
    LEGACY_FEATURE_COUNT,
    DQNConfig,
)
from agent_code.DagobertDuckDQNTask2.migration import (
    migrate_escape_continuation_network,
)
from agent_code.DagobertDuckDQNTask2.model import QNetwork, build_q_network


def _linear_layers(network: QNetwork) -> list[torch.nn.Linear]:
    return [layer for layer in network.layers if isinstance(layer, torch.nn.Linear)]


def test_escape_migration_preserves_task2_function_and_zeroes_new_columns() -> None:
    source = build_q_network(DQNConfig(input_dim=LEGACY_FEATURE_COUNT), seed=85)
    migrated = migrate_escape_continuation_network(
        source, config=DEFAULT_CONFIG, seed=87
    )
    source_layers = _linear_layers(source)
    migrated_layers = _linear_layers(migrated)

    assert torch.equal(
        source_layers[0].weight,
        migrated_layers[0].weight[:, :LEGACY_FEATURE_COUNT],
    )
    assert torch.equal(
        migrated_layers[0].weight[:, LEGACY_FEATURE_COUNT:],
        torch.zeros_like(migrated_layers[0].weight[:, LEGACY_FEATURE_COUNT:]),
    )
    assert torch.equal(source_layers[0].bias, migrated_layers[0].bias)
    for source_layer, migrated_layer in zip(
        source_layers[1:], migrated_layers[1:], strict=True
    ):
        assert torch.equal(source_layer.weight, migrated_layer.weight)
        assert torch.equal(source_layer.bias, migrated_layer.bias)


def test_escape_migration_keeps_new_columns_trainable() -> None:
    source = build_q_network(DQNConfig(input_dim=LEGACY_FEATURE_COUNT), seed=85)
    migrated = migrate_escape_continuation_network(source, config=DEFAULT_CONFIG, seed=87)
    input_layer = _linear_layers(migrated)[0]

    migrated(torch.ones(DEFAULT_CONFIG.input_dim)).sum().backward()

    assert input_layer.weight.grad is not None
    assert bool(input_layer.weight.grad[:, LEGACY_FEATURE_COUNT:].abs().any())


def test_escape_migration_rejects_the_wrong_source_shape() -> None:
    incompatible = build_q_network(DEFAULT_CONFIG, seed=85)

    with pytest.raises(ValueError, match="21-feature"):
        migrate_escape_continuation_network(incompatible, config=DEFAULT_CONFIG)


def test_escape_migration_rejects_a_non_current_destination() -> None:
    source = build_q_network(DQNConfig(input_dim=LEGACY_FEATURE_COUNT), seed=85)
    legacy_destination = DQNConfig(input_dim=LEGACY_FEATURE_COUNT)

    with pytest.raises(ValueError, match="26-feature"):
        migrate_escape_continuation_network(source, config=legacy_destination)


def test_escape_migration_rejects_incompatible_hidden_sizes() -> None:
    source_config = SimpleNamespace(
        input_dim=LEGACY_FEATURE_COUNT,
        hidden_sizes=(32, 32),
        output_dim=6,
    )
    source = QNetwork(source_config)

    with pytest.raises(ValueError, match="hidden sizes"):
        migrate_escape_continuation_network(source, config=DEFAULT_CONFIG)
