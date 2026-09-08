"""Versioned one-way checkpoint migrations for the Task 2 DQN.

The original Task 1 migration grows the network from 8 inputs and 5 outputs
to the configured Task 2 shape. Issue #87 adds a second migration from the
corrected Issue #85 21-input artifact to the common 26-input architecture.
Both migrations zero every newly added input column so inherited action
values are preserved before learning. Hidden layers and compatible output rows
are copied verbatim; the new BOMB row remains conservative.
"""

from __future__ import annotations

import torch
from torch import nn

from .config import DEFAULT_CONFIG, FEATURE_COUNT, LEGACY_FEATURE_COUNT, DQNConfig
from .model import QNetwork, build_q_network

PARENT_INPUT_DIM = 8
PARENT_OUTPUT_DIM = 5
PARENT_HIDDEN_SIZES = (64, 64)
TASK2_OUTPUT_DIM = 6

MIGRATION_INIT_SEED = 44
BOMB_OUTPUT_BIAS = -1.0
INHERITED_Q_VALUE_TOLERANCE = 1e-5


def migrate_online_network(
    parent_network: QNetwork,
    *,
    config: DQNConfig = DEFAULT_CONFIG,
    seed: int = MIGRATION_INIT_SEED,
) -> QNetwork:
    """Build a Task 2 network whose compatible weights come from `parent_network`."""
    if parent_network.config.input_dim != PARENT_INPUT_DIM:
        raise ValueError(
            f"Parent network input_dim must be {PARENT_INPUT_DIM}, "
            f"got {parent_network.config.input_dim}."
        )

    if parent_network.config.output_dim != PARENT_OUTPUT_DIM:
        raise ValueError(
            f"Parent network output_dim must be {PARENT_OUTPUT_DIM}, "
            f"got {parent_network.config.output_dim}."
        )

    if tuple(parent_network.config.hidden_sizes) != PARENT_HIDDEN_SIZES:
        raise ValueError(
            f"Parent network hidden_sizes must be {PARENT_HIDDEN_SIZES}, "
            f"got {parent_network.config.hidden_sizes}."
        )

    if tuple(config.hidden_sizes) != PARENT_HIDDEN_SIZES:
        raise ValueError(
            "Migration only supports a Task 2 config with the same hidden "
            f"sizes as the parent, {PARENT_HIDDEN_SIZES}; "
            f"got {config.hidden_sizes}."
        )

    migrated = build_q_network(config, seed=seed)

    parent_layers = [layer for layer in parent_network.layers if isinstance(layer, nn.Linear)]
    migrated_layers = [layer for layer in migrated.layers if isinstance(layer, nn.Linear)]

    if len(parent_layers) != len(migrated_layers):
        raise ValueError("Migration requires the same number of linear layers.")

    with torch.no_grad():
        _migrate_input_layer(parent_layers[0], migrated_layers[0])

        for parent_layer, migrated_layer in zip(
            parent_layers[1:-1], migrated_layers[1:-1], strict=True
        ):
            migrated_layer.weight.copy_(parent_layer.weight)
            migrated_layer.bias.copy_(parent_layer.bias)

        _migrate_output_layer(parent_layers[-1], migrated_layers[-1])

    return migrated


def migrate_escape_continuation_network(
    task2_network: QNetwork,
    *,
    config: DQNConfig = DEFAULT_CONFIG,
    seed: int = MIGRATION_INIT_SEED,
) -> QNetwork:
    """Append issue #87 inputs to the corrected Issue #85 Task 2 network."""
    if task2_network.config.input_dim != LEGACY_FEATURE_COUNT:
        raise ValueError(
            "Issue #87 migration requires the corrected 21-feature Task 2 "
            f"network, got input_dim {task2_network.config.input_dim}."
        )

    if task2_network.config.output_dim != TASK2_OUTPUT_DIM:
        raise ValueError(
            f"Issue #87 migration requires {TASK2_OUTPUT_DIM} actions, "
            f"got {task2_network.config.output_dim}."
        )

    if config.input_dim != FEATURE_COUNT:
        raise ValueError(
            "Issue #87 migration requires the current 26-feature config, "
            f"got input_dim {config.input_dim}."
        )

    if tuple(task2_network.config.hidden_sizes) != PARENT_HIDDEN_SIZES:
        raise ValueError(
            "Migration requires the unchanged hidden sizes "
            f"{PARENT_HIDDEN_SIZES}."
        )

    if tuple(config.hidden_sizes) != PARENT_HIDDEN_SIZES:
        raise ValueError(
            "Migration requires the unchanged hidden sizes "
            f"{PARENT_HIDDEN_SIZES}."
        )

    migrated = build_q_network(config, seed=seed)
    source_layers = [
        layer for layer in task2_network.layers if isinstance(layer, nn.Linear)
    ]
    migrated_layers = [
        layer for layer in migrated.layers if isinstance(layer, nn.Linear)
    ]

    if len(source_layers) != len(migrated_layers):
        raise ValueError("Migration requires the same number of linear layers.")

    with torch.no_grad():
        migrated_layers[0].weight[:, :LEGACY_FEATURE_COUNT].copy_(
            source_layers[0].weight
        )
        migrated_layers[0].weight[:, LEGACY_FEATURE_COUNT:].zero_()
        migrated_layers[0].bias.copy_(source_layers[0].bias)

        for source_layer, migrated_layer in zip(
            source_layers[1:], migrated_layers[1:], strict=True
        ):
            migrated_layer.weight.copy_(source_layer.weight)
            migrated_layer.bias.copy_(source_layer.bias)

    return migrated


def _migrate_input_layer(parent_layer: nn.Linear, migrated_layer: nn.Linear) -> None:
    """Copy matching inputs and neutralize the Task 2-only contribution."""
    migrated_layer.weight[:, :PARENT_INPUT_DIM].copy_(parent_layer.weight)
    migrated_layer.weight[:, PARENT_INPUT_DIM:].zero_()
    migrated_layer.bias.copy_(parent_layer.bias)


def _migrate_output_layer(parent_layer: nn.Linear, migrated_layer: nn.Linear) -> None:
    """Copy the five Task 1 output rows; initialize BOMB conservatively."""
    migrated_layer.weight[:PARENT_OUTPUT_DIM, :].copy_(parent_layer.weight)
    migrated_layer.bias[:PARENT_OUTPUT_DIM].copy_(parent_layer.bias)

    migrated_layer.weight[PARENT_OUTPUT_DIM, :].zero_()
    migrated_layer.bias[PARENT_OUTPUT_DIM] = BOMB_OUTPUT_BIAS
