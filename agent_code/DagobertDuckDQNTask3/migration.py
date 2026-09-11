"""One-way migration from the provisional Task 2 DQN into Task 3.

The six-action output contract and both hidden layers are unchanged. Only the
input expands from 21 or 26 to 39 features; the appended columns are zeroed so the
initial Task 3 network is function-preserving for the inherited feature
prefix.
"""

from __future__ import annotations

from dataclasses import asdict, fields

import torch
from torch import nn

from .config import DQNConfig
from .model import QNetwork, build_q_network

PARENT_INPUT_DIM = 21
PARENT_OUTPUT_DIM = 6
PARENT_HIDDEN_SIZES = (64, 64)
NEW_INPUT_DIM = 39
MIGRATION_INIT_SEED = 44
INHERITED_Q_VALUE_TOLERANCE = 1e-5


def migrate_online_network(
    parent_network: QNetwork,
    *,
    config: DQNConfig | None = None,
    seed: int = MIGRATION_INIT_SEED,
) -> QNetwork:
    """Build a Task 3 network with compatible Task 2 weights copied exactly."""
    parent_input_dim = parent_network.config.input_dim
    if parent_input_dim not in (21, 26):
        raise ValueError("Parent network input_dim must be 21 or 26")
    if config is None:
        config = successor_config(parent_network.config)
    if config.action_masking != parent_network.config.action_masking or (
        config.escape_continuation_features != parent_network.config.escape_continuation_features
    ):
        raise ValueError("Migration must preserve parent mask and escape modes")
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
    if config.input_dim != NEW_INPUT_DIM:
        raise ValueError(f"Task 3 config input_dim must be {NEW_INPUT_DIM}.")
    if tuple(config.hidden_sizes) != PARENT_HIDDEN_SIZES:
        raise ValueError(
            "Migration requires the inherited hidden sizes "
            f"{PARENT_HIDDEN_SIZES}; got {config.hidden_sizes}."
        )
    if config.output_dim != PARENT_OUTPUT_DIM:
        raise ValueError("Task 3 must retain the six-action output contract.")

    migrated = build_q_network(config, seed=seed)
    parent_layers = [layer for layer in parent_network.layers if isinstance(layer, nn.Linear)]
    migrated_layers = [layer for layer in migrated.layers if isinstance(layer, nn.Linear)]

    if len(parent_layers) != len(migrated_layers):
        raise ValueError("Migration requires the same number of linear layers.")

    with torch.no_grad():
        migrated_layers[0].weight[:, :parent_input_dim].copy_(parent_layers[0].weight)
        migrated_layers[0].weight[:, parent_input_dim:].zero_()
        migrated_layers[0].bias.copy_(parent_layers[0].bias)

        for parent_layer, migrated_layer in zip(
            parent_layers[1:], migrated_layers[1:], strict=True
        ):
            migrated_layer.weight.copy_(parent_layer.weight)
            migrated_layer.bias.copy_(parent_layer.bias)

    return migrated


def successor_config(parent_config) -> DQNConfig:
    """Keep inherited hyperparameters/modes; only expand the input schema."""
    values = asdict(parent_config)
    supported = {field.name for field in fields(DQNConfig)}
    return DQNConfig(
        **{
            **{key: value for key, value in values.items() if key in supported},
            "input_dim": NEW_INPUT_DIM,
        }
    )
