"""One-way checkpoint migration from the frozen Task 1 network to Task 2.

Only the input and output layers change shape (8 -> 21 inputs, 5 -> 6
outputs); the hidden layers stay (64, 64) and are copied verbatim. New
parameters are not left at whatever a freshly seeded network happens to
produce: the new input columns come from that seeded network's own
initialization (documented, deterministic, reproducible), and the new BOMB
output row is deliberately overwritten to a fixed, conservative estimate
(see `BOMB_OUTPUT_BIAS` below) so a freshly migrated policy does not select
an untested action by chance of initialization.
"""

from __future__ import annotations

import torch
from torch import nn

from .config import DEFAULT_CONFIG, DQNConfig
from .model import QNetwork, build_q_network

PARENT_INPUT_DIM = 8
PARENT_OUTPUT_DIM = 5
PARENT_HIDDEN_SIZES = (64, 64)

MIGRATION_INIT_SEED = 44
BOMB_OUTPUT_BIAS = -1.0


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


def _migrate_input_layer(parent_layer: nn.Linear, migrated_layer: nn.Linear) -> None:
    """Copy matching input columns; leave new columns at their seeded init."""
    migrated_layer.weight[:, :PARENT_INPUT_DIM].copy_(parent_layer.weight)
    migrated_layer.bias.copy_(parent_layer.bias)


def _migrate_output_layer(parent_layer: nn.Linear, migrated_layer: nn.Linear) -> None:
    """Copy the five Task 1 output rows; initialize BOMB conservatively."""
    migrated_layer.weight[:PARENT_OUTPUT_DIM, :].copy_(parent_layer.weight)
    migrated_layer.bias[:PARENT_OUTPUT_DIM].copy_(parent_layer.bias)

    migrated_layer.weight[PARENT_OUTPUT_DIM, :].zero_()
    migrated_layer.bias[PARENT_OUTPUT_DIM] = BOMB_OUTPUT_BIAS
