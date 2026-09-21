"""Restricted offline fine-tuning for the DQN BOMB output only.

The hidden representation and every non-BOMB action value are immutable.  The
training signal combines native, owner-attributed kill outcomes with a narrow
geometric trapped-opponent teacher.  An anchor loss keeps the BOMB value close
to the parent on ordinary rollout states.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from agent_code.DagobertDuckDQNAntiLoop.config import ACTIONS
from agent_code.DagobertDuckDQNAntiLoop.persistence import load_evaluation_checkpoint

BOMB_INDEX = ACTIONS.index("BOMB")


@dataclass(frozen=True)
class HeadDataset:
    """Validated arrays used by the restricted optimizer."""

    positive: torch.Tensor
    negative: torch.Tensor
    anchors: torch.Tensor
    anchor_bomb_values: torch.Tensor
    empirical_positive_count: int
    trapped_teacher_count: int
    negative_count: int


def _states(records: list[dict[str, Any]], input_dim: int) -> torch.Tensor:
    values = np.asarray([record["features"] for record in records], dtype=np.float32)
    if values.shape != (len(records), input_dim) or not np.all(np.isfinite(values)):
        raise ValueError("Head-training feature matrix is invalid")
    return torch.from_numpy(values)


def build_dataset(
    rows: list[dict[str, Any]],
    *,
    input_dim: int,
    minimum_empirical_positives: int,
) -> HeadDataset:
    """Build causal positive/negative examples and ordinary-state anchors."""
    anchors: list[dict[str, Any]] = []
    bombs: list[dict[str, Any]] = []
    for row in rows:
        trace = row.get("head_training_trace")
        if not isinstance(trace, dict):
            raise ValueError("Collection row is missing its head-training trace")
        anchors.extend(trace.get("anchors", ()))
        bombs.extend(trace.get("bombs", ()))

    empirical = [
        row for row in bombs if int(row["kill_credits"]) > 0 and int(row["self_kill_credits"]) == 0
    ]
    trapped = [row for row in anchors if bool(row["trapped_attack"])]
    negative = [
        row for row in bombs if int(row["self_kill_credits"]) > 0 and int(row["kill_credits"]) == 0
    ]
    if len(empirical) < minimum_empirical_positives:
        raise ValueError(
            f"Only {len(empirical)} empirical positive bombs; need {minimum_empirical_positives}"
        )
    if not anchors:
        raise ValueError("No ordinary rollout anchors were collected")

    # Duplicate observations add no information and would overweight long loops.
    def unique(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        selected: dict[bytes, dict[str, Any]] = {}
        for record in records:
            key = np.asarray(record["features"], dtype=np.float32).tobytes()
            selected.setdefault(key, record)
        return list(selected.values())

    positive_records = unique([*empirical, *trapped])
    negative_records = unique(negative)
    anchor_records = unique(anchors)
    anchor_values = np.asarray(
        [record["q_values"][BOMB_INDEX] for record in anchor_records],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(anchor_values)):
        raise ValueError("Anchor Q-values are non-finite")
    negative_tensor = (
        _states(negative_records, input_dim)
        if negative_records
        else torch.empty((0, input_dim), dtype=torch.float32)
    )
    return HeadDataset(
        positive=_states(positive_records, input_dim),
        negative=negative_tensor,
        anchors=_states(anchor_records, input_dim),
        anchor_bomb_values=torch.from_numpy(anchor_values),
        empirical_positive_count=len(empirical),
        trapped_teacher_count=len(trapped),
        negative_count=len(negative),
    )


def _final_linear(network: nn.Module) -> tuple[str, nn.Linear]:
    matches = [
        (name, module) for name, module in network.named_modules() if isinstance(module, nn.Linear)
    ]
    if not matches or matches[-1][1].out_features != len(ACTIONS):
        raise ValueError("Could not identify the six-action output layer")
    return matches[-1]


def _sample(tensor: torch.Tensor, count: int, rng: np.random.Generator) -> torch.Tensor:
    if len(tensor) == 0:
        return tensor
    indices = rng.choice(len(tensor), size=count, replace=len(tensor) < count)
    return tensor[torch.from_numpy(np.asarray(indices, dtype=np.int64))]


def train_snapshots(
    parent: Path,
    dataset: HeadDataset,
    *,
    snapshot_steps: list[int],
    learning_rate: float,
    margin: float,
    anchor_weight: float,
    negative_weight: float,
    l2_weight: float,
    batch_size: int,
    seed: int,
) -> tuple[dict[int, dict[str, torch.Tensor]], list[dict[str, float]]]:
    """Return online-network snapshots while proving all protected rows stay exact."""
    if (
        snapshot_steps != sorted(set(snapshot_steps))
        or not snapshot_steps
        or snapshot_steps[0] <= 0
    ):
        raise ValueError("Snapshot steps must be unique positive integers")
    if min(learning_rate, margin, anchor_weight, negative_weight, l2_weight) < 0:
        raise ValueError("Fine-tuning hyperparameters must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    loaded = load_evaluation_checkpoint(parent)
    network = loaded.network
    network.train()
    layer_name, output = _final_linear(network)
    original = {name: value.detach().clone() for name, value in network.state_dict().items()}
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    output.weight.requires_grad_(True)
    output.bias.requires_grad_(True)
    optimizer = torch.optim.Adam((output.weight, output.bias), lr=learning_rate)
    rng = np.random.default_rng(seed)
    snapshots: dict[int, dict[str, torch.Tensor]] = {}
    history: list[dict[str, float]] = []
    weight_key, bias_key = f"{layer_name}.weight", f"{layer_name}.bias"
    original_weight = original[weight_key]
    original_bias = original[bias_key]

    for step in range(1, snapshot_steps[-1] + 1):
        positive = _sample(dataset.positive, batch_size, rng)
        anchor_indices = rng.choice(
            len(dataset.anchors), size=batch_size * 2, replace=len(dataset.anchors) < batch_size * 2
        )
        anchors = dataset.anchors[torch.from_numpy(np.asarray(anchor_indices, dtype=np.int64))]
        anchor_targets = dataset.anchor_bomb_values[
            torch.from_numpy(np.asarray(anchor_indices, dtype=np.int64))
        ]

        positive_values = network(positive)
        positive_best = positive_values[:, :BOMB_INDEX].max(dim=1).values.detach()
        positive_loss = F.softplus(positive_best + margin - positive_values[:, BOMB_INDEX]).mean()

        if len(dataset.negative):
            negative = _sample(dataset.negative, batch_size, rng)
            negative_values = network(negative)
            negative_best = negative_values[:, :BOMB_INDEX].max(dim=1).values.detach()
            negative_loss = F.softplus(
                negative_values[:, BOMB_INDEX] - negative_best + margin
            ).mean()
        else:
            negative_loss = positive_loss * 0.0

        anchor_predictions = network(anchors)[:, BOMB_INDEX]
        anchor_loss = F.mse_loss(anchor_predictions, anchor_targets)
        parameter_loss = F.mse_loss(output.weight[BOMB_INDEX], original_weight[BOMB_INDEX])
        parameter_loss = parameter_loss + F.mse_loss(
            output.bias[BOMB_INDEX], original_bias[BOMB_INDEX]
        )
        loss = (
            positive_loss
            + negative_weight * negative_loss
            + anchor_weight * anchor_loss
            + l2_weight * parameter_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not torch.isfinite(loss):
            raise ValueError("Non-finite BOMB-head loss")
        if output.weight.grad is None or output.bias.grad is None:
            raise ValueError("BOMB output did not receive a gradient")
        output.weight.grad[:BOMB_INDEX].zero_()
        output.bias.grad[:BOMB_INDEX].zero_()
        optimizer.step()

        with torch.no_grad():
            output.weight[:BOMB_INDEX].copy_(original_weight[:BOMB_INDEX])
            output.bias[:BOMB_INDEX].copy_(original_bias[:BOMB_INDEX])
        for name, value in network.state_dict().items():
            if name in {weight_key, bias_key}:
                protected = value[:BOMB_INDEX]
                expected = original[name][:BOMB_INDEX]
            else:
                protected, expected = value, original[name]
            if not torch.equal(protected, expected):
                raise ValueError(f"Protected parameter changed: {name}")

        history.append(
            {
                "step": float(step),
                "loss": float(loss.detach()),
                "positive_loss": float(positive_loss.detach()),
                "negative_loss": float(negative_loss.detach()),
                "anchor_loss": float(anchor_loss.detach()),
                "parameter_loss": float(parameter_loss.detach()),
            }
        )
        if step in snapshot_steps:
            snapshots[step] = deepcopy(network.state_dict())

    return snapshots, history


def install_online_state(parent: Path, destination: Path, state: dict[str, torch.Tensor]) -> None:
    """Create an evaluation checkpoint with only the online policy replaced."""
    payload = torch.load(parent, map_location="cpu", weights_only=True)
    payload["learner_state"]["online_network"] = deepcopy(state)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    loaded = load_evaluation_checkpoint(destination)
    actual = loaded.network.state_dict()
    if actual.keys() != state.keys() or any(not torch.equal(actual[k], state[k]) for k in state):
        raise ValueError("Saved BOMB-head checkpoint failed round-trip validation")
