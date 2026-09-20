"""Grouped-CV training for the conservative learned attack gate."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from agent_code.DagobertDuckDQNAntiLoop.persistence import load_evaluation_checkpoint


@dataclass(frozen=True)
class AttackDataset:
    """Causal safe-bomb outcomes grouped by collection episode."""

    states: torch.Tensor
    labels: torch.Tensor
    groups: np.ndarray


def load_dataset(path: Path, *, input_dim: int) -> AttackDataset:
    """Load every registered row without outcome-based filtering."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != 1 or not isinstance(payload.get("records"), list):
        raise ValueError("Attack dataset has an invalid schema")
    records = payload["records"]
    states = np.asarray([row["features"] for row in records], dtype=np.float32)
    labels = np.asarray([row["label"] for row in records], dtype=np.float32)
    groups = np.asarray([row["episode_index"] for row in records], dtype=np.int64)
    if states.shape != (len(records), input_dim) or not np.all(np.isfinite(states)):
        raise ValueError("Attack dataset features are invalid")
    if not set(labels.tolist()) <= {0.0, 1.0} or labels.sum() < 2:
        raise ValueError("Attack dataset labels are invalid")
    if len(np.unique(groups)) < 2:
        raise ValueError("Attack dataset needs multiple episode groups")
    return AttackDataset(torch.from_numpy(states), torch.from_numpy(labels), groups)


def hidden_representation(parent: Path, states: torch.Tensor) -> torch.Tensor:
    """Use the frozen DQN representation without changing its parameters."""
    network = load_evaluation_checkpoint(parent).network
    network.eval()
    with torch.no_grad():
        hidden = network.layers[:-1](states)
    if hidden.ndim != 2 or not torch.isfinite(hidden).all():
        raise ValueError("Frozen DQN produced invalid hidden representations")
    return hidden


def _fit(
    values: torch.Tensor,
    labels: torch.Tensor,
    *,
    epochs: int,
    learning_rate: float,
    l2_weight: float,
    positive_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    mean = values.mean(dim=0)
    scale = values.std(dim=0, unbiased=False).clamp_min(1e-4)
    normalized = (values - mean) / scale
    weight = torch.zeros(values.shape[1], dtype=torch.float32, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam((weight, bias), lr=learning_rate)
    pos_weight = torch.tensor(positive_weight, dtype=torch.float32)
    for _ in range(epochs):
        logits = normalized @ weight + bias
        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
        loss = loss + l2_weight * weight.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    if not all(torch.isfinite(value).all() for value in (mean, scale, weight, bias)):
        raise ValueError("Attack classifier training produced non-finite parameters")
    return mean.detach(), scale.detach(), weight.detach(), float(bias.detach())


def _predict(
    values: torch.Tensor,
    parameters: tuple[torch.Tensor, torch.Tensor, torch.Tensor, float],
) -> np.ndarray:
    mean, scale, weight, bias = parameters
    with torch.no_grad():
        return torch.sigmoid(((values - mean) / scale) @ weight + bias).numpy()


def train_grouped_classifier(
    hidden: torch.Tensor,
    dataset: AttackDataset,
    *,
    folds: int,
    epochs: int,
    learning_rate: float,
    l2_weight: float,
    positive_weight: float,
    thresholds: list[float],
    minimum_precision: float,
    minimum_true_positives: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Select a fixed threshold from grouped out-of-fold predictions."""
    if folds < 2 or sorted(set(thresholds)) != thresholds:
        raise ValueError("Invalid fold or threshold configuration")
    fold_ids = dataset.groups % folds
    if set(fold_ids.tolist()) != set(range(folds)):
        raise ValueError("Every grouped fold must contain observations")
    oof = np.full(len(dataset.labels), np.nan, dtype=np.float64)
    fold_metrics = []
    for fold in range(folds):
        validation = fold_ids == fold
        training = ~validation
        train_labels = dataset.labels[torch.from_numpy(training)]
        validation_labels = dataset.labels[torch.from_numpy(validation)]
        if train_labels.sum() == 0 or validation_labels.sum() == 0:
            raise ValueError("Every validation fold must contain positive outcomes")
        parameters = _fit(
            hidden[torch.from_numpy(training)],
            train_labels,
            epochs=epochs,
            learning_rate=learning_rate,
            l2_weight=l2_weight,
            positive_weight=positive_weight,
        )
        probabilities = _predict(hidden[torch.from_numpy(validation)], parameters)
        oof[validation] = probabilities
        fold_metrics.append(
            {
                "fold": fold,
                "rows": int(validation.sum()),
                "positives": int(validation_labels.sum()),
            }
        )
    if not np.all(np.isfinite(oof)):
        raise ValueError("Incomplete grouped predictions")

    labels = dataset.labels.numpy().astype(bool)
    threshold_metrics = []
    for threshold in thresholds:
        predicted = oof >= threshold
        true_positive = int(np.count_nonzero(predicted & labels))
        false_positive = int(np.count_nonzero(predicted & ~labels))
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / int(np.count_nonzero(labels))
        beta2 = 0.25
        f_beta = (
            (1 + beta2) * precision * recall / (beta2 * precision + recall)
            if precision + recall
            else 0.0
        )
        threshold_metrics.append(
            {
                "threshold": threshold,
                "predicted": int(predicted.sum()),
                "true_positive": true_positive,
                "false_positive": false_positive,
                "precision": precision,
                "recall": recall,
                "f0_5": f_beta,
                "eligible": precision >= minimum_precision
                and true_positive >= minimum_true_positives,
            }
        )
    eligible = [row for row in threshold_metrics if row["eligible"]]
    report = {
        "rows": len(labels),
        "positives": int(labels.sum()),
        "folds": fold_metrics,
        "thresholds": threshold_metrics,
        "selected_threshold": None,
    }
    if not eligible:
        return None, report
    selected = max(eligible, key=lambda row: (row["f0_5"], row["precision"], row["threshold"]))
    report["selected_threshold"] = selected["threshold"]
    final = _fit(
        hidden,
        dataset.labels,
        epochs=epochs,
        learning_rate=learning_rate,
        l2_weight=l2_weight,
        positive_weight=positive_weight,
    )
    mean, scale, weight, bias = final
    return {
        "mean": mean,
        "scale": scale,
        "weight": weight,
        "bias": bias,
        "threshold": selected["threshold"],
    }, report


def save_artifact(
    path: Path,
    parameters: dict[str, Any],
    *,
    max_better_actions: int,
    parent_checkpoint_sha256: str,
) -> None:
    """Write the small self-contained evaluation artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": 1,
            **parameters,
            "max_better_actions": max_better_actions,
            "parent_checkpoint_sha256": parent_checkpoint_sha256,
        },
        path,
    )
