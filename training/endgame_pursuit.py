"""Teacher generation and supervised fitting for an endgame pursuit head."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from agent_code.DagobertDuckDQNAntiLoop.config import ACTIONS
from agent_code.DagobertDuckDQNAntiLoop.endgame_pursuit_guard import pursuit_eligible
from agent_code.DagobertDuckDQNAntiLoop.features.bombs_and_crates import (
    blast_footprint,
    build_danger_map,
    safe_escape_exists,
)
from agent_code.DagobertDuckDQNAntiLoop.features.navigation import DIRECTIONS

ACTION_BY_DIRECTION = dict(zip(DIRECTIONS, ACTIONS[:4], strict=True))
BOMB_INDEX = ACTIONS.index("BOMB")


def _safe_attack_tile(field: np.ndarray, position: tuple[int, int], opponents: set) -> bool:
    if not opponents.intersection(blast_footprint(position, field)):
        return False
    bombs = [(position, 3)]
    danger = build_danger_map(field, bombs, np.zeros_like(field))
    return safe_escape_exists(field, danger, opponents | {position}, bombs, position)


def pursuit_teacher_action(game_state: dict, legal_mask: np.ndarray) -> str | None:
    """Return a public-state shortest-path action toward a safe attack tile."""
    if not pursuit_eligible(game_state):
        return None
    field = np.asarray(game_state["field"])
    start = tuple(int(value) for value in game_state["self"][3])
    opponents = {tuple(int(value) for value in other[3]) for other in game_state["others"]}
    blocked = set(opponents)
    queue = deque([start])
    parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    target = None
    while queue:
        position = queue.popleft()
        if _safe_attack_tile(field, position, opponents):
            target = position
            break
        for dx, dy in DIRECTIONS:
            neighbour = position[0] + dx, position[1] + dy
            if neighbour not in parents and neighbour not in blocked and field[neighbour] == 0:
                parents[neighbour] = position
                queue.append(neighbour)
    if target is None:
        return None
    if target == start:
        return "BOMB" if legal_mask[BOMB_INDEX] else None
    step = target
    while parents[step] != start:
        parent = parents[step]
        if parent is None:
            raise ValueError("Broken pursuit path")
        step = parent
    direction = step[0] - start[0], step[1] - start[1]
    action = ACTION_BY_DIRECTION[direction]
    return action if legal_mask[ACTIONS.index(action)] else None


@dataclass(frozen=True)
class PursuitDataset:
    """Teacher-labelled endgame states grouped by rollout episode."""

    inputs: torch.Tensor
    labels: torch.Tensor
    groups: np.ndarray


def build_dataset(rows: list[dict[str, Any]], *, minimum_examples: int, minimum_bombs: int):
    examples = []
    for episode_index, row in enumerate(rows):
        for example in row.get("pursuit_training_trace", ()):
            examples.append((episode_index, example))
    if len(examples) < minimum_examples:
        raise ValueError(f"Only {len(examples)} pursuit examples; need {minimum_examples}")
    values = np.asarray([example["input"] for _, example in examples], dtype=np.float32)
    labels = np.asarray([ACTIONS.index(example["teacher_action"]) for _, example in examples])
    groups = np.asarray([group for group, _ in examples], dtype=np.int64)
    if values.shape != (len(examples), 74) or not np.all(np.isfinite(values)):
        raise ValueError("Pursuit training inputs are invalid")
    if int(np.count_nonzero(labels == BOMB_INDEX)) < minimum_bombs:
        raise ValueError("Insufficient pursuit BOMB examples")
    return PursuitDataset(
        inputs=torch.from_numpy(values),
        labels=torch.from_numpy(labels),
        groups=groups,
    )


class PursuitNetwork(nn.Module):
    """Small classifier exported independently of the DQN checkpoint."""

    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(74, 32)
        self.output = nn.Linear(32, len(ACTIONS))

    def forward(self, values):
        return self.output(F.relu(self.hidden(values)))


def _fit(values, labels, *, epochs, learning_rate, l2_weight, seed):
    mean = values.mean(dim=0)
    scale = values.std(dim=0, unbiased=False).clamp_min(1e-4)
    normalized = (values - mean) / scale
    counts = torch.bincount(labels, minlength=len(ACTIONS)).float()
    weights = torch.where(counts > 0, len(labels) / (len(ACTIONS) * counts), 0.0).clamp_max(5.0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = PursuitNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_weight)
    for _ in range(epochs):
        loss = F.cross_entropy(model(normalized), labels, weight=weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    if not all(torch.isfinite(value).all() for value in model.parameters()):
        raise ValueError("Pursuit training produced non-finite parameters")
    return mean, scale, model


def _predict(values, parameters):
    mean, scale, model = parameters
    with torch.no_grad():
        return F.softmax(model((values - mean) / scale), dim=1).numpy()


def train_grouped_pursuit(
    dataset: PursuitDataset,
    *,
    folds: int,
    epochs: int,
    learning_rate: float,
    l2_weight: float,
    seed: int,
    thresholds: list[float],
    minimum_coverage: float,
    minimum_accuracy: float,
    minimum_bomb_precision: float,
    minimum_bomb_recall: float,
):
    """Fit grouped models, screen fixed confidence thresholds, then fit all data."""
    fold_ids = dataset.groups % folds
    probabilities = np.full((len(dataset.labels), len(ACTIONS)), np.nan)
    fold_report = []
    for fold in range(folds):
        validation = fold_ids == fold
        training = ~validation
        if not validation.any() or not training.any():
            raise ValueError("Every pursuit fold needs train and validation groups")
        parameters = _fit(
            dataset.inputs[torch.from_numpy(training)],
            dataset.labels[torch.from_numpy(training)],
            epochs=epochs,
            learning_rate=learning_rate,
            l2_weight=l2_weight,
            seed=seed + fold,
        )
        probabilities[validation] = _predict(
            dataset.inputs[torch.from_numpy(validation)], parameters
        )
        fold_report.append({"fold": fold, "rows": int(validation.sum())})
    labels = dataset.labels.numpy()
    prediction = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    threshold_report = []
    for threshold in thresholds:
        accepted = confidence >= threshold
        predicted_bomb = accepted & (prediction == BOMB_INDEX)
        actual_bomb = labels == BOMB_INDEX
        correct_bombs = int(np.count_nonzero(predicted_bomb & actual_bomb))
        coverage = float(accepted.mean())
        accuracy = (
            float((prediction[accepted] == labels[accepted]).mean()) if accepted.any() else 0.0
        )
        bomb_precision = correct_bombs / max(1, int(predicted_bomb.sum()))
        bomb_recall = correct_bombs / int(actual_bomb.sum())
        threshold_report.append(
            {
                "threshold": threshold,
                "coverage": coverage,
                "accuracy": accuracy,
                "bomb_precision": bomb_precision,
                "bomb_recall": bomb_recall,
                "eligible": coverage >= minimum_coverage
                and accuracy >= minimum_accuracy
                and bomb_precision >= minimum_bomb_precision
                and bomb_recall >= minimum_bomb_recall,
            }
        )
    accepted_thresholds = [row["threshold"] for row in threshold_report if row["eligible"]]
    report = {
        "rows": len(labels),
        "bomb_labels": int(np.count_nonzero(labels == BOMB_INDEX)),
        "folds": fold_report,
        "thresholds": threshold_report,
        "accepted_thresholds": accepted_thresholds,
    }
    if not accepted_thresholds:
        return None, report
    mean, scale, model = _fit(
        dataset.inputs,
        dataset.labels,
        epochs=epochs,
        learning_rate=learning_rate,
        l2_weight=l2_weight,
        seed=seed + folds,
    )
    parameters = {
        "mean": mean.detach(),
        "scale": scale.detach(),
        "weight1": model.hidden.weight.detach(),
        "bias1": model.hidden.bias.detach(),
        "weight2": model.output.weight.detach(),
        "bias2": model.output.bias.detach(),
    }
    return parameters, report


def save_artifact(path: Path, parameters, *, threshold, parent_checkpoint_sha256):
    """Write a self-contained learned pursuit policy artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": 1,
            **parameters,
            "threshold": threshold,
            "parent_checkpoint_sha256": parent_checkpoint_sha256,
        },
        path,
    )
