"""Tests for grouped training and prospective result gates."""

import numpy as np
import torch

from training.learned_attack_gate import AttackDataset, train_grouped_classifier


def test_grouped_classifier_selects_only_registered_high_precision_threshold():
    values = []
    labels = []
    groups = []
    for group in range(5):
        values.extend(([2.0, 1.0], [-2.0, -1.0], [-1.5, -0.5]))
        labels.extend((1.0, 0.0, 0.0))
        groups.extend((group, group, group))
    dataset = AttackDataset(
        states=torch.zeros((15, 1), dtype=torch.float32),
        labels=torch.tensor(labels, dtype=torch.float32),
        groups=np.asarray(groups, dtype=np.int64),
    )
    parameters, report = train_grouped_classifier(
        torch.tensor(values, dtype=torch.float32),
        dataset,
        folds=5,
        epochs=100,
        learning_rate=0.05,
        l2_weight=0.001,
        positive_weight=1.0,
        thresholds=[0.5, 0.8],
        minimum_precision=0.9,
        minimum_true_positives=3,
    )
    assert parameters is not None
    assert report["selected_threshold"] in {0.5, 0.8}
    selected = next(
        row for row in report["thresholds"] if row["threshold"] == report["selected_threshold"]
    )
    assert selected["eligible"]
    assert selected["precision"] >= 0.9


def test_grouped_classifier_rejects_when_precision_gate_is_impossible():
    hidden = torch.zeros((10, 2), dtype=torch.float32)
    labels = torch.tensor([1.0, 0.0] * 5)
    dataset = AttackDataset(
        states=torch.zeros((10, 1)),
        labels=labels,
        groups=np.repeat(np.arange(5), 2),
    )
    parameters, report = train_grouped_classifier(
        hidden,
        dataset,
        folds=5,
        epochs=5,
        learning_rate=0.01,
        l2_weight=0.0,
        positive_weight=1.0,
        thresholds=[0.5],
        minimum_precision=1.0,
        minimum_true_positives=6,
    )
    assert parameters is None
    assert report["selected_threshold"] is None
