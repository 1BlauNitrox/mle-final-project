"""Plot the recorded Issue #204 shared-target comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from training.run_experiment import REPOSITORY_ROOT

EXPERIMENT = (
    REPOSITORY_ROOT
    / "experiments"
    / "2026-09-16-task3-shared-target-DerKleineKonkurrenzvernichter"
)
DEFAULT_RESULT = EXPERIMENT / "result.json"
DEFAULT_OUTPUT = EXPERIMENT / "figures"


def plot_results(
    result_path: Path = DEFAULT_RESULT,
    output_directory: Path = DEFAULT_OUTPUT,
) -> list[Path]:
    """Create hunting and retention/safety comparison figures."""
    result = json.loads(Path(result_path).read_text(encoding="utf-8"))
    summaries = result["summaries"]
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    return [
        _plot_hunting(summaries, output_directory),
        _plot_retention(summaries, output_directory),
        _plot_efficiency(result["training_diagnostics"], output_directory),
    ]


def _plot_hunting(summaries: list[dict[str, Any]], output: Path) -> Path:
    replicas = [f"r{index}" for index in range(1, 6)]
    x = np.arange(len(replicas))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for offset, treatment, label, color in (
        (-width / 2, "control", "Separate targets", "#4C78A8"),
        (width / 2, "candidate", "Shared target", "#F28E2B"),
    ):
        values = [
            _replica_value(summaries, treatment, replica, "peaceful", "mean_opponents_eliminated")
            for replica in replicas
        ]
        axis.bar(x + offset, values, width, label=label, color=color)
    axis.set_xticks(x, replicas)
    axis.set_ylabel("Mean opponents eliminated per episode")
    axis.set_xlabel("Matched training replica")
    axis.set_ylim(bottom=0)
    axis.set_title("Peaceful-opponent hunting")
    axis.legend()
    figure.tight_layout()
    path = output / "peaceful_hunting_comparison.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _plot_retention(summaries: list[dict[str, Any]], output: Path) -> Path:
    scenarios = ("classic", "coin-heaven", "loot-crate")
    labels = ("Classic", "Coin Heaven", "Loot Crate")
    x = np.arange(len(scenarios))
    width = 0.36
    figure, (collection_axis, safety_axis) = plt.subplots(1, 2, figsize=(11, 4.5))
    for offset, treatment, label, color in (
        (-width / 2, "control", "Separate targets", "#4C78A8"),
        (width / 2, "candidate", "Shared target", "#F28E2B"),
    ):
        collection_axis.bar(
            x + offset,
            [
                _mean(summaries, treatment, scenario, "mean_collection_fraction")
                for scenario in scenarios
            ],
            width,
            label=label,
            color=color,
        )
        safety_axis.bar(
            x + offset,
            [_mean(summaries, treatment, scenario, "self_kill_rate") for scenario in scenarios],
            width,
            label=label,
            color=color,
        )
    for axis in (collection_axis, safety_axis):
        axis.set_xticks(x, labels)
        axis.set_ylim(bottom=0)
        axis.legend()
    collection_axis.set_ylabel("Mean coin collection fraction")
    collection_axis.set_title("Task 2 retention")
    safety_axis.set_ylabel("Mean self-kill rate")
    safety_axis.set_title("Safety")
    figure.tight_layout()
    path = output / "retention_and_safety.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _plot_efficiency(diagnostics: list[dict[str, Any]], output: Path) -> Path:
    metrics = ("q_table_size", "mean_visits_per_state", "singleton_state_fraction")
    labels = ("Q-table states", "Mean visits/state", "Singleton fraction")
    figure, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for axis, metric, label in zip(axes, metrics, labels, strict=True):
        axis.bar(
            ("Separate", "Shared"),
            [
                fmean(float(row[metric]) for row in diagnostics if row["treatment"] == treatment)
                for treatment in ("control", "candidate")
            ],
            color=("#4C78A8", "#F28E2B"),
        )
        axis.set_title(label)
        axis.set_ylim(bottom=0)
    figure.suptitle("Training-state efficiency")
    figure.tight_layout()
    path = output / "learning_efficiency.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _mean(
    rows: list[dict[str, Any]], treatment: str, scenario: str, metric: str
) -> float:
    values = [
        float(row[metric])
        for row in rows
        if row["treatment"] == treatment and row["scenario"] == scenario
    ]
    if len(values) != 5:
        raise ValueError(f"Expected five values for {treatment}/{scenario}/{metric}")
    return fmean(values)


def _replica_value(
    rows: list[dict[str, Any]],
    treatment: str,
    replica: str,
    scenario: str,
    metric: str,
) -> float:
    values = [
        float(row[metric])
        for row in rows
        if row["treatment"] == treatment
        and row["replica"] == replica
        and row["scenario"] == scenario
    ]
    if len(values) != 1:
        raise ValueError(f"Expected one value for {treatment}/{replica}/{scenario}")
    return values[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)
    for path in plot_results(arguments.result, arguments.output):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
