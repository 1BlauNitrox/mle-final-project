"""Plot the recorded Issue #186 Task 3 baseline results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any

import matplotlib.pyplot as plt

from training.run_experiment import REPOSITORY_ROOT

EXPERIMENT = (
    REPOSITORY_ROOT
    / "experiments"
    / "2026-09-15-tabular-task3-peaceful-baseline"
)
DEFAULT_RESULT = EXPERIMENT / "result.json"
DEFAULT_OUTPUT = EXPERIMENT / "figures"


def plot_results(
    result_path: Path = DEFAULT_RESULT,
    output_directory: Path = DEFAULT_OUTPUT,
) -> list[Path]:
    """Create capability and regression figures from aggregated evidence."""
    result = json.loads(Path(result_path).read_text(encoding="utf-8"))
    summaries = result["summaries"]
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    return [
        _plot_hunting(summaries, output_directory),
        _plot_regressions(summaries, output_directory),
    ]


def _plot_hunting(
    summaries: list[dict[str, Any]], output_directory: Path
) -> Path:
    rows = sorted(
        (row for row in summaries if row["scenario"] == "peaceful"),
        key=lambda row: row["replica"],
    )
    if len(rows) != 5:
        raise ValueError("Expected five peaceful-opponent replica summaries")

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(
        [row["replica"] for row in rows],
        [row["mean_opponents_eliminated"] for row in rows],
        color="#4C78A8",
    )
    axis.axhline(0.20, color="#E45756", linestyle="--", label="Registered target")
    axis.set_ylabel("Mean opponents eliminated per episode")
    axis.set_xlabel("Training replica")
    axis.set_ylim(bottom=0)
    axis.set_title("Peaceful-opponent hunting capability")
    axis.legend()
    figure.tight_layout()
    path = output_directory / "peaceful_hunting.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _plot_regressions(
    summaries: list[dict[str, Any]], output_directory: Path
) -> Path:
    scenarios = ("classic", "coin-heaven", "loot-crate")
    labels = ("Classic", "Coin Heaven", "Loot Crate")
    collection = [_mean(summaries, scenario, "mean_collection_fraction") for scenario in scenarios]
    self_kills = [_mean(summaries, scenario, "self_kill_rate") for scenario in scenarios]

    figure, (collection_axis, safety_axis) = plt.subplots(1, 2, figsize=(10, 4.5))
    collection_axis.bar(labels, collection, color="#59A14F")
    collection_axis.set_ylabel("Mean coin collection fraction")
    collection_axis.set_ylim(bottom=0)
    collection_axis.set_title("Inherited-task retention")

    safety_axis.bar(labels, self_kills, color="#F28E2B")
    safety_axis.axhline(0.15, color="#E45756", linestyle="--", label="Aggregate limit")
    safety_axis.set_ylabel("Mean self-kill rate")
    safety_axis.set_ylim(bottom=0)
    safety_axis.set_title("Scenario safety")
    safety_axis.legend()

    figure.tight_layout()
    path = output_directory / "retention_and_safety.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _mean(
    summaries: list[dict[str, Any]], scenario: str, metric: str
) -> float:
    values = [row[metric] for row in summaries if row["scenario"] == scenario]
    if len(values) != 5:
        raise ValueError(f"Expected five values for {scenario}/{metric}")
    return fmean(values)


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
