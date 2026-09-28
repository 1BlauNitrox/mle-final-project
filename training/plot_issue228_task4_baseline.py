"""Plot the recorded Issue #228 Task 4 competitive baseline."""

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
    / "2026-09-20-task4-competitive-baseline-DerKleineKonkurrenzvernichter"
)
DEFAULT_RESULT = EXPERIMENT / "result.json"
DEFAULT_OUTPUT = EXPERIMENT / "figures"


def plot_results(
    result_path: Path = DEFAULT_RESULT,
    output_directory: Path = DEFAULT_OUTPUT,
) -> list[Path]:
    """Create competitive, retention, and safety figures."""

    result = json.loads(Path(result_path).read_text(encoding="utf-8"))
    summaries = result["summaries"]
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    return [
        _plot_competitive(summaries, output_directory),
        _plot_retention(summaries, output_directory),
        _plot_safety(summaries, output_directory),
    ]


def _plot_competitive(rows: list[dict[str, Any]], output: Path) -> Path:
    competitive = sorted(
        (row for row in rows if row["scenario"] == "competitive"),
        key=lambda row: row["replica"],
    )
    if len(competitive) != 5:
        raise ValueError("Expected five competitive replica summaries")
    labels = [row["replica"] for row in competitive]
    x = np.arange(len(labels))
    width = 0.38
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(
        x - width / 2,
        [row["first_place_rate"] for row in competitive],
        width,
        label="Strict first place",
        color="#4C78A8",
    )
    axis.bar(
        x + width / 2,
        [row["tied_first_rate"] for row in competitive],
        width,
        label="Tied first",
        color="#F2CF5B",
    )
    axis.axhline(0.10, color="#E45756", linestyle="--", label="Registered target")
    axis.set_xticks(x, labels)
    axis.set_ylabel("Episode rate")
    axis.set_xlabel("Training replica")
    axis.set_ylim(bottom=0)
    axis.set_title("Competitive outcomes against three rule-based agents")
    axis.legend()
    figure.tight_layout()
    path = output / "competitive_outcomes.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _plot_retention(rows: list[dict[str, Any]], output: Path) -> Path:
    scenarios = ("classic", "coin-heaven", "loot-crate")
    labels = ("Classic", "Coin Heaven", "Loot Crate")
    values = [_mean(rows, scenario, "mean_collection_fraction") for scenario in scenarios]
    thresholds = (0.15, 0.90, 0.20)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels))
    axis.bar(x, values, color="#59A14F")
    axis.scatter(x, thresholds, marker="_", s=800, color="#E45756", label="Gate")
    axis.set_xticks(x, labels)
    axis.set_ylabel("Mean coin collection fraction")
    axis.set_ylim(bottom=0)
    axis.set_title("Retained Task 2 collection")
    axis.legend()
    figure.tight_layout()
    path = output / "capability_retention.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _plot_safety(rows: list[dict[str, Any]], output: Path) -> Path:
    scenarios = ("competitive", "mixed", "peaceful", "classic", "coin-heaven", "loot-crate")
    labels = ("Competitive", "Mixed", "Peaceful", "Classic", "Coin Heaven", "Loot Crate")
    self_kills = [_mean(rows, scenario, "self_kill_rate") for scenario in scenarios]
    survival = [_mean(rows, scenario, "survival_rate") for scenario in scenarios]
    x = np.arange(len(labels))
    width = 0.38
    figure, axis = plt.subplots(figsize=(10, 4.8))
    axis.bar(x - width / 2, self_kills, width, label="Self-kill rate", color="#E15759")
    axis.bar(x + width / 2, survival, width, label="Survival rate", color="#76B7B2")
    axis.set_xticks(x, labels, rotation=20)
    axis.set_ylabel("Episode rate")
    axis.set_ylim(0, 1)
    axis.set_title("Safety and survival by evaluation suite")
    axis.legend()
    figure.tight_layout()
    path = output / "safety_and_survival.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _mean(rows: list[dict[str, Any]], scenario: str, metric: str) -> float:
    values = [float(row[metric]) for row in rows if row["scenario"] == scenario]
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
