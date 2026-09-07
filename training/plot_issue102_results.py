"""Create reproducible figures for Issue #102."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = (
    ROOT
    / "experiments"
    / "2026-09-06-task2-development-baseline-DerKleineSprengstoffkapitalist"
)
SUMMARY = EXPERIMENT / "summary.csv"
FIGURES = EXPERIMENT / "figures"


def read_rows() -> list[dict[str, str]]:
    with SUMMARY.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    rows = read_rows()
    FIGURES.mkdir(parents=True, exist_ok=True)

    trained_classic = sorted(
        (
            row
            for row in rows
            if row["treatment"] == "trained" and row["scenario"] == "classic"
        ),
        key=lambda row: row["model"],
    )
    untrained_classic = next(
        row
        for row in rows
        if row["treatment"] == "untrained" and row["scenario"] == "classic"
    )

    models = [row["model"] for row in trained_classic]
    x = np.arange(len(models))

    # Task 2 collection performance
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(
        x,
        [float(row["mean_collection_fraction"]) for row in trained_classic],
        color="#4C78A8",
        label="Trained replicas",
    )
    axis.axhline(
        float(untrained_classic["mean_collection_fraction"]),
        color="#E45756",
        linestyle="--",
        label="Untrained control",
    )
    axis.axhline(
        0.30,
        color="#F2CF5B",
        linestyle=":",
        label="Registered threshold",
    )
    axis.set_xticks(x, models)
    axis.set_ylabel("Mean hidden-coin collection fraction")
    axis.set_title("Task 2 classic performance")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURES / "classic-collection-by-replica.png", dpi=160)
    plt.close(figure)

    # Failure modes
    figure, axis = plt.subplots(figsize=(8, 4.5))
    width = 0.36
    axis.bar(
        x - width / 2,
        [float(row["self_kill_rate"]) for row in trained_classic],
        width,
        label="Self-kill rate",
        color="#E45756",
    )
    axis.bar(
        x + width / 2,
        [float(row["invalid_action_rate"]) for row in trained_classic],
        width,
        label="Invalid-action rate",
        color="#72B7B2",
    )
    axis.axhline(0.20, color="#E45756", linestyle=":", label="Self-kill limit")
    axis.set_xticks(x, models)
    axis.set_ylabel("Rate")
    axis.set_title("Task 2 failure modes")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURES / "classic-failure-modes.png", dpi=160)
    plt.close(figure)

    # Action distribution
    actions = ["up", "right", "down", "left", "wait", "bomb"]
    totals = np.array(
        [
            sum(float(row[f"action_{action}"]) for row in trained_classic)
            for action in actions
        ]
    )
    fractions = totals / totals.sum()

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar([action.upper() for action in actions], fractions, color="#59A14F")
    axis.set_ylabel("Fraction of actions")
    axis.set_title("Aggregate classic action distribution")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURES / "classic-action-distribution.png", dpi=160)
    plt.close(figure)

    print(f"Created figures in {FIGURES}")


if __name__ == "__main__":
    main()