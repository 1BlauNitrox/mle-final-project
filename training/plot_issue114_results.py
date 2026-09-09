"""Create the committed figures for Issue #114."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from training.run_experiment import REPOSITORY_ROOT

DEFAULT_EXPERIMENT = (
    REPOSITORY_ROOT
    / "experiments"
    / "2026-09-07-task2-useful-bomb-reward-DerKleineSprengstoffkapitalist"
)


def plot(experiment: Path = DEFAULT_EXPERIMENT) -> None:
    experiment = Path(experiment).resolve()
    figures = experiment / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    result = json.loads((experiment / "result.json").read_text(encoding="utf-8"))
    summary = list(csv.DictReader((experiment / "summary.csv").open(encoding="utf-8")))
    training = list(csv.DictReader((experiment / "training-summary.csv").open(encoding="utf-8")))

    scenarios = ["classic", "coin-heaven", "loot-crate"]
    comparisons = [result["comparisons"][name] for name in scenarios]
    means = [item["mean_difference"] for item in comparisons]
    errors = [
        [mean - item["ci_lower"] for mean, item in zip(means, comparisons, strict=True)],
        [item["ci_upper"] - mean for mean, item in zip(means, comparisons, strict=True)],
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(scenarios, means, yerr=errors, fmt="o", capsize=5, color="#2864dc")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Candidate − control collection fraction")
    ax.set_title("Paired collection effect (95% bootstrap CI)")
    fig.tight_layout()
    fig.savefig(figures / "collection-effects.png", dpi=180)
    plt.close(fig)

    useful = {
        row["arm"]: float(row["mean_useful_bombs"]) for row in training if row["stage"] == "all"
    }
    crates = {
        row["arm"]: float(row["mean_crates_destroyed"])
        for row in summary
        if row["scenario"] == "classic"
    }
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    arms = ["control", "candidate"]
    axes[0].bar(arms, [useful[x] for x in arms], color=["#777777", "#2864dc"])
    axes[0].set_ylabel("Mean per training episode")
    axes[0].set_title("Useful bombs")
    axes[1].bar(arms, [crates[x] for x in arms], color=["#777777", "#2864dc"])
    axes[1].set_ylabel("Mean per classic episode")
    axes[1].set_title("Crates destroyed")
    fig.tight_layout()
    fig.savefig(figures / "bomb-behavior.png", dpi=180)
    plt.close(fig)

    self_kill = result["classic_self_kill_rates"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(arms, [self_kill[x] for x in arms], color=["#777777", "#d4483b"])
    ax.axhline(
        self_kill["control"] + 0.02, color="black", linestyle="--", label="control + 0.02 margin"
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Classic self-kill rate")
    ax.set_title("Safety regression")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "classic-self-kills.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    args = parser.parse_args()
    plot(args.experiment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
