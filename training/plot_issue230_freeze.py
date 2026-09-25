"""Plot the Issue #230 final tabular confirmation."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = (
    ROOT
    / "experiments/2026-09-21-final-tabular-freeze-"
    "DerKleineKonkurrenzvernichter"
)
SUMMARY = EXPERIMENT / "summary.csv"
FIGURES = EXPERIMENT / "figures"


def load_summary(path: Path = SUMMARY) -> list[dict[str, str]]:
    """Load the durable aggregate table."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_confirmation(rows: list[dict[str, str]], output: Path) -> None:
    """Plot capability retention and safety across all suites."""
    suites = [row["suite"] for row in rows]
    collection = [float(row["collection_fraction"]) for row in rows]
    self_kills = [float(row["self_kill_rate"]) for row in rows]
    eliminations = [float(row["eliminations_per_round"]) for row in rows]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    positions = range(len(suites))
    axes[0].bar(positions, collection, color="#4C78A8")
    axes[0].set_title("Collection retention")
    axes[0].set_ylabel("Mean collection fraction")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xticks(positions, suites, rotation=35, ha="right")

    width = 0.38
    axes[1].bar(
        [position - width / 2 for position in positions],
        self_kills,
        width,
        label="Self-kill rate",
        color="#E45756",
    )
    axes[1].bar(
        [position + width / 2 for position in positions],
        eliminations,
        width,
        label="Eliminations / round",
        color="#59A14F",
    )
    axes[1].axhline(0.20, color="black", linestyle="--", label="Safety limit")
    axes[1].set_title("Safety and opponent elimination")
    axes[1].set_ylabel("Rate")
    axes[1].set_xticks(positions, suites, rotation=35, ha="right")
    axes[1].legend()
    figure.suptitle("Issue #230 final tabular confirmation")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    """Generate the committed confirmation figure."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    output = FIGURES / "capability_and_safety.png"
    plot_confirmation(load_summary(), output)
    print(output)


if __name__ == "__main__":
    main()
