"""Create figures for the Issue #110 legal-action-masking experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from training.run_experiment import REPOSITORY_ROOT

DEFAULT_EXPERIMENT_DIRECTORY = (
    REPOSITORY_ROOT
    / "experiments"
    / "2026-09-07-task2-legal-action-masking-DerKleineSprengstoffkapitalist"
)

TREATMENT_COLORS = {
    "unmasked": "#4C78A8",
    "masked": "#F58518",
}


def create_figures(
    experiment_directory: Path = DEFAULT_EXPERIMENT_DIRECTORY,
) -> list[Path]:
    """Create all registered Issue #110 figures."""

    experiment_directory = Path(experiment_directory).resolve()
    summary_path = experiment_directory / "summary.csv"
    evidence_path = experiment_directory / "evidence.csv"
    result_path = experiment_directory / "result.json"
    figure_directory = experiment_directory / "figures"

    summary = _read_csv(summary_path)
    evidence = _read_csv(evidence_path)
    result = _read_json(result_path)

    figure_directory.mkdir(parents=True, exist_ok=True)

    paths = [
        _plot_classic_collection_by_replica(
            summary,
            figure_directory,
        ),
        _plot_scenario_comparisons(
            result,
            figure_directory,
        ),
        _plot_safety_metrics(
            evidence,
            figure_directory,
        ),
    ]

    return paths


def _plot_classic_collection_by_replica(
    summary: list[dict[str, str]],
    output_directory: Path,
) -> Path:
    """Compare classic collection fractions for every trained replica."""

    replicas = ["r1", "r2", "r3", "r4", "r5"]

    unmasked = [
        100.0
        * _summary_value(
            summary,
            treatment="unmasked",
            model=replica,
            scenario="classic",
            metric="mean_collection_fraction",
        )
        for replica in replicas
    ]

    masked = [
        100.0
        * _summary_value(
            summary,
            treatment="masked",
            model=replica,
            scenario="classic",
            metric="mean_collection_fraction",
        )
        for replica in replicas
    ]

    positions = np.arange(len(replicas))
    width = 0.36

    figure, axis = plt.subplots(figsize=(8.5, 5.0))

    unmasked_bars = axis.bar(
        positions - width / 2,
        unmasked,
        width,
        label="Unmasked control",
        color=TREATMENT_COLORS["unmasked"],
    )
    masked_bars = axis.bar(
        positions + width / 2,
        masked,
        width,
        label="Framework-legal masking",
        color=TREATMENT_COLORS["masked"],
    )

    axis.bar_label(
        unmasked_bars,
        fmt="%.3f",
        padding=3,
        fontsize=8,
    )
    axis.bar_label(
        masked_bars,
        fmt="%.3f",
        padding=3,
        fontsize=8,
    )

    axis.set_title("Classic coin collection by training replica")
    axis.set_xlabel("Training replica")
    axis.set_ylabel("Mean collection fraction (%)")
    axis.set_xticks(positions, replicas)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)

    figure.tight_layout()

    output_path = output_directory / "classic_collection_by_replica.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    return output_path


def _plot_scenario_comparisons(
    result: dict[str, Any],
    output_directory: Path,
) -> Path:
    """Plot masked-minus-unmasked estimates and confidence intervals."""

    scenarios = ["classic", "coin-heaven", "loot-crate"]
    labels = ["Classic", "Coin Heaven", "Loot Crate"]

    comparisons = result["comparisons"]

    estimates: list[float] = []
    lower_errors: list[float] = []
    upper_errors: list[float] = []

    for scenario in scenarios:
        comparison = comparisons[f"{scenario}_masked_minus_unmasked"]

        mean = 100.0 * float(comparison["mean_difference"])
        lower = 100.0 * float(comparison["ci_lower"])
        upper = 100.0 * float(comparison["ci_upper"])

        estimates.append(mean)
        lower_errors.append(mean - lower)
        upper_errors.append(upper - mean)

    positions = np.arange(len(scenarios))

    figure, axis = plt.subplots(figsize=(8.5, 5.0))

    axis.errorbar(
        positions,
        estimates,
        yerr=np.asarray([lower_errors, upper_errors]),
        fmt="o",
        markersize=8,
        capsize=6,
        linewidth=2,
        color=TREATMENT_COLORS["masked"],
        ecolor="#444444",
    )

    axis.axhline(
        0.0,
        color="black",
        linewidth=1,
        linestyle="--",
    )

    axis.set_title("Effect of framework-legal action masking")
    axis.set_xlabel("Evaluation scenario")
    axis.set_ylabel("Masked − unmasked collection fraction\n(percentage points, 95% CI)")
    axis.set_xticks(positions, labels)
    axis.grid(axis="y", alpha=0.25)

    for position, estimate in zip(
        positions,
        estimates,
        strict=True,
    ):
        axis.annotate(
            f"{estimate:+.3f}",
            (position, estimate),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    figure.tight_layout()

    output_path = output_directory / "scenario_comparisons_ci.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    return output_path


def _plot_safety_metrics(
    evidence: list[dict[str, str]],
    output_directory: Path,
) -> Path:
    """Compare aggregate classic invalid-action and self-kill rates."""

    treatments = ["unmasked", "masked"]
    labels = ["Unmasked", "Masked"]

    invalid_rates: list[float] = []
    self_kill_rates: list[float] = []

    for treatment in treatments:
        rows = [
            row
            for row in evidence
            if (
                row["evaluation_pass"] == "primary"
                and row["treatment"] == treatment
                and row["scenario"] == "classic"
            )
        ]

        if not rows:
            raise ValueError(f"No classic primary evidence for {treatment}")

        attempted_actions = sum(int(row["attempted_actions"]) for row in rows)
        invalid_actions = sum(int(row["invalid_actions"]) for row in rows)
        self_kills = sum(int(row["self_kills"]) for row in rows)

        invalid_rates.append(
            100.0 * invalid_actions / attempted_actions if attempted_actions else 0.0
        )
        self_kill_rates.append(100.0 * self_kills / len(rows))

    positions = np.arange(len(treatments))
    width = 0.36

    figure, axis = plt.subplots(figsize=(8.5, 5.0))

    invalid_bars = axis.bar(
        positions - width / 2,
        invalid_rates,
        width,
        label="Invalid-action rate",
        color="#E45756",
    )
    self_kill_bars = axis.bar(
        positions + width / 2,
        self_kill_rates,
        width,
        label="Self-kill rate",
        color="#72B7B2",
    )

    axis.bar_label(
        invalid_bars,
        fmt="%.2f%%",
        padding=3,
        fontsize=9,
    )
    axis.bar_label(
        self_kill_bars,
        fmt="%.2f%%",
        padding=3,
        fontsize=9,
    )

    axis.set_title("Classic safety diagnostics")
    axis.set_xlabel("Treatment")
    axis.set_ylabel("Rate (%)")
    axis.set_xticks(positions, labels)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)

    figure.tight_layout()

    output_path = output_directory / "safety_metrics.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    return output_path


def _summary_value(
    rows: list[dict[str, str]],
    *,
    treatment: str,
    model: str,
    scenario: str,
    metric: str,
) -> float:
    """Read exactly one value from the aggregate summary."""

    matches = [
        row
        for row in rows
        if (row["treatment"] == treatment and row["model"] == model and row["scenario"] == scenario)
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected one summary row for {treatment}/{model}/{scenario}, got {len(matches)}"
        )

    return float(matches[0][metric])


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file as dictionaries."""

    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    """Read and validate a JSON object."""

    if not path.is_file():
        raise FileNotFoundError(path)

    value = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")

    return value


def parse_arguments(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-directory",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIRECTORY,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Create and report all Issue #110 figures."""

    arguments = parse_arguments(argv)
    paths = create_figures(arguments.experiment_directory)

    for path in paths:
        print(path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
