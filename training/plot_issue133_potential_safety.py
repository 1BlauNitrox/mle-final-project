"""Plot the recorded results of the Issue #133 potential-safety experiment."""

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
    / "2026-09-10-task2-compact-potential-safety-DerKleineSprengstoffkapitalist"
)
DEFAULT_RESULT = EXPERIMENT / "result.json"
DEFAULT_OUTPUT = EXPERIMENT / "figures"

TREATMENTS = ("control", "candidate")
LABELS = {
    "control": "No shaping",
    "candidate": "Safety shaping",
}
COLORS = {
    "control": "#4C78A8",
    "candidate": "#F58518",
}


def plot_results(
    result_path: Path = DEFAULT_RESULT,
    output_directory: Path = DEFAULT_OUTPUT,
) -> list[Path]:
    """Create the registered performance and efficiency figures."""

    result = json.loads(Path(result_path).read_text(encoding="utf-8"))
    summaries = result["summaries"]
    diagnostics = result["training_diagnostics"]

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    paths = [
        _plot_performance(summaries, output_directory),
        _plot_efficiency(diagnostics, output_directory),
    ]

    return paths


def _plot_performance(
    summaries: list[dict[str, Any]],
    output_directory: Path,
) -> Path:
    scenarios = ("classic", "coin-heaven", "loot-crate")
    scenario_labels = ("Classic", "Coin Heaven", "Loot Crate")
    positions = range(len(scenarios))
    width = 0.34

    figure, (collection_axis, safety_axis) = plt.subplots(
        1,
        2,
        figsize=(11, 4.5),
    )

    for offset, treatment in zip(
        (-width / 2, width / 2),
        TREATMENTS,
        strict=True,
    ):
        values = [
            _mean_summary_metric(
                summaries,
                treatment,
                scenario,
                "mean_collection_fraction",
            )
            for scenario in scenarios
        ]
        collection_axis.bar(
            [position + offset for position in positions],
            values,
            width,
            label=LABELS[treatment],
            color=COLORS[treatment],
        )

    collection_axis.set_xticks(list(positions), scenario_labels)
    collection_axis.set_ylabel("Mean coin collection fraction")
    collection_axis.set_ylim(bottom=0)
    collection_axis.set_title("Evaluation performance")
    collection_axis.legend()

    classic_self_kills = [
        _mean_summary_metric(
            summaries,
            treatment,
            "classic",
            "self_kill_rate",
        )
        for treatment in TREATMENTS
    ]
    safety_axis.bar(
        [LABELS[treatment] for treatment in TREATMENTS],
        classic_self_kills,
        color=[COLORS[treatment] for treatment in TREATMENTS],
    )
    safety_axis.set_ylabel("Classic self-kill rate")
    safety_axis.set_ylim(bottom=0)
    safety_axis.set_title("Classic safety outcome")

    figure.tight_layout()
    path = output_directory / "performance_and_safety.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _plot_efficiency(
    diagnostics: list[dict[str, Any]],
    output_directory: Path,
) -> Path:
    metrics = (
        ("q_table_size", "Materialized states"),
        ("mean_visits_per_state", "Mean visits per state"),
        ("singleton_state_fraction", "Singleton-state fraction"),
        ("q_table_artifact_size_bytes", "Artifact size (KiB)"),
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(10, 7),
    )

    for axis, (metric, title) in zip(
        axes.flat,
        metrics,
        strict=True,
    ):
        values = []

        for treatment in TREATMENTS:
            treatment_values = [
                row[metric]
                for row in diagnostics
                if row["treatment"] == treatment
            ]

            if not treatment_values:
                raise ValueError(
                    f"Missing {metric} diagnostics for {treatment}"
                )

            value = fmean(treatment_values)

            if metric == "q_table_artifact_size_bytes":
                value /= 1024

            values.append(value)

        axis.bar(
            [LABELS[treatment] for treatment in TREATMENTS],
            values,
            color=[COLORS[treatment] for treatment in TREATMENTS],
        )
        axis.set_title(title)
        axis.set_ylim(bottom=0)

    figure.suptitle("Compact-state learning efficiency")
    figure.tight_layout()
    path = output_directory / "learning_efficiency.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _mean_summary_metric(
    summaries: list[dict[str, Any]],
    treatment: str,
    scenario: str,
    metric: str,
) -> float:
    values = [
        row[metric]
        for row in summaries
        if (
            row["treatment"] == treatment
            and row["scenario"] == scenario
        )
    ]

    if len(values) != 5:
        raise ValueError(
            f"Expected five values for {treatment}/{scenario}/{metric}"
        )

    return fmean(values)


def parse_arguments(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        type=Path,
        default=DEFAULT_RESULT,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    paths = plot_results(arguments.result, arguments.output)

    for path in paths:
        print(path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())