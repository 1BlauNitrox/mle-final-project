"""Generate plots from stored Bomberman episode metrics."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from training.aggregate import read_episodes_csv

ACTION_COLUMNS = (
    "action_up",
    "action_right",
    "action_down",
    "action_left",
    "action_wait",
    "action_bomb",
)

ACTION_LABELS = (
    "UP",
    "RIGHT",
    "DOWN",
    "LEFT",
    "WAIT",
    "BOMB",
)


def plot_run(
    run_directory: Path,
    *,
    rolling_window: int = 10,
) -> list[Path]:
    """Generate all experiment figures from a stored episodes.csv."""
    if rolling_window < 1:
        raise ValueError("Rolling window must be positive")

    run_directory = run_directory.resolve()
    episodes_path = run_directory / "episodes.csv"

    rows = read_episodes_csv(episodes_path)
    rows_by_agent = _group_rows_by_agent(rows)
    observed_agent = _load_observed_agent(run_directory)
    if observed_agent not in rows_by_agent:
        raise ValueError(
            f"Observed agent {observed_agent!r} has no episode rows"
        )
    observed_rows = rows_by_agent[observed_agent]

    figures_directory = run_directory / "figures"
    figures_directory.mkdir(parents=True, exist_ok=True)

    output_paths = [
        _plot_learning_curve(
            rows_by_agent,
            figures_directory / "learning_curve.png",
            rolling_window=rolling_window,
        ),
        _plot_task_metrics(
            rows_by_agent,
            figures_directory / "task_metrics.png",
        ),
        _plot_behavior_diagnostics(
            observed_rows,
            {observed_agent: observed_rows},
            figures_directory / "behavior_diagnostics.png",
        ),
    ]

    return output_paths


def _load_observed_agent(run_directory: Path) -> str:
    """Read the observed agent name from run metadata."""
    metadata_path = run_directory / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Run metadata do not exist: {metadata_path}")

    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)

    observed_agent = metadata.get("observed_agent", metadata.get("agent"))
    if not isinstance(observed_agent, str) or not observed_agent:
        raise ValueError("Run metadata contain no valid observed agent")
    return observed_agent


def _plot_learning_curve(
    rows_by_agent: Mapping[str, list[dict[str, Any]]],
    output_path: Path,
    *,
    rolling_window: int,
) -> Path:
    """Plot coins and score over episodes."""
    figure = Figure(figsize=(10, 6))
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1)

    for agent, rows in rows_by_agent.items():
        rounds = [row["round"] for row in rows]
        coins = [row["coins_collected"] for row in rows]
        scores = [row["score"] for row in rows]

        window = min(rolling_window, len(rows))

        axis.plot(
            rounds,
            coins,
            alpha=0.25,
            marker="o",
            linewidth=1,
            label=f"{agent}: coins",
        )
        axis.plot(
            rounds,
            _rolling_mean(coins, window),
            linewidth=2,
            label=f"{agent}: coins rolling mean",
        )
        axis.plot(
            rounds,
            scores,
            alpha=0.25,
            linestyle="--",
            marker="x",
            linewidth=1,
            label=f"{agent}: score",
        )
        axis.plot(
            rounds,
            _rolling_mean(scores, window),
            linestyle="--",
            linewidth=2,
            label=f"{agent}: score rolling mean",
        )

    axis.set_title("Episode performance")
    axis.set_xlabel("Episode")
    axis.set_ylabel("Coins / environment score")
    axis.grid(alpha=0.3)
    axis.legend(loc="best", fontsize="small")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    return output_path


def _plot_task_metrics(
    rows_by_agent: Mapping[str, list[dict[str, Any]]],
    output_path: Path,
) -> Path:
    """Plot episode length and invalid-action rate."""
    figure = Figure(figsize=(10, 6))
    FigureCanvasAgg(figure)

    steps_axis = figure.add_subplot(1, 1, 1)
    invalid_axis = steps_axis.twinx()

    for agent, rows in rows_by_agent.items():
        rounds = [row["round"] for row in rows]
        episode_steps = [row["episode_steps"] for row in rows]

        invalid_rounds = [
            row["round"]
            for row in rows
            if row["invalid_action_rate"] is not None
        ]
        invalid_rates = [
            row["invalid_action_rate"]
            for row in rows
            if row["invalid_action_rate"] is not None
        ]

        steps_axis.plot(
            rounds,
            episode_steps,
            marker="o",
            linewidth=2,
            label=f"{agent}: episode steps",
        )

        if invalid_rates:
            invalid_axis.plot(
                invalid_rounds,
                invalid_rates,
                marker="x",
                linestyle="--",
                linewidth=2,
                label=f"{agent}: invalid-action rate",
            )

    steps_axis.set_title("Task metrics")
    steps_axis.set_xlabel("Episode")
    steps_axis.set_ylabel("Episode steps")
    invalid_axis.set_ylabel("Invalid-action rate")
    invalid_axis.set_ylim(bottom=0)

    steps_axis.grid(alpha=0.3)

    handles_1, labels_1 = steps_axis.get_legend_handles_labels()
    handles_2, labels_2 = invalid_axis.get_legend_handles_labels()
    steps_axis.legend(
        handles_1 + handles_2,
        labels_1 + labels_2,
        loc="best",
        fontsize="small",
    )

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    return output_path


def _plot_behavior_diagnostics(
    rows: list[dict[str, Any]],
    rows_by_agent: Mapping[str, list[dict[str, Any]]],
    output_path: Path,
) -> Path:
    """Plot action counts and optional learning diagnostics."""
    figure = Figure(figsize=(10, 9))
    FigureCanvasAgg(figure)

    actions_axis = figure.add_subplot(2, 1, 1)
    learning_axis = figure.add_subplot(2, 1, 2)

    action_totals = [
        sum(row[column] for row in rows)
        for column in ACTION_COLUMNS
    ]

    actions_axis.bar(
        ACTION_LABELS,
        action_totals,
        color="tab:blue",
    )
    actions_axis.set_title("Action distribution")
    actions_axis.set_xlabel("Action")
    actions_axis.set_ylabel("Attempted actions")
    actions_axis.grid(axis="y", alpha=0.3)

    optional_metric_plotted = False

    for agent, agent_rows in rows_by_agent.items():
        epsilon_rows = [
            row for row in agent_rows
            if row.get("epsilon") is not None
        ]
        if epsilon_rows:
            learning_axis.plot(
                [row["round"] for row in epsilon_rows],
                [row["epsilon"] for row in epsilon_rows],
                marker="o",
                linewidth=2,
                label=f"{agent}: epsilon",
            )
            optional_metric_plotted = True

        q_table_rows = [
            row for row in agent_rows
            if row.get("q_table_size") is not None
        ]
        if q_table_rows:
            learning_axis.plot(
                [row["round"] for row in q_table_rows],
                [row["q_table_size"] for row in q_table_rows],
                marker="x",
                linestyle="--",
                linewidth=2,
                label=f"{agent}: Q-table size",
            )
            optional_metric_plotted = True

        td_error_rows = [
            row for row in agent_rows
            if row.get("mean_abs_td_error") is not None
        ]
        if td_error_rows:
            learning_axis.plot(
                [row["round"] for row in td_error_rows],
                [
                    row["mean_abs_td_error"]
                    for row in td_error_rows
                ],
                marker="s",
                linestyle=":",
                linewidth=2,
                label=f"{agent}: mean absolute TD error",
            )
            optional_metric_plotted = True

    learning_axis.set_title("Optional learning diagnostics")
    learning_axis.set_xlabel("Episode")
    learning_axis.grid(alpha=0.3)

    if optional_metric_plotted:
        learning_axis.legend(
            loc="best",
            fontsize="small",
        )
    else:
        learning_axis.text(
            0.5,
            0.5,
            "No optional learning metrics available",
            horizontalalignment="center",
            verticalalignment="center",
            transform=learning_axis.transAxes,
        )
        learning_axis.set_xticks([])
        learning_axis.set_yticks([])

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    return output_path


def _group_rows_by_agent(
    rows: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group episode rows by agent and order them by round."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        grouped[row["agent"]].append(row)

    return {
        agent: sorted(
            agent_rows,
            key=lambda row: row["round"],
        )
        for agent, agent_rows in sorted(grouped.items())
    }


def _rolling_mean(
    values: list[int | float],
    window: int,
) -> list[float]:
    """Return a rolling mean with a shorter window at the beginning."""
    if window < 1:
        raise ValueError("Rolling window must be positive")

    result = []

    for index in range(len(values)):
        start = max(0, index - window + 1)
        current_values = values[start : index + 1]
        result.append(
            sum(current_values) / len(current_values)
        )

    return result


def parse_arguments(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "run_directory",
        type=Path,
        help="Run directory containing episodes.csv",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=10,
        help="Rolling-mean window for learning curves",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the plotting command."""
    arguments = parse_arguments(argv)

    try:
        output_paths = plot_run(
            arguments.run_directory,
            rolling_window=arguments.rolling_window,
        )
    except Exception as error:
        print(f"Plotting failed: {error}")
        return 1

    for output_path in output_paths:
        print(f"Created {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
