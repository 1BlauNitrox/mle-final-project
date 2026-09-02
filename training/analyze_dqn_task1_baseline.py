"""Create the compact Issue #41 record from retained raw DQN outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from training.aggregate import read_episodes_csv
from training.evaluate_dqn_task1_baseline import (
    AGENT,
    DEFAULT_PROFILE,
    DEVELOPMENT_SEEDS,
    MODELS,
    PROFILES,
    ExperimentProfile,
    models_for,
)

COIN_HEAVEN_INITIAL_COINS = 50
ROLLING_WINDOW = 250
SUMMARY_FIELDS = (
    "model",
    "training_world_seed",
    "agent_seed",
    "training_episodes",
    "training_duration_seconds",
    "evaluation_episodes",
    "mean_coins",
    "mean_collection_fraction",
    "std_collection_fraction",
    "full_clear_count",
    "full_clear_rate",
    "zero_coin_count",
    "zero_coin_rate",
    "total_coins",
    "steps_per_coin",
    "coins_per_100_steps",
    "invalid_actions",
    "attempted_actions",
    "invalid_action_rate",
    "action_up",
    "action_right",
    "action_down",
    "action_left",
    "action_wait",
    "action_bomb",
    "decision_count",
    "decision_time_p95_ms",
    "max_decision_time_ms",
    "model_sha256",
)


def analyze(
    series_directory: Path,
    experiment_directory: Path,
    profile: ExperimentProfile = DEFAULT_PROFILE,
) -> dict[str, Any]:
    """Analyze retained outputs and write compact tables and figures."""
    models = models_for(profile)
    series_directory = series_directory.resolve()
    experiment_directory = experiment_directory.resolve()
    figures_directory = experiment_directory / "figures"
    figures_directory.mkdir(parents=True, exist_ok=True)

    manifest = _read_json(series_directory / "evaluation_manifest.json")
    if manifest.get("status") != "completed":
        raise ValueError("Evaluation manifest is not completed")

    training_runs = _completed_training_runs(series_directory, models)
    model_summaries: list[dict[str, Any]] = []
    evaluation_commits: set[str] = set()
    evaluation_duration_seconds = 0.0

    for model in models:
        model_name = f"run-{model.run:02d}"
        training = training_runs[model.agent_seed]
        rows: list[dict[str, Any]] = []
        decision_times: list[float] = []

        for seed in DEVELOPMENT_SEEDS:
            job = manifest["jobs"][f"{model_name}:primary:{seed}"]
            run_directory = series_directory / job["run_directory"]
            episode_rows = read_episodes_csv(run_directory / "episodes.csv")
            matching = [row for row in episode_rows if row["agent"] == AGENT]
            if len(matching) != 1:
                raise ValueError(f"Expected one DQN row for {model_name}/{seed}")
            row = dict(matching[0])
            statistics = _agent_round_statistics(
                run_directory / "framework_stats.json"
            )
            available = statistics["initially_available_coins"]
            row["initially_available_coins"] = available
            rows.append(row)
            decision_times.extend(statistics["decision_times_ms"])

            metadata = _read_json(run_directory / "metadata.json")
            evaluation_commits.add(metadata["git_commit"])
            evaluation_duration_seconds += metadata["duration_seconds"]

        model_summaries.append(
            _summarize_rows(
                model=model_name,
                rows=rows,
                decision_times=decision_times,
                training_world_seed=training["world_seed"],
                agent_seed=model.agent_seed,
                training_episodes=training["rounds"],
                training_duration_seconds=training["duration_seconds"],
                model_sha256=manifest["artifacts"][model_name]["sha256"],
            )
        )

    aggregate = _aggregate_summaries(model_summaries)
    _write_summary_csv(
        experiment_directory / "summary.csv",
        [*model_summaries, aggregate],
    )
    _plot_evaluation(model_summaries, figures_directory)
    _plot_failures(model_summaries, figures_directory)
    _plot_learning_curves(training_runs, figures_directory, models)

    criteria = _evaluate_criteria(model_summaries, aggregate, manifest)
    result = {
        "schema_version": 1,
        "series_directory": series_directory.name,
        "evaluation_manifest_sha256": _sha256(
            series_directory / "evaluation_manifest.json"
        ),
        "evaluation_commits": sorted(evaluation_commits),
        "evaluation_duration_seconds_primary": evaluation_duration_seconds,
        "evaluation_repeat_episodes": _completed_repeat_episodes(manifest, models),
        "criteria": criteria,
    }
    _write_json(experiment_directory / "result.json", result)
    return result


def _completed_repeat_episodes(
    manifest: dict[str, Any],
    models: tuple[Any, ...] = MODELS,
) -> int:
    """Count completed determinism-repeat jobs recorded in the manifest."""
    completed = sum(
        1
        for job in manifest["jobs"].values()
        if job.get("pass") == "repeat" and job.get("status") == "completed"
    )
    expected = len(models) * len(DEVELOPMENT_SEEDS)
    if completed != expected:
        raise ValueError(
            f"Expected {expected} completed repeat evaluations, found {completed}"
        )
    return completed


def _summarize_rows(
    *,
    model: str,
    rows: list[dict[str, Any]],
    decision_times: list[float],
    training_world_seed: int | str,
    agent_seed: int | str,
    training_episodes: int | str,
    training_duration_seconds: float | str,
    model_sha256: str,
) -> dict[str, Any]:
    fractions = np.asarray(
        [
            row["coins_collected"] / row["initially_available_coins"]
            for row in rows
        ],
        dtype=float,
    )
    total_coins = sum(row["coins_collected"] for row in rows)
    # docs/0007 defines the step basis for these metrics as survival_steps, not
    # episode_steps. The two coincide only while the agent cannot die.
    total_steps = sum(row["survival_steps"] for row in rows)
    attempted = sum(row["attempted_actions"] for row in rows)
    invalid = sum(row["invalid_actions"] for row in rows)
    decisions = np.asarray(decision_times, dtype=float)
    action_totals = {
        name: sum(row[name] for row in rows)
        for name in (
            "action_up",
            "action_right",
            "action_down",
            "action_left",
            "action_wait",
            "action_bomb",
        )
    }
    return {
        "model": model,
        "training_world_seed": training_world_seed,
        "agent_seed": agent_seed,
        "training_episodes": training_episodes,
        "training_duration_seconds": training_duration_seconds,
        "evaluation_episodes": len(rows),
        "mean_coins": float(np.mean([row["coins_collected"] for row in rows])),
        "mean_collection_fraction": float(np.mean(fractions)),
        "std_collection_fraction": float(np.std(fractions, ddof=1)),
        "full_clear_count": int(np.count_nonzero(fractions == 1.0)),
        "full_clear_rate": float(np.mean(fractions == 1.0)),
        "zero_coin_count": int(np.count_nonzero(fractions == 0.0)),
        "zero_coin_rate": float(np.mean(fractions == 0.0)),
        "total_coins": total_coins,
        "total_steps": total_steps,
        "steps_per_coin": total_steps / total_coins if total_coins else None,
        "coins_per_100_steps": (
            100.0 * total_coins / total_steps if total_steps else None
        ),
        "invalid_actions": invalid,
        "attempted_actions": attempted,
        "invalid_action_rate": invalid / attempted if attempted else 0.0,
        **action_totals,
        "decision_count": len(decisions),
        "decision_time_p95_ms": float(np.percentile(decisions, 95)),
        "max_decision_time_ms": float(np.max(decisions)),
        "model_sha256": model_sha256,
    }


def _aggregate_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    weighted_fields = (
        "total_coins",
        "total_steps",
        "invalid_actions",
        "attempted_actions",
        "action_up",
        "action_right",
        "action_down",
        "action_left",
        "action_wait",
        "action_bomb",
        "decision_count",
        "full_clear_count",
        "zero_coin_count",
    )
    totals = {field: sum(row[field] for row in summaries) for field in weighted_fields}
    episodes = sum(row["evaluation_episodes"] for row in summaries)
    total_steps = totals["total_steps"]
    # The combined sample moments can be recovered exactly from group moments.
    combined_mean = sum(
        row["mean_collection_fraction"] * row["evaluation_episodes"]
        for row in summaries
    ) / episodes
    combined_ss = sum(
        (row["evaluation_episodes"] - 1)
        * row["std_collection_fraction"] ** 2
        + row["evaluation_episodes"]
        * (row["mean_collection_fraction"] - combined_mean) ** 2
        for row in summaries
    )
    combined_std = math.sqrt(combined_ss / (episodes - 1))
    return {
        "model": "aggregate",
        "training_world_seed": "",
        "agent_seed": "",
        "training_episodes": sum(row["training_episodes"] for row in summaries),
        "training_duration_seconds": sum(
            row["training_duration_seconds"] for row in summaries
        ),
        "evaluation_episodes": episodes,
        "mean_coins": totals["total_coins"] / episodes,
        "mean_collection_fraction": combined_mean,
        "std_collection_fraction": combined_std,
        "full_clear_count": totals["full_clear_count"],
        "full_clear_rate": totals["full_clear_count"] / episodes,
        "zero_coin_count": totals["zero_coin_count"],
        "zero_coin_rate": totals["zero_coin_count"] / episodes,
        "total_coins": totals["total_coins"],
        "total_steps": total_steps,
        "steps_per_coin": (
            total_steps / totals["total_coins"] if totals["total_coins"] else None
        ),
        "coins_per_100_steps": (
            100.0 * totals["total_coins"] / total_steps if total_steps else None
        ),
        "invalid_actions": totals["invalid_actions"],
        "attempted_actions": totals["attempted_actions"],
        "invalid_action_rate": totals["invalid_actions"] / totals["attempted_actions"],
        **{field: totals[field] for field in weighted_fields if field.startswith("action_")},
        "decision_count": totals["decision_count"],
        "decision_time_p95_ms": max(
            row["decision_time_p95_ms"] for row in summaries
        ),
        "max_decision_time_ms": max(
            row["max_decision_time_ms"] for row in summaries
        ),
        "model_sha256": "",
    }


def _evaluate_criteria(
    summaries: list[dict[str, Any]],
    aggregate: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    models_above_threshold = sum(
        row["mean_collection_fraction"] >= 0.75 for row in summaries
    )
    maximum_model_invalid = max(row["invalid_action_rate"] for row in summaries)
    deterministic = all(record["deterministic"] for record in manifest["models"].values())
    immutable = all(record["immutable"] for record in manifest["models"].values())
    return {
        "aggregate_fraction": {
            "value": aggregate["mean_collection_fraction"],
            "threshold": ">= 0.80",
            "passed": aggregate["mean_collection_fraction"] >= 0.80,
        },
        "individual_models": {
            "value": models_above_threshold,
            "threshold": ">= 4 models at >= 0.75",
            "passed": models_above_threshold >= 4,
        },
        "paired_non_inferiority": {
            "value": None,
            "threshold": "95% CI lower bound > -0.02",
            "passed": None,
            "reason": "tabular per-seed rows and original artifacts unavailable",
        },
        "invalid_actions": {
            "value_aggregate": aggregate["invalid_action_rate"],
            "value_maximum_model": maximum_model_invalid,
            "threshold": "< 0.01",
            "passed": aggregate["invalid_action_rate"] < 0.01
            and maximum_model_invalid < 0.01,
        },
        "bomb_actions": {
            "value": aggregate["action_bomb"],
            "threshold": "= 0",
            "passed": aggregate["action_bomb"] == 0,
        },
        "deterministic": {"value": deterministic, "passed": deterministic},
        "immutable": {"value": immutable, "passed": immutable},
        "latency": {
            "value_p95_ms": aggregate["decision_time_p95_ms"],
            "value_max_ms": aggregate["max_decision_time_ms"],
            "threshold": "p95 < 50 ms and max < 100 ms",
            "passed": aggregate["decision_time_p95_ms"] < 50
            and aggregate["max_decision_time_ms"] < 100,
        },
    }


def _completed_training_runs(
    series_directory: Path,
    models: tuple[Any, ...] = MODELS,
) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for metadata_path in sorted(series_directory.glob("**/metadata.json")):
        if "evaluations" in metadata_path.parts:
            continue
        metadata = _read_json(metadata_path)
        if (
            metadata.get("status") == "completed"
            and metadata.get("mode") == "training"
            and metadata.get("rounds") == 10_000
        ):
            metadata["directory"] = metadata_path.parent
            records[metadata["agent_seed"]] = metadata
    expected = {model.agent_seed for model in models}
    if set(records) != expected:
        raise ValueError(f"Completed training seeds differ: {sorted(records)}")
    return records


def _agent_round_statistics(path: Path) -> dict[str, Any]:
    rounds = _read_json(path)["by_round"]
    if len(rounds) != 1:
        raise ValueError(f"Expected one round in {path}")
    agents = next(iter(rounds.values()))["agents"]
    return agents[AGENT]


def _plot_evaluation(
    summaries: list[dict[str, Any]], figures_directory: Path
) -> None:
    labels = [row["model"] for row in summaries]
    means = [row["mean_collection_fraction"] for row in summaries]
    standard_deviations = [row["std_collection_fraction"] for row in summaries]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(labels, means, yerr=standard_deviations, capsize=4)
    axis.axhline(0.75, color="tab:orange", linestyle="--", label="per-model gate")
    axis.axhline(0.80, color="tab:red", linestyle=":", label="aggregate gate")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Coin-collection fraction")
    axis.set_title("Registered Task 1 development evaluation")
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures_directory / "evaluation-performance.png", dpi=160)
    plt.close(figure)


def _plot_failures(
    summaries: list[dict[str, Any]], figures_directory: Path
) -> None:
    labels = [row["model"] for row in summaries]
    invalid = [row["invalid_action_rate"] for row in summaries]
    waits = [row["action_wait"] / row["attempted_actions"] for row in summaries]
    full_clears = [row["full_clear_rate"] for row in summaries]
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, values, title in zip(
        axes,
        (invalid, waits, full_clears),
        ("Invalid-action rate", "WAIT fraction", "Full-clear rate"),
        strict=True,
    ):
        axis.bar(labels, values)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=45)
    axes[0].axhline(0.01, color="tab:red", linestyle="--")
    figure.tight_layout()
    figure.savefig(figures_directory / "failure-modes.png", dpi=160)
    plt.close(figure)


def _plot_learning_curves(
    training_runs: dict[int, dict[str, Any]],
    figures_directory: Path,
    models: tuple[Any, ...] = MODELS,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    kernel = np.ones(ROLLING_WINDOW) / ROLLING_WINDOW
    for model in models:
        metadata = training_runs[model.agent_seed]
        rows = read_episodes_csv(metadata["directory"] / "episodes.csv")
        coins = np.asarray(
            [
                row["coins_collected"] / COIN_HEAVEN_INITIAL_COINS
                for row in rows
            ],
            dtype=float,
        )
        rolling = np.convolve(coins, kernel, mode="valid")
        episodes = np.arange(ROLLING_WINDOW, len(coins) + 1)
        axis.plot(episodes, rolling, label=f"run-{model.run:02d}", alpha=0.85)
    axis.axhline(0.80, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("Training episode")
    axis.set_ylabel(f"Coin fraction ({ROLLING_WINDOW}-episode mean)")
    axis.set_ylim(0, 1.05)
    axis.set_title("Complete registered DQN learning curves")
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(figures_directory / "learning-curves.png", dpi=160)
    plt.close(figure)


def _write_summary_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    # total_steps is an aggregation input rather than a reported metric, so it is
    # carried on the summary rows but excluded from the committed CSV schema.
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("series_directory", type=Path)
    parser.add_argument("experiment_directory", type=Path)
    parser.add_argument(
        "--issue",
        type=int,
        default=DEFAULT_PROFILE.issue,
        choices=sorted(PROFILES),
        help="Preregistered experiment profile the series belongs to",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    analyze(
        arguments.series_directory,
        arguments.experiment_directory,
        PROFILES[arguments.issue],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
