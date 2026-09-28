"""Analyze the preregistered Issue #210 five-step experiment."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

from training.aggregate import read_episodes_csv
from training.analyze_issue135_escape_distance import (
    DETERMINISTIC_COLUMNS,
    _read_json,
    _write_summary,
)
from training.analyze_issue186_task3_baseline import _suite_rows
from training.paired_bootstrap import paired_bootstrap
from training.run_experiment import REPOSITORY_ROOT

PLAN_ROOT = REPOSITORY_ROOT / "training_outputs" / "run-plans"
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "experiments"
    / "2026-09-19-task3-five-step-q-learning-DerKleineKonkurrenzvernichter"
)
PLAN_IDS = {
    "control": "issue210-task3-one-step",
    "candidate": "issue210-task3-five-step",
}
SUITES = {
    "peaceful-primary": "peaceful",
    "coincollector-primary": "coincollector",
    "classic-primary": "classic",
    "coin-heaven-primary": "coin-heaven",
    "loot-crate-primary": "loot-crate",
}


def analyze(plan_root: Path = PLAN_ROOT, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Require both completed plans and evaluate every registered gate."""
    plan_root = Path(plan_root).resolve()
    output = Path(output).resolve()
    evidence: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    deterministic = True
    latency_ok = True

    for treatment, plan_id in PLAN_IDS.items():
        directory = plan_root / plan_id
        status = _read_json(directory / "status.json")
        resolved = _read_json(directory / "resolved_plan.json")
        if status.get("status") != "completed":
            raise ValueError(f"Run plan is not completed: {plan_id}")
        for replica in (item["replica_id"] for item in resolved["replicas"]):
            diagnostics.append(
                _training_diagnostic(directory, status, replica, treatment)
            )
            for suite_id, scenario in SUITES.items():
                primary = _suite_rows(directory, status, resolved, replica, suite_id)
                repeat = _suite_rows(
                    directory,
                    status,
                    resolved,
                    replica,
                    suite_id.replace("-primary", "-repeat"),
                )
                for first, second in zip(primary, repeat, strict=True):
                    if first["world_seed"] != second["world_seed"]:
                        raise ValueError("Repeat seed mismatch")
                    repeat_match = all(
                        first.get(column) == second.get(column)
                        for column in DETERMINISTIC_COLUMNS
                    )
                    deterministic &= repeat_match
                    latency_ok &= all(
                        (
                            row["decision_time_p95_ms"] < 50
                            and row["decision_time_max_ms"] < 100
                        )
                        for row in (first, second)
                    )
                    available = int(first["initially_available_coins"])
                    evidence.append(
                        {
                            "treatment": treatment,
                            "replica": replica,
                            "scenario": scenario,
                            "world_seed": first["world_seed"],
                            "agent_seed": first["agent_seed"],
                            "collection_fraction": (
                                first["coins_collected"] / available if available else 0.0
                            ),
                            "opponents_eliminated": first["opponents_eliminated"],
                            "self_kills": first["self_kills"],
                            "score": first["score"],
                            "invalid_actions": first["invalid_actions"],
                            "decision_time_p95_ms": first["decision_time_p95_ms"],
                            "decision_time_max_ms": first["decision_time_max_ms"],
                            "action_sequence_sha256": first[
                                "executed_action_sequence_sha256"
                            ],
                            "repeat_match": repeat_match,
                        }
                    )

    summaries = summarize(evidence)
    comparison = peaceful_elimination_comparison(evidence)
    criteria = evaluate_criteria(summaries, comparison, deterministic, latency_ok)
    result = {
        "schema_version": 1,
        "issue": 210,
        "plan_ids": PLAN_IDS,
        "primary_evaluation_episodes": len(evidence),
        "repeat_evaluation_episodes": len(evidence),
        "summaries": summaries,
        "comparisons": {"peaceful_eliminations_candidate_minus_control": comparison},
        "criteria": criteria,
        "passed": all(criteria.values()),
        "training_diagnostics": diagnostics,
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "evidence.csv", evidence)
    _write_summary(output / "summary.csv", summaries)
    _write_summary(output / "training_diagnostics.csv", diagnostics)
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate the twenty primary seeds per treatment, replica, and scenario."""
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["treatment"], row["replica"], row["scenario"])].append(row)
    return [
        {
            "treatment": treatment,
            "replica": replica,
            "scenario": scenario,
            "episodes": len(group),
            "mean_collection_fraction": fmean(
                row["collection_fraction"] for row in group
            ),
            "mean_opponents_eliminated": fmean(
                row["opponents_eliminated"] for row in group
            ),
            "self_kill_rate": fmean(row["self_kills"] for row in group),
            "mean_score": fmean(row["score"] for row in group),
            "invalid_action_rate": fmean(row["invalid_actions"] for row in group),
            "decision_time_p95_ms": max(
                row["decision_time_p95_ms"] for row in group
            ),
            "decision_time_max_ms": max(
                row["decision_time_max_ms"] for row in group
            ),
        }
        for (treatment, replica, scenario), group in sorted(grouped.items())
    ]


def _training_diagnostic(
    plan_directory: Path,
    status: dict[str, Any],
    replica: str,
    treatment: str,
) -> dict[str, Any]:
    """Read Issue #210's peaceful-hunting final-checkpoint diagnostics."""
    job_id = f"train-{replica}-peaceful-hunting"
    job = status["jobs"].get(job_id)
    if not isinstance(job, dict):
        raise ValueError(f"Missing final training job: {job_id}")
    attempts = job.get("attempts", [])
    if job.get("status") != "completed" or not attempts:
        raise ValueError(f"Final training job is incomplete: {job_id}")
    run_directory = plan_directory / attempts[-1]["output"]
    rows = read_episodes_csv(run_directory / "episodes.csv")
    if not rows:
        raise ValueError(f"Final training job contains no episodes: {job_id}")
    final = max(rows, key=lambda row: row["round"])
    metrics = (
        "q_table_size",
        "total_state_visits",
        "mean_visits_per_state",
        "singleton_state_fraction",
    )
    if any(final.get(metric) is None for metric in metrics):
        raise ValueError(f"Final training diagnostics are incomplete: {job_id}")
    return {
        "treatment": treatment,
        "replica": replica,
        **{metric: final[metric] for metric in metrics},
    }


def peaceful_elimination_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the registered matched-replica/seed bootstrap comparison."""
    values: dict[str, dict[str, dict[int, float]]] = defaultdict(dict)
    for row in rows:
        if row["scenario"] != "peaceful":
            continue
        replica_values = values[row["treatment"]].setdefault(row["replica"], {})
        seed = int(row["world_seed"])
        if seed in replica_values:
            raise ValueError("Duplicate peaceful primary episode")
        replica_values[seed] = float(row["opponents_eliminated"])
    return paired_bootstrap(
        values["candidate"], values["control"], resampler_seed=210
    ).as_dict()


def evaluate_criteria(
    summaries: list[dict[str, Any]],
    comparison: dict[str, Any],
    deterministic: bool,
    latency_ok: bool,
) -> dict[str, bool]:
    """Evaluate the exact gates preregistered in config.yaml."""
    aggregate = _aggregate(summaries)
    peaceful = {
        treatment: {
            row["replica"]: row["mean_opponents_eliminated"]
            for row in summaries
            if row["treatment"] == treatment and row["scenario"] == "peaceful"
        }
        for treatment in PLAN_IDS
    }
    if any(len(values) != 5 for values in peaceful.values()):
        raise ValueError("Expected five peaceful replicas per treatment")
    self_kill_difference = fmean(
        aggregate[("candidate", scenario)]["self_kill_rate"]
        - aggregate[("control", scenario)]["self_kill_rate"]
        for scenario in SUITES.values()
    )
    return {
        "peaceful_elimination_mean_improves": comparison["mean_difference"] > 0,
        "at_least_four_replicas_improve_peaceful_elimination": sum(
            peaceful["candidate"][replica] > peaceful["control"][replica]
            for replica in peaceful["candidate"]
        )
        >= 4,
        "peaceful_elimination_ci_lower_at_least_minus_0_02": (
            comparison["ci_lower"] >= -0.02
        ),
        "classic_collection_difference_at_least_minus_0_05": (
            _difference(aggregate, "classic", "collection_fraction") >= -0.05
        ),
        "coin_heaven_collection_difference_at_least_minus_0_05": (
            _difference(aggregate, "coin-heaven", "collection_fraction") >= -0.05
        ),
        "loot_crate_collection_difference_at_least_minus_0_05": (
            _difference(aggregate, "loot-crate", "collection_fraction") >= -0.05
        ),
        "aggregate_self_kill_difference_at_most_0_03": self_kill_difference <= 0.03,
        "deterministic_repeats": deterministic,
        "latency_within_limits": latency_ok,
    }


def _aggregate(
    summaries: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, float]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        grouped[(row["treatment"], row["scenario"])].append(row)
    return {
        key: {
            "collection_fraction": fmean(
                row["mean_collection_fraction"] for row in group
            ),
            "self_kill_rate": fmean(row["self_kill_rate"] for row in group),
            "opponents_eliminated": fmean(
                row["mean_opponents_eliminated"] for row in group
            ),
        }
        for key, group in grouped.items()
    }


def _difference(
    aggregate: dict[tuple[str, str], dict[str, float]], scenario: str, metric: str
) -> float:
    return aggregate[("candidate", scenario)][metric] - aggregate[("control", scenario)][metric]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write empty Issue #210 evidence")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-root", type=Path, default=PLAN_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    result = analyze(arguments.plan_root, arguments.output)
    print(json.dumps(result["criteria"], indent=2, sort_keys=True))
    print(f"Overall pass: {result['passed']}")


if __name__ == "__main__":
    main()
