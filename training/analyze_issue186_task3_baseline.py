"""Analyze the preregistered Issue #186 tabular Task 3 baseline."""

from __future__ import annotations

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
)
from training.run_experiment import REPOSITORY_ROOT

PLAN_ID = "issue186-tabular-task3-peaceful-baseline"
PLAN_ROOT = REPOSITORY_ROOT / "training_outputs" / "run-plans"
OUTPUT = REPOSITORY_ROOT / "experiments" / "2026-09-15-tabular-task3-peaceful-baseline"
SUITES = {
    "peaceful-primary": "peaceful",
    "coincollector-primary": "coincollector",
    "classic-primary": "classic",
    "coin-heaven-primary": "coin-heaven",
    "loot-crate-primary": "loot-crate",
}


def _suite_rows(
    plan_directory: Path,
    status: dict[str, Any],
    resolved: dict[str, Any],
    replica: str,
    suite_id: str,
) -> list[dict[str, Any]]:
    """Return only the observed agent row from each multi-agent episode."""
    prefix = f"eval-{replica}-{suite_id}-seed-"
    keys = sorted(key for key in status["jobs"] if key.startswith(prefix))
    expected = {
        job["run_id"]: job
        for job in resolved["jobs"]
        if job["kind"] == "evaluation"
        and job["replica"] == replica
        and job["stage_or_suite"] == suite_id
    }
    if set(keys) != set(expected):
        raise ValueError(f"Evaluation matrix mismatch for {prefix}")

    result: list[dict[str, Any]] = []
    for key in keys:
        job = status["jobs"][key]
        attempts = job.get("attempts", [])
        if job.get("status") != "completed" or not attempts:
            raise ValueError(f"Evaluation job is not complete: {key}")
        run_directory = plan_directory / attempts[-1]["output"]
        metadata = _read_json(run_directory / "metadata.json")
        registered = expected[key]
        for field in ("world_seed", "agent_seed", "scenario", "rounds"):
            if metadata.get(field) != registered[field]:
                raise ValueError(f"Metadata mismatch for {key}: {field}")
        if (
            metadata.get("mode") != "evaluation"
            or metadata.get("opponents") != registered["opponents"]
        ):
            raise ValueError(f"Metadata mismatch for {key}: evaluation conditions")

        observed_agent = metadata.get("observed_agent")
        episode_rows = [
            row
            for row in read_episodes_csv(run_directory / "episodes.csv")
            if row.get("agent") == observed_agent
        ]
        if len(episode_rows) != 1:
            raise ValueError(f"Expected one observed-agent episode row for {key}")
        row = dict(episode_rows[0])
        row["world_seed"] = metadata["world_seed"]
        row["agent_seed"] = metadata["agent_seed"]
        result.append(row)
    return result


def analyze(plan_root: Path = PLAN_ROOT, output: Path = OUTPUT) -> dict[str, Any]:
    directory = Path(plan_root).resolve() / PLAN_ID
    status = _read_json(directory / "status.json")
    resolved = _read_json(directory / "resolved_plan.json")
    if status.get("status") != "completed":
        raise ValueError("Issue #186 run plan is not completed")

    evidence: list[dict[str, Any]] = []
    deterministic = True
    repeat_latency_ok = True
    for replica in (item["replica_id"] for item in resolved["replicas"]):
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
                repeat_match = all(
                    first.get(column) == second.get(column)
                    for column in DETERMINISTIC_COLUMNS
                )
                deterministic &= repeat_match
                repeat_latency_ok &= (
                    second["decision_time_p95_ms"] < 50
                    and second["decision_time_max_ms"] < 100
                )
                available = int(first["initially_available_coins"])
                evidence.append(
                    {
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
    criteria = evaluate_criteria(summaries, deterministic, repeat_latency_ok)
    result = {
        "schema_version": 1,
        "issue": 186,
        "plan_id": PLAN_ID,
        "primary_episodes": len(evidence),
        "repeat_episodes": len(evidence),
        "summaries": summaries,
        "criteria": criteria,
        "passed": all(criteria.values()),
    }
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "evidence.csv", evidence)
    _write_csv(output / "summary.csv", summaries)
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["replica"], row["scenario"])].append(row)
    return [
        {
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
            "invalid_action_rate": sum(row["invalid_actions"] for row in group)
            / len(group),
            "decision_time_p95_ms": max(
                row["decision_time_p95_ms"] for row in group
            ),
            "decision_time_max_ms": max(
                row["decision_time_max_ms"] for row in group
            ),
        }
        for (replica, scenario), group in sorted(grouped.items())
    ]


def evaluate_criteria(
    summaries: list[dict[str, Any]],
    deterministic: bool,
    repeat_latency_ok: bool,
) -> dict[str, bool]:
    def mean(scenario: str, metric: str) -> float:
        return fmean(
            row[metric] for row in summaries if row["scenario"] == scenario
        )

    peaceful = [row for row in summaries if row["scenario"] == "peaceful"]
    return {
        "mean_peaceful_eliminations_at_least_0_20": (
            mean("peaceful", "mean_opponents_eliminated") >= 0.20
        ),
        "four_replicas_positive_peaceful_elimination": sum(
            row["mean_opponents_eliminated"] > 0 for row in peaceful
        )
        >= 4,
        "classic_collection_at_least_0_15": (
            mean("classic", "mean_collection_fraction") >= 0.15
        ),
        "coin_heaven_collection_at_least_0_90": (
            mean("coin-heaven", "mean_collection_fraction") >= 0.90
        ),
        "loot_crate_collection_at_least_0_20": (
            mean("loot-crate", "mean_collection_fraction") >= 0.20
        ),
        "aggregate_self_kill_at_most_0_15": (
            fmean(row["self_kill_rate"] for row in summaries) <= 0.15
        ),
        "deterministic_repeats": deterministic,
        "latency_within_limits": repeat_latency_ok
        and all(
            row["decision_time_p95_ms"] < 50
            and row["decision_time_max_ms"] < 100
            for row in summaries
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write empty Issue #186 evidence")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    result = analyze()
    print(json.dumps(result["criteria"], indent=2, sort_keys=True))
    print(f"Overall pass: {result['passed']}")


if __name__ == "__main__":
    main()
