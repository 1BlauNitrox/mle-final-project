"""Analyze the preregistered Issue #228 Task 4 competitive baseline."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

from training.aggregate import read_episodes_csv
from training.analyze_issue135_escape_distance import DETERMINISTIC_COLUMNS, _read_json
from training.run_experiment import REPOSITORY_ROOT

PLAN_ID = "issue228-task4-competitive-baseline"
PLAN_ROOT = REPOSITORY_ROOT / "training_outputs" / "run-plans"
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "experiments"
    / "2026-09-20-task4-competitive-baseline-DerKleineKonkurrenzvernichter"
)
SUITES = {
    "competitive-primary": "competitive",
    "mixed-primary": "mixed",
    "peaceful-primary": "peaceful",
    "classic-primary": "classic",
    "coin-heaven-primary": "coin-heaven",
    "loot-crate-primary": "loot-crate",
}


def analyze(plan_root: Path = PLAN_ROOT, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Require the completed plan and evaluate every registered gate."""

    directory = Path(plan_root).resolve() / PLAN_ID
    status = _read_json(directory / "status.json")
    resolved = _read_json(directory / "resolved_plan.json")
    if status.get("status") != "completed":
        raise ValueError("Issue #228 run plan is not completed")

    evidence: list[dict[str, Any]] = []
    deterministic = True
    latency_ok = True
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
                if (
                    first["world_seed"] != second["world_seed"]
                    or first["agent_seed"] != second["agent_seed"]
                ):
                    raise ValueError("Repeat seed mismatch")
                repeat_match = all(
                    first.get(column) == second.get(column)
                    for column in DETERMINISTIC_COLUMNS
                )
                deterministic &= repeat_match
                latency_ok &= all(
                    row["decision_time_p95_ms"] < 50
                    and row["decision_time_max_ms"] < 100
                    for row in (first, second)
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
                        "survived": first["survived"],
                        "first_place": first["first_place"],
                        "tied_first": first["tied_first"],
                        "placement": first["placement"],
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
    uncertainty = competitive_first_place_interval(summaries)
    criteria = evaluate_criteria(summaries, deterministic, latency_ok)
    ranking = rank_replicas(summaries)
    passed = all(criteria.values())
    result = {
        "schema_version": 1,
        "issue": 228,
        "plan_id": PLAN_ID,
        "primary_evaluation_episodes": len(evidence),
        "repeat_evaluation_episodes": len(evidence),
        "summaries": summaries,
        "competitive_first_place_interval": uncertainty,
        "criteria": criteria,
        "passed": passed,
        "replica_ranking": ranking,
        "selected_replica": ranking[0] if passed else None,
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


def _suite_rows(
    plan_directory: Path,
    status: dict[str, Any],
    resolved: dict[str, Any],
    replica: str,
    suite_id: str,
) -> list[dict[str, Any]]:
    """Return observed-agent rows enriched with strict score placement."""

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
            raise ValueError(f"Metadata mismatch for {key}: conditions")

        observed_agent = metadata.get("observed_agent")
        rows = read_episodes_csv(run_directory / "episodes.csv")
        observed = [row for row in rows if row.get("agent") == observed_agent]
        if len(observed) != 1:
            raise ValueError(f"Expected one observed-agent row for {key}")
        row = dict(observed[0])
        opponent_scores = [
            int(other["score"])
            for other in rows
            if other.get("agent") != observed_agent
        ]
        row["placement"] = (
            1 + sum(score > int(row["score"]) for score in opponent_scores)
            if opponent_scores
            else None
        )
        row["world_seed"] = metadata["world_seed"]
        row["agent_seed"] = metadata["agent_seed"]
        result.append(row)
    return result


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate twenty primary episodes per replica and suite."""

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["replica"], row["scenario"])].append(row)
    result = []
    for (replica, scenario), group in sorted(grouped.items()):
        placements = [row["placement"] for row in group if row["placement"] is not None]
        first_places = [row["first_place"] for row in group if row["first_place"] is not None]
        tied_first = [row["tied_first"] for row in group if row["tied_first"] is not None]
        result.append(
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
                "survival_rate": fmean(row["survived"] for row in group),
                "first_place_rate": fmean(first_places) if first_places else None,
                "tied_first_rate": fmean(tied_first) if tied_first else None,
                "mean_placement": fmean(placements) if placements else None,
                "invalid_action_rate": fmean(row["invalid_actions"] for row in group),
                "decision_time_p95_ms": max(
                    row["decision_time_p95_ms"] for row in group
                ),
                "decision_time_max_ms": max(
                    row["decision_time_max_ms"] for row in group
                ),
            }
        )
    return result


def competitive_first_place_interval(
    summaries: list[dict[str, Any]],
) -> dict[str, float | int]:
    """Cluster-bootstrap the five competitive replica means."""

    values = np.asarray(
        [
            row["first_place_rate"]
            for row in summaries
            if row["scenario"] == "competitive"
        ],
        dtype=float,
    )
    if values.shape != (5,):
        raise ValueError("Expected five competitive replica summaries")
    rng = np.random.default_rng(228)
    samples = values[rng.integers(0, len(values), size=(10_000, len(values)))].mean(
        axis=1
    )
    return {
        "mean": float(values.mean()),
        "ci_lower": float(np.quantile(samples, 0.025)),
        "ci_upper": float(np.quantile(samples, 0.975)),
        "replica_clusters": 5,
        "resamples": 10_000,
        "seed": 228,
    }


def evaluate_criteria(
    summaries: list[dict[str, Any]],
    deterministic: bool,
    latency_ok: bool,
) -> dict[str, bool]:
    """Evaluate the exact gates registered before execution."""

    def mean(scenario: str, metric: str) -> float:
        values = [
            row[metric]
            for row in summaries
            if row["scenario"] == scenario and row[metric] is not None
        ]
        if len(values) != 5:
            raise ValueError(f"Expected five values for {scenario}/{metric}")
        return fmean(values)

    competitive = [row for row in summaries if row["scenario"] == "competitive"]
    return {
        "competitive_first_place_rate_at_least_0_10": (
            mean("competitive", "first_place_rate") >= 0.10
        ),
        "four_replicas_positive_first_place_rate": sum(
            row["first_place_rate"] > 0 for row in competitive
        )
        >= 4,
        "competitive_mean_score_positive": mean("competitive", "mean_score") > 0,
        "peaceful_eliminations_at_least_0_10": (
            mean("peaceful", "mean_opponents_eliminated") >= 0.10
        ),
        "classic_collection_at_least_0_15": (
            mean("classic", "mean_collection_fraction") >= 0.15
        ),
        "coin_heaven_collection_at_least_0_90": (
            mean("coin-heaven", "mean_collection_fraction") >= 0.90
        ),
        "loot_crate_collection_at_least_0_20": (
            mean("loot-crate", "mean_collection_fraction") >= 0.20
        ),
        "aggregate_self_kill_at_most_0_20": (
            fmean(row["self_kill_rate"] for row in summaries) <= 0.20
        ),
        "deterministic_repeats": deterministic,
        "latency_within_limits": latency_ok,
    }


def rank_replicas(summaries: list[dict[str, Any]]) -> list[str]:
    """Apply the prospectively registered competitive checkpoint ordering."""

    rows = [row for row in summaries if row["scenario"] == "competitive"]
    if len(rows) != 5:
        raise ValueError("Expected five competitive replica summaries")
    return [
        row["replica"]
        for row in sorted(
            rows,
            key=lambda row: (
                -row["first_place_rate"],
                -row["mean_score"],
                -row["survival_rate"],
                -row["mean_opponents_eliminated"],
                row["self_kill_rate"],
                row["replica"],
            ),
        )
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write empty Issue #228 evidence")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-root", type=Path, default=PLAN_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)
    result = analyze(arguments.plan_root, arguments.output)
    print(json.dumps(result["criteria"], indent=2, sort_keys=True))
    print(f"Selected replica: {result['selected_replica']}")
    print(f"Overall pass: {result['passed']}")


if __name__ == "__main__":
    main()
