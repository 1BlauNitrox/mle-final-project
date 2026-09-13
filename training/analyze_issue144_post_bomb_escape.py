"""Analyze the preregistered Issue #144 post-bomb escape experiment."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

from training.analyze_issue135_escape_distance import (
    DETERMINISTIC_COLUMNS,
    _evaluation_unseen_rate,
    _read_json,
    _suite_rows,
    _summarize,
    _training_diagnostic,
    _write_summary,
)
from training.paired_bootstrap import paired_bootstrap
from training.run_experiment import REPOSITORY_ROOT

PLAN_ROOT = REPOSITORY_ROOT / "training_outputs" / "run-plans"
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "experiments"
    / "2026-09-11-task2-post-bomb-escape-status-DerKleineSprengstoffkapitalist"
)
PLAN_IDS = {
    "control": "issue144-compact-task2-control",
    "candidate": "issue144-compact-task2-post-bomb-escape",
}
SUITES = {
    "classic-primary": "classic",
    "coin-heaven-primary": "coin-heaven",
    "loot-crate-primary": "loot-crate",
}


def analyze(plan_root: Path = PLAN_ROOT, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Require complete plans and evaluate every registered criterion."""
    plan_root = Path(plan_root).resolve()
    output = Path(output).resolve()
    rows: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    deterministic = True
    repeat_p95_ok = True
    repeat_max_ok = True

    for treatment, plan_id in PLAN_IDS.items():
        directory = plan_root / plan_id
        status = _read_json(directory / "status.json")
        resolved = _read_json(directory / "resolved_plan.json")
        if status.get("status") != "completed":
            raise ValueError(f"Run plan is not completed: {plan_id}")
        for replica in (item["replica_id"] for item in resolved["replicas"]):
            diagnostics.append(_training_diagnostic(directory, status, replica, treatment))
            for suite_id, scenario in SUITES.items():
                primary = _suite_rows(directory, status, resolved, replica, suite_id)
                repeat = _suite_rows(
                    directory, status, resolved, replica, suite_id.replace("-primary", "-repeat")
                )
                for first, second in zip(primary, repeat, strict=True):
                    if first["world_seed"] != second["world_seed"]:
                        raise ValueError("Repeat seed mismatch")
                    if any(
                        first.get(name) is None or second.get(name) is None
                        for name in DETERMINISTIC_COLUMNS
                    ):
                        raise ValueError("Missing deterministic evidence")
                    repeat_match = all(
                        first.get(name) == second.get(name)
                        for name in DETERMINISTIC_COLUMNS
                    )
                    deterministic &= repeat_match
                    repeat_p95_ok &= second["decision_time_p95_ms"] < 50
                    repeat_max_ok &= second["decision_time_max_ms"] < 100
                    available = first.get("initially_available_coins")
                    if not isinstance(available, int) or available <= 0:
                        raise ValueError("Missing available-coin count")
                    collection = first["coins_collected"] / available
                    rows.append(
                        {
                            **first,
                            "treatment": treatment,
                            "model": replica,
                            "scenario": scenario,
                            "collection_fraction": collection,
                        }
                    )
                    evidence.append(
                        {
                            "treatment": treatment,
                            "model": replica,
                            "scenario": scenario,
                            "world_seed": first["world_seed"],
                            "coins_collected": first["coins_collected"],
                            "initially_available_coins": available,
                            "collection_fraction": collection,
                            "self_kills": first["self_kills"],
                            "evaluation_decisions": first["evaluation_decisions"],
                            "evaluation_unseen_decisions": first[
                                "evaluation_unseen_decisions"
                            ],
                            "decision_time_p95_ms": first["decision_time_p95_ms"],
                            "decision_time_max_ms": first["decision_time_max_ms"],
                            "action_sequence_sha256": first[
                                "executed_action_sequence_sha256"
                            ],
                            "repeat_match": repeat_match,
                            "repeat_decision_time_p95_ms": second[
                                "decision_time_p95_ms"
                            ],
                            "repeat_decision_time_max_ms": second[
                                "decision_time_max_ms"
                            ],
                            "repeat_action_sequence_sha256": second[
                                "executed_action_sequence_sha256"
                            ],
                        }
                    )

    summaries = _summarize(rows)
    comparison = _classic_collection_comparison(rows)
    criteria = _criteria(
        summaries, comparison, diagnostics, deterministic, repeat_p95_ok, repeat_max_ok
    )
    result = {
        "schema_version": 1,
        "issue": 144,
        "deterministic": deterministic,
        "primary_evaluation_episodes": len(rows),
        "repeat_evaluation_episodes": len(rows),
        "summaries": summaries,
        "comparisons": {"classic_collection_candidate_minus_control": comparison},
        "criteria": criteria,
        "passed": all(criteria.values()),
        "training_diagnostics": diagnostics,
        "evaluation_unseen_state_rates": {
            treatment: _evaluation_unseen_rate(rows, treatment) for treatment in PLAN_IDS
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_summary(output / "summary.csv", summaries)
    _write_csv(output / "evidence.csv", evidence)
    _write_summary(output / "training_diagnostics.csv", diagnostics)
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def verify_from_evidence(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Recompute the decision from committed seed-level evidence."""
    output = Path(output).resolve()
    with (output / "evidence.csv").open(encoding="utf-8", newline="") as handle:
        raw_evidence = list(csv.DictReader(handle))
    with (output / "training_diagnostics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        raw_diagnostics = list(csv.DictReader(handle))
    if len(raw_evidence) != 1200 or len(raw_diagnostics) != 10:
        raise ValueError("Committed evidence has an unexpected row count")
    evidence = [
        {
            **row,
            "world_seed": int(row["world_seed"]),
            "collection_fraction": float(row["collection_fraction"]),
            "self_kills": int(row["self_kills"]),
            "decision_time_p95_ms": float(row["decision_time_p95_ms"]),
            "decision_time_max_ms": float(row["decision_time_max_ms"]),
            "repeat_decision_time_p95_ms": float(row["repeat_decision_time_p95_ms"]),
            "repeat_decision_time_max_ms": float(row["repeat_decision_time_max_ms"]),
            "repeat_match": row["repeat_match"] == "True",
        }
        for row in raw_evidence
    ]
    diagnostics = [
        {**row, "mean_visits_per_state": float(row["mean_visits_per_state"])}
        for row in raw_diagnostics
    ]
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in evidence:
        grouped[(row["treatment"], row["model"], row["scenario"])].append(row)
    summaries = [
        {
            "treatment": key[0],
            "model": key[1],
            "scenario": key[2],
            "mean_collection_fraction": fmean(row["collection_fraction"] for row in group),
            "self_kill_rate": fmean(row["self_kills"] for row in group),
            "decision_time_p95_ms": max(row["decision_time_p95_ms"] for row in group),
            "decision_time_max_ms": max(row["decision_time_max_ms"] for row in group),
        }
        for key, group in sorted(grouped.items())
    ]
    comparison = _classic_collection_comparison(evidence)
    criteria = _criteria(
        summaries,
        comparison,
        diagnostics,
        all(row["repeat_match"] for row in evidence),
        all(row["repeat_decision_time_p95_ms"] < 50 for row in evidence),
        all(row["repeat_decision_time_max_ms"] < 100 for row in evidence),
    )
    committed = _read_json(output / "result.json")
    expected = committed["comparisons"]["classic_collection_candidate_minus_control"]
    if comparison != expected or criteria != committed["criteria"]:
        raise ValueError("Recomputed evidence does not match committed result.json")
    return {"comparison": comparison, "criteria": criteria, "passed": all(criteria.values())}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _classic_collection_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, dict[str, dict[int, float]]] = defaultdict(dict)
    for row in rows:
        if row["scenario"] != "classic":
            continue
        model_values = values[row["treatment"]].setdefault(row["model"], {})
        seed = row["world_seed"]
        if seed in model_values:
            raise ValueError("Duplicate primary Classic episode")
        model_values[seed] = row["collection_fraction"]
    return paired_bootstrap(
        values["candidate"], values["control"], resampler_seed=144
    ).as_dict()


def _criteria(
    summaries: list[dict[str, Any]],
    comparison: dict[str, Any],
    diagnostics: list[dict[str, Any]],
    deterministic: bool,
    repeat_p95_ok: bool,
    repeat_max_ok: bool,
) -> dict[str, bool]:
    aggregate = _aggregate(summaries)
    control_collection = _replica_metric(summaries, "control", "mean_collection_fraction")
    candidate_collection = _replica_metric(
        summaries, "candidate", "mean_collection_fraction"
    )
    improved = sum(
        candidate_collection[model] > control_collection[model]
        for model in control_collection
    )
    mean_visits = {
        treatment: fmean(
            row["mean_visits_per_state"]
            for row in diagnostics
            if row["treatment"] == treatment
        )
        for treatment in PLAN_IDS
    }
    if mean_visits["control"] <= 0:
        raise ValueError("Control mean visits per state must be positive")
    candidate_classic = aggregate[("candidate", "classic")]
    control_classic = aggregate[("control", "classic")]
    return {
        "classic_collection_mean_above_control": comparison["mean_difference"] > 0,
        "classic_collection_ci_lower_above_zero": comparison["ci_lower"] > 0,
        "at_least_four_replicas_higher_classic_collection": improved >= 4,
        "candidate_classic_self_kill_at_most_0_055": (
            candidate_classic["self_kill_rate"] <= 0.055
        ),
        "candidate_classic_self_kill_not_above_control": (
            candidate_classic["self_kill_rate"] <= control_classic["self_kill_rate"]
        ),
        "coin_heaven_collection_decrease_within_0_05": (
            aggregate[("candidate", "coin-heaven")]["collection_fraction"]
            - aggregate[("control", "coin-heaven")]["collection_fraction"]
            >= -0.05
        ),
        "mean_visits_per_state_ratio_at_least_0_90": (
            mean_visits["candidate"] / mean_visits["control"] >= 0.90
        ),
        "deterministic_repeats": deterministic,
        "decision_time_p95_within_limit": (
            all(row["decision_time_p95_ms"] < 50 for row in summaries) and repeat_p95_ok
        ),
        "decision_time_max_within_limit": (
            all(row["decision_time_max_ms"] < 100 for row in summaries) and repeat_max_ok
        ),
    }


def _aggregate(
    summaries: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, float]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        grouped[(row["treatment"], row["scenario"])].append(row)
    return {
        key: {
            "collection_fraction": fmean(row["mean_collection_fraction"] for row in group),
            "self_kill_rate": fmean(row["self_kill_rate"] for row in group),
        }
        for key, group in grouped.items()
    }


def _replica_metric(
    summaries: list[dict[str, Any]], treatment: str, metric: str
) -> dict[str, float]:
    values = {
        row["model"]: row[metric]
        for row in summaries
        if row["treatment"] == treatment and row["scenario"] == "classic"
    }
    if len(values) != 5:
        raise ValueError("Expected five Classic replicas per treatment")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-root", type=Path, default=PLAN_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-from-evidence", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        result = (
            verify_from_evidence(arguments.output)
            if arguments.verify_from_evidence
            else analyze(arguments.plan_root, arguments.output)
        )
    except Exception as error:
        print(f"Analysis failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result["criteria"], indent=2, sort_keys=True))
    print(f"Overall pass: {result['passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
