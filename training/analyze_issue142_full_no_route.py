"""Analyze the preregistered Issue #142 full no-route penalty experiment."""

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
    / "2026-09-11-task2-full-no-route-penalty-DerKleineSprengstoffkapitalist"
)
PLAN_IDS = {
    "control": "issue142-compact-task2-uniform-half",
    "candidate": "issue142-compact-task2-full-no-route",
}
SUITES = {
    "classic-primary": "classic",
    "coin-heaven-primary": "coin-heaven",
    "loot-crate-primary": "loot-crate",
}


def analyze(plan_root: Path = PLAN_ROOT, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Require completed plans and evaluate all preregistered gates."""
    rows: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    deterministic = True
    repeat_p95_ok = True
    repeat_max_ok = True
    plan_root = Path(plan_root).resolve()
    output = Path(output).resolve()

    for treatment, plan_id in PLAN_IDS.items():
        directory = plan_root / plan_id
        status = _read_json(directory / "status.json")
        resolved = _read_json(directory / "resolved_plan.json")
        if status.get("status") != "completed":
            raise ValueError(f"Run plan is not completed: {plan_id}")
        for replica in (item["replica_id"] for item in resolved["replicas"]):
            diagnostics.append(_training_diagnostic(directory, status, replica, treatment))
            for suite_id, scenario in SUITES.items():
                primary_rows = _suite_rows(directory, status, resolved, replica, suite_id)
                repeat_rows = _suite_rows(
                    directory, status, resolved, replica, suite_id.replace("-primary", "-repeat")
                )
                for primary, repeat in zip(primary_rows, repeat_rows, strict=True):
                    if primary["world_seed"] != repeat["world_seed"]:
                        raise ValueError("Repeat seed mismatch")
                    if any(
                        primary.get(name) is None or repeat.get(name) is None
                        for name in DETERMINISTIC_COLUMNS
                    ):
                        raise ValueError("Missing deterministic evidence")
                    repeat_match = all(
                        primary.get(name) == repeat.get(name)
                        for name in DETERMINISTIC_COLUMNS
                    )
                    deterministic &= repeat_match
                    repeat_p95_ok &= repeat["decision_time_p95_ms"] < 50
                    repeat_max_ok &= repeat["decision_time_max_ms"] < 100
                    available = primary.get("initially_available_coins")
                    if not isinstance(available, int) or available <= 0:
                        raise ValueError("Missing available-coin count")
                    collection = primary["coins_collected"] / available
                    rows.append(
                        {
                            **primary,
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
                            "world_seed": primary["world_seed"],
                            "coins_collected": primary["coins_collected"],
                            "initially_available_coins": available,
                            "collection_fraction": collection,
                            "self_kills": primary["self_kills"],
                            "evaluation_decisions": primary["evaluation_decisions"],
                            "evaluation_unseen_decisions": primary["evaluation_unseen_decisions"],
                            "decision_time_p95_ms": primary["decision_time_p95_ms"],
                            "decision_time_max_ms": primary["decision_time_max_ms"],
                            "action_sequence_sha256": primary["executed_action_sequence_sha256"],
                            "repeat_match": repeat_match,
                            "repeat_decision_time_p95_ms": repeat["decision_time_p95_ms"],
                            "repeat_decision_time_max_ms": repeat["decision_time_max_ms"],
                            "repeat_action_sequence_sha256": repeat[
                                "executed_action_sequence_sha256"
                            ],
                        }
                    )

    summaries = _summarize(rows)
    comparisons = _comparisons(rows)
    criteria = _criteria(
        summaries, comparisons, diagnostics, deterministic, repeat_p95_ok, repeat_max_ok
    )
    result = {
        "schema_version": 1,
        "issue": 142,
        "deterministic": deterministic,
        "primary_evaluation_episodes": len(rows),
        "repeat_evaluation_episodes": len(rows),
        "summaries": summaries,
        "comparisons": comparisons,
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
    """Recompute the registered decision from committed compact evidence."""
    output = Path(output).resolve()
    with (output / "evidence.csv").open(encoding="utf-8", newline="") as handle:
        evidence = list(csv.DictReader(handle))
    with (output / "training_diagnostics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        diagnostics = list(csv.DictReader(handle))
    if len(evidence) != 1200 or len(diagnostics) != 10:
        raise ValueError("Committed evidence has an unexpected row count")
    numeric = [
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
        for row in evidence
    ]
    numeric_diagnostics = [
        {**row, "mean_visits_per_state": float(row["mean_visits_per_state"])}
        for row in diagnostics
    ]
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in numeric:
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
    comparisons = _comparisons(numeric)
    criteria = _criteria(
        summaries,
        comparisons,
        numeric_diagnostics,
        all(row["repeat_match"] for row in numeric),
        all(row["repeat_decision_time_p95_ms"] < 50 for row in numeric),
        all(row["repeat_decision_time_max_ms"] < 100 for row in numeric),
    )
    committed = _read_json(output / "result.json")
    if comparisons != committed["comparisons"] or criteria != committed["criteria"]:
        raise ValueError("Recomputed evidence does not match committed result.json")
    return {"comparisons": comparisons, "criteria": criteria, "passed": all(criteria.values())}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _paired_metric(
    rows: list[dict[str, Any]], scenario: str, metric: str, seed: int
) -> dict[str, Any]:
    values: dict[str, dict[str, dict[int, float]]] = defaultdict(dict)
    for row in rows:
        if row["scenario"] != scenario:
            continue
        model_values = values[row["treatment"]].setdefault(row["model"], {})
        world_seed = row["world_seed"]
        if world_seed in model_values:
            raise ValueError(f"Duplicate primary {scenario} episode")
        model_values[world_seed] = row[metric]
    return paired_bootstrap(
        values["candidate"], values["control"], resampler_seed=seed
    ).as_dict()


def _comparisons(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "classic_self_kill_candidate_minus_control": _paired_metric(
            rows, "classic", "self_kills", 142
        ),
        "classic_collection_candidate_minus_control": _paired_metric(
            rows, "classic", "collection_fraction", 143
        ),
    }


def _criteria(
    summaries: list[dict[str, Any]],
    comparisons: dict[str, Any],
    diagnostics: list[dict[str, Any]],
    deterministic: bool,
    repeat_p95_ok: bool,
    repeat_max_ok: bool,
) -> dict[str, bool]:
    aggregate = _aggregate_metrics(summaries)
    control_rates = _replica_rates(summaries, "control")
    candidate_rates = _replica_rates(summaries, "candidate")
    lower_replicas = sum(
        candidate_rates[model] < control_rates[model] for model in control_rates
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
    self_kill = comparisons["classic_self_kill_candidate_minus_control"]
    return {
        "classic_self_kill_mean_below_control": self_kill["mean_difference"] < 0,
        "classic_self_kill_ci_upper_below_zero": self_kill["ci_upper"] < 0,
        "at_least_four_replicas_lower_self_kill": lower_replicas >= 4,
        "classic_collection_decrease_within_0_05": (
            aggregate[("candidate", "classic")]["collection_fraction"]
            - aggregate[("control", "classic")]["collection_fraction"]
            >= -0.05
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


def _aggregate_metrics(
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


def _replica_rates(
    summaries: list[dict[str, Any]], treatment: str
) -> dict[str, float]:
    rates = {
        row["model"]: row["self_kill_rate"]
        for row in summaries
        if row["treatment"] == treatment and row["scenario"] == "classic"
    }
    if len(rates) != 5:
        raise ValueError("Expected five Classic replicas per treatment")
    return rates


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
