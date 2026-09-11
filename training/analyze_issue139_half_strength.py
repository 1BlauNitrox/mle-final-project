"""Analyze the preregistered Issue #139 half-strength experiment."""

from __future__ import annotations

import argparse
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
    / "2026-09-11-task2-half-strength-escape-distance-"
    "DerKleineSprengstoffkapitalist"
)
PLAN_IDS = {
    "control": "issue139-compact-task2-full-strength",
    "candidate": "issue139-compact-task2-half-strength",
}
SUITES = {
    "classic-primary": "classic",
    "coin-heaven-primary": "coin-heaven",
    "loot-crate-primary": "loot-crate",
}


def analyze(plan_root: Path = PLAN_ROOT, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Require completed plans and evaluate all registered gates."""
    plan_root = Path(plan_root).resolve()
    output = Path(output).resolve()
    rows: list[dict[str, Any]] = []
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
                primary_rows = _suite_rows(directory, status, resolved, replica, suite_id)
                repeat_rows = _suite_rows(
                    directory,
                    status,
                    resolved,
                    replica,
                    suite_id.replace("-primary", "-repeat"),
                )
                for primary, repeat in zip(primary_rows, repeat_rows, strict=True):
                    if primary["world_seed"] != repeat["world_seed"]:
                        raise ValueError("Repeat seed mismatch")
                    if any(
                        primary.get(name) is None or repeat.get(name) is None
                        for name in DETERMINISTIC_COLUMNS
                    ):
                        raise ValueError("Missing deterministic evidence")
                    deterministic &= all(
                        primary.get(name) == repeat.get(name)
                        for name in DETERMINISTIC_COLUMNS
                    )
                    repeat_p95_ok &= repeat["decision_time_p95_ms"] < 50
                    repeat_max_ok &= repeat["decision_time_max_ms"] < 100
                    available = primary.get("initially_available_coins")
                    if not isinstance(available, int) or available <= 0:
                        raise ValueError("Missing available-coin count")
                    rows.append(
                        {
                            **primary,
                            "treatment": treatment,
                            "model": replica,
                            "scenario": scenario,
                            "collection_fraction": primary["coins_collected"] / available,
                        }
                    )

    summaries = _summarize(rows)
    comparison = _classic_collection_comparison(rows)
    criteria = _criteria(
        summaries,
        comparison,
        diagnostics,
        deterministic,
        repeat_p95_ok,
        repeat_max_ok,
    )
    result = {
        "schema_version": 1,
        "issue": 139,
        "deterministic": deterministic,
        "primary_evaluation_episodes": len(rows),
        "repeat_evaluation_episodes": len(rows),
        "summaries": summaries,
        "comparisons": {
            "classic_collection_candidate_minus_control": comparison,
        },
        "criteria": criteria,
        "passed": all(criteria.values()),
        "training_diagnostics": diagnostics,
        "evaluation_unseen_state_rates": {
            treatment: _evaluation_unseen_rate(rows, treatment)
            for treatment in PLAN_IDS
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_summary(output / "summary.csv", summaries)
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


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
        values["candidate"],
        values["control"],
        resampler_seed=139,
    ).as_dict()


def _criteria(
    summaries: list[dict[str, Any]],
    comparison: dict[str, Any],
    diagnostics: list[dict[str, Any]],
    deterministic: bool,
    repeat_p95_ok: bool,
    repeat_max_ok: bool,
) -> dict[str, bool]:
    """Evaluate every preregistered Issue #139 criterion."""
    aggregate = _aggregate_metrics(summaries)
    candidate_replica_rates = [
        row["self_kill_rate"]
        for row in summaries
        if row["treatment"] == "candidate" and row["scenario"] == "classic"
    ]
    if len(candidate_replica_rates) != 5:
        raise ValueError("Expected five candidate Classic replicas")
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
    primary_p95_ok = all(row["decision_time_p95_ms"] < 50 for row in summaries)
    primary_max_ok = all(row["decision_time_max_ms"] < 100 for row in summaries)

    return {
        "classic_collection_mean_above_control": comparison["mean_difference"] > 0,
        "classic_collection_ci_lower_above_zero": comparison["ci_lower"] > 0,
        "candidate_classic_self_kill_below_0_055": (
            aggregate[("candidate", "classic")]["self_kill_rate"] < 0.055
        ),
        "at_least_four_candidate_replicas_below_0_055": (
            sum(rate < 0.055 for rate in candidate_replica_rates) >= 4
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
        "decision_time_p95_within_limit": primary_p95_ok and repeat_p95_ok,
        "decision_time_max_within_limit": primary_max_ok and repeat_max_ok,
    }


def _aggregate_metrics(summaries: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, float]]:
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


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-root", type=Path, default=PLAN_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    try:
        result = analyze(arguments.plan_root, arguments.output)
    except Exception as error:
        print(f"Analysis failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result["criteria"], indent=2, sort_keys=True))
    print(f"Overall pass: {result['passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
