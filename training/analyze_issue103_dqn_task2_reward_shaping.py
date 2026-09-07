"""Validate and summarize the preregistered Issue #103 run-plan outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean, median
from typing import Any

from training.aggregate import _parse_csv_row, read_episodes_csv
from training.paired_bootstrap import paired_bootstrap
from training.run_experiment import REPOSITORY_ROOT

PLAN_ROOT = REPOSITORY_ROOT / "training_outputs" / "run-plans"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "training_outputs" / "issue103-analysis"
PLAN_IDS = {
    "control": "issue103-dqn-task2-reward-control",
    "survival_rebalance": "issue103-dqn-task2-reward-survival-rebalance",
    "safety_bomb": "issue103-dqn-task2-reward-safety-bomb",
}
TREATMENTS = ("survival_rebalance", "safety_bomb")
PRIMARY_SUITES = {
    "classic-primary": "classic",
    "coin-heaven-primary": "coin-heaven",
    "loot-crate-primary": "loot-crate",
}
NON_REGRESSION_MARGIN = -0.05


def analyze(plan_root: Path = PLAN_ROOT, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Require completed paired plans and evaluate the registered decision rule."""
    plan_root = Path(plan_root).resolve()
    rows: list[dict[str, Any]] = []
    deterministic_repeats = True

    for treatment, plan_id in PLAN_IDS.items():
        directory = plan_root / plan_id
        status = _read_json(directory / "status.json")
        resolved = _read_json(directory / "resolved_plan.json")
        if status.get("status") != "completed":
            raise ValueError(f"Run plan is not completed: {plan_id}")
        if resolved.get("reward_variant") != treatment:
            raise ValueError(f"Unexpected reward_variant for {plan_id}")

        for replica in (item["replica_id"] for item in resolved["replicas"]):
            for suite_id, scenario in PRIMARY_SUITES.items():
                primary = _suite_rows(directory, status, resolved, replica, suite_id)
                repeat = _suite_rows(
                    directory,
                    status,
                    resolved,
                    replica,
                    suite_id.replace("-primary", "-repeat"),
                )
                for primary_row, repeat_row in zip(primary, repeat, strict=True):
                    if primary_row["world_seed"] != repeat_row["world_seed"]:
                        raise ValueError("Primary and repeat world seeds differ")
                    if primary_row.get("executed_action_sequence_sha256") != repeat_row.get(
                        "executed_action_sequence_sha256"
                    ):
                        deterministic_repeats = False
                    available = primary_row.get("initially_available_coins")
                    if not isinstance(available, int) or available <= 0:
                        raise ValueError("Evaluation row has no available-coin count")
                    rows.append(
                        {
                            **primary_row,
                            "treatment": treatment,
                            "scenario": scenario,
                            "replica": replica,
                            "collection_fraction": primary_row["coins_collected"] / available,
                        }
                    )

    return _compute_and_write_result(rows, deterministic_repeats, output)


def rows_from_evidence(evidence_root: Path) -> tuple[list[dict[str, Any]], bool]:
    """Rebuild the same analysis-ready rows from committed evidence CSVs.

    Reads only `evidence_root/<plan_id>/evaluation-episodes.csv` (written by
    `training/export_evidence.py`), which already carries every column
    `_suite_rows` would have read from the raw job tree plus the identifying
    columns (`treatment` here comes from the plan's own `reward_variant`, not
    re-derived). This is what makes the committed result independently
    reproducible without the ~4.4 GiB raw output tree or an external archive:
    a reviewer with only this repository checkout can run this against the
    committed evidence and get byte-identical numbers.
    """
    evidence_root = Path(evidence_root).resolve()
    rows: list[dict[str, Any]] = []
    deterministic_repeats = True

    for treatment, plan_id in PLAN_IDS.items():
        manifest = _read_json(evidence_root / plan_id / "manifest.json")
        if manifest.get("reward_variant") != treatment:
            raise ValueError(f"Unexpected reward_variant for {plan_id}")
        by_key: dict[tuple[str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
        with (evidence_root / plan_id / "evaluation-episodes.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            for row_number, raw_row in enumerate(csv.DictReader(handle), start=2):
                row = _parse_csv_row(raw_row, row_number)
                row["world_seed"] = int(raw_row["world_seed"])
                row["agent_seed"] = int(raw_row["agent_seed"])
                by_key[(row["replica"], row["stage_or_suite"])][row["world_seed"]] = row

        replicas = {
            row["replica"] for rows_by_seed in by_key.values() for row in rows_by_seed.values()
        }
        for replica in replicas:
            for suite_id, scenario in PRIMARY_SUITES.items():
                primary = by_key[(replica, suite_id)]
                repeat = by_key[(replica, suite_id.replace("-primary", "-repeat"))]
                for world_seed in sorted(primary):
                    primary_row, repeat_row = primary[world_seed], repeat[world_seed]
                    if primary_row.get("executed_action_sequence_sha256") != repeat_row.get(
                        "executed_action_sequence_sha256"
                    ):
                        deterministic_repeats = False
                    available = primary_row.get("initially_available_coins")
                    if not isinstance(available, int) or available <= 0:
                        raise ValueError("Evaluation row has no available-coin count")
                    rows.append(
                        {
                            **primary_row,
                            "treatment": treatment,
                            "scenario": scenario,
                            "collection_fraction": primary_row["coins_collected"] / available,
                        }
                    )
    return rows, deterministic_repeats


def verify_from_evidence(
    evidence_root: Path, expected_result_path: Path
) -> tuple[dict[str, Any], bool]:
    """Recompute the result from evidence and compare it to the committed one."""
    rows, deterministic_repeats = rows_from_evidence(evidence_root)
    recomputed = _compute_and_write_result(rows, deterministic_repeats, output=None)
    expected = _read_json(Path(expected_result_path))
    return recomputed, recomputed == expected


def _compute_and_write_result(
    rows: list[dict[str, Any]],
    deterministic_repeats: bool,
    output: Path | None,
) -> dict[str, Any]:
    summaries = _summaries(rows)
    comparisons = {
        treatment: {
            "paired_collection_fraction_minus_control": _paired_comparisons(
                rows, "collection_fraction", treatment
            ),
            "paired_survival_rate_minus_control": _paired_comparisons(
                rows, "survived", treatment
            ),
            "paired_self_kill_rate_minus_control": _paired_comparisons(
                rows, "self_kills", treatment
            ),
        }
        for treatment in TREATMENTS
    }
    criteria = {
        treatment: _criteria(
            rows,
            comparisons[treatment]["paired_collection_fraction_minus_control"],
            comparisons[treatment]["paired_survival_rate_minus_control"],
            deterministic_repeats,
        )
        for treatment in TREATMENTS
    }
    result = {
        "schema_version": 1,
        "issue": 103,
        "deterministic_repeats": deterministic_repeats,
        "primary_evaluation_episodes": len(rows),
        "comparisons": comparisons,
        "criteria": criteria,
        "decision": {
            treatment: (
                f"adopt_{treatment}"
                if all(criteria[treatment].values())
                else f"reject_{treatment}_for_this_training_configuration"
            )
            for treatment in TREATMENTS
        },
    }
    if output is not None:
        output = Path(output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        _write_summary(output / "summary.csv", summaries)
        (output / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return result


def _suite_rows(
    directory: Path,
    status: dict[str, Any],
    resolved: dict[str, Any],
    replica: str,
    suite: str,
) -> list[dict[str, Any]]:
    expected = {
        job["run_id"]: job
        for job in resolved["jobs"]
        if job["kind"] == "evaluation"
        and job["replica"] == replica
        and job["stage_or_suite"] == suite
    }
    actual = {
        run_id: job
        for run_id, job in status["jobs"].items()
        if run_id in expected
    }
    if set(actual) != set(expected):
        raise ValueError(f"Evaluation matrix mismatch for {replica}/{suite}")

    rows: list[dict[str, Any]] = []
    for run_id in sorted(actual):
        job = actual[run_id]
        attempts = job.get("attempts", [])
        if job.get("status") != "completed" or not attempts:
            raise ValueError(f"Evaluation job is not complete: {run_id}")
        run_directory = directory / attempts[-1]["output"]
        episode_rows = read_episodes_csv(run_directory / "episodes.csv")
        if len(episode_rows) != 1:
            raise ValueError(f"Expected exactly one episode row: {run_id}")
        metadata = _read_json(run_directory / "metadata.json")
        registered = expected[run_id]
        for field in ("world_seed", "agent_seed", "scenario", "rounds", "opponents"):
            if metadata.get(field) != registered[field]:
                raise ValueError(f"Metadata mismatch for {run_id}: {field}")
        if metadata.get("mode") != "evaluation":
            raise ValueError(f"Job is not evaluation mode: {run_id}")
        row = dict(episode_rows[0])
        row["world_seed"] = metadata["world_seed"]
        rows.append(row)
    return rows


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["scenario"], row["treatment"])].append(row)
    result: list[dict[str, Any]] = []
    for scenario in PRIMARY_SUITES.values():
        for treatment in PLAN_IDS:
            group = groups[(scenario, treatment)]
            episodes = len(group)
            attempted = sum(row["attempted_actions"] for row in group)
            coins = sum(row["coins_collected"] for row in group)
            survival_steps = sum(row["survival_steps"] for row in group)
            decision_medians = [
                row["decision_time_median_ms"]
                for row in group
                if row["decision_time_median_ms"] is not None
            ]
            decision_p95 = [
                row["decision_time_p95_ms"]
                for row in group
                if row["decision_time_p95_ms"] is not None
            ]
            decision_max = [
                row["decision_time_max_ms"]
                for row in group
                if row["decision_time_max_ms"] is not None
            ]
            result.append(
                {
                    "scenario": scenario,
                    "treatment": treatment,
                    "episodes": episodes,
                    "mean_collection_fraction": fmean(row["collection_fraction"] for row in group),
                    "survival_rate": sum(row["survived"] for row in group) / episodes,
                    "self_kill_rate": sum(row["self_kills"] for row in group) / episodes,
                    "invalid_action_rate": sum(row["invalid_actions"] for row in group) / attempted,
                    "mean_coins": coins / episodes,
                    "mean_crates_destroyed": fmean(row["crates_destroyed"] for row in group),
                    "mean_survival_steps": survival_steps / episodes,
                    "steps_per_coin": survival_steps / coins if coins else None,
                    "action_up": sum(row["action_up"] for row in group),
                    "action_right": sum(row["action_right"] for row in group),
                    "action_down": sum(row["action_down"] for row in group),
                    "action_left": sum(row["action_left"] for row in group),
                    "action_wait": sum(row["action_wait"] for row in group),
                    "action_bomb": sum(row["action_bomb"] for row in group),
                    "action_unknown": sum(row["action_unknown"] for row in group),
                    "decision_time_median_ms": median(decision_medians)
                    if decision_medians
                    else None,
                    "decision_time_p95_ms": max(decision_p95) if decision_p95 else None,
                    "decision_time_max_ms": max(decision_max) if decision_max else None,
                }
            )
    return result


def _paired_comparisons(
    rows: list[dict[str, Any]], metric: str, treatment: str
) -> dict[str, dict[str, float | int]]:
    comparisons: dict[str, dict[str, float | int]] = {}
    for scenario in PRIMARY_SUITES.values():
        arms: dict[str, dict[str, dict[int, float]]] = {
            "control": {},
            treatment: {},
        }
        for row in rows:
            if row["scenario"] != scenario or row["treatment"] not in arms:
                continue
            arms[row["treatment"]].setdefault(row["replica"], {})[row["world_seed"]] = float(
                row[metric]
            )
        comparisons[scenario] = paired_bootstrap(
            arms[treatment], arms["control"], resampler_seed=103
        ).as_dict()
    return comparisons


def _criteria(
    rows: list[dict[str, Any]],
    collection: dict[str, dict[str, float | int]],
    survival: dict[str, dict[str, float | int]],
    deterministic_repeats: bool,
) -> dict[str, bool]:
    timing = [
        row
        for row in rows
        if row.get("decision_time_p95_ms") is not None
        and row.get("decision_time_max_ms") is not None
    ]
    return {
        "deterministic_repeats": deterministic_repeats,
        "collection_non_regression": all(
            comparison["ci_lower"] >= NON_REGRESSION_MARGIN
            for comparison in collection.values()
        ),
        "classic_survival_improved": survival["classic"]["ci_lower"] >= 0,
        "timing_within_limits": bool(timing)
        and all(
            row["decision_time_p95_ms"] < 50 and row["decision_time_max_ms"] < 100
            for row in timing
        ),
    }


def _write_summary(path: Path, summaries: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-root", type=Path, default=PLAN_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--verify-from-evidence",
        type=Path,
        default=None,
        metavar="EVIDENCE_ROOT",
        help=(
            "Instead of reading the raw job tree under --plan-root, recompute "
            "the result from committed evidence CSVs under EVIDENCE_ROOT "
            "(see training/export_evidence.py) and compare it to --output's "
            "committed result.json. Exits non-zero on any mismatch."
        ),
    )
    args = parser.parse_args()
    if args.verify_from_evidence is not None:
        recomputed, matches = verify_from_evidence(
            args.verify_from_evidence, args.output / "result.json"
        )
        print(json.dumps(recomputed, indent=2, sort_keys=True))
        print("MATCHES committed result.json" if matches else "MISMATCH vs committed result.json")
        return 0 if matches else 1
    print(json.dumps(analyze(args.plan_root, args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
