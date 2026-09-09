"""Analyze the preregistered Issue #114 tabular Task 2 experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

from training.aggregate import read_episodes_csv
from training.paired_bootstrap import paired_bootstrap
from training.run_experiment import REPOSITORY_ROOT

PLAN_ROOT = REPOSITORY_ROOT / "training_outputs" / "run-plans"
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "experiments"
    / "2026-09-07-task2-useful-bomb-reward-DerKleineSprengstoffkapitalist"
)
PLANS = {
    "control": "issue114-tabular-task2-control",
    "candidate": "issue114-tabular-task2-useful-bomb-reward",
}
DIAGNOSTIC_PLANS = {
    "control": "issue114-tabular-task2-control-diagnostics-v2",
    "candidate": "issue114-tabular-task2-useful-bomb-reward-diagnostics-v2",
}
TASK1_PLAN = "issue114-tabular-task1-frozen"
SCENARIOS = ("classic", "coin-heaven", "loot-crate")


def analyze(plan_root: Path = PLAN_ROOT, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    plan_root, output = Path(plan_root).resolve(), Path(output).resolve()
    rows: list[dict[str, Any]] = []
    deterministic = True

    for arm, plan_id in PLANS.items():
        directory, status, resolved = _completed(plan_root, plan_id)
        expected_reward = 1.0 if arm == "candidate" else 0.0
        if resolved["useful_bomb_reward"] != expected_reward:
            raise ValueError(f"Unexpected treatment in {plan_id}")
        for replica in (item["replica_id"] for item in resolved["replicas"]):
            for scenario in SCENARIOS:
                primary = _suite_rows(directory, status, resolved, replica, f"{scenario}-primary")
                repeat = _suite_rows(directory, status, resolved, replica, f"{scenario}-repeat")
                for first, second in zip(primary, repeat, strict=True):
                    if first["world_seed"] != second["world_seed"]:
                        raise ValueError("Primary/repeat seed mismatch")
                    if (
                        first["executed_action_sequence_sha256"]
                        != second["executed_action_sequence_sha256"]
                    ):
                        deterministic = False
                    available = int(first["initially_available_coins"])
                    rows.append(
                        {
                            **first,
                            "arm": arm,
                            "replica": replica,
                            "scenario": scenario,
                            "pass": "primary",
                            "collection_fraction": first["coins_collected"] / available,
                        }
                    )

    task1_directory, task1_status, task1_resolved = _completed(plan_root, TASK1_PLAN)
    task1_replica = task1_resolved["replicas"][0]["replica_id"]
    task1_primary = _suite_rows(
        task1_directory, task1_status, task1_resolved, task1_replica, "coin-heaven-primary"
    )
    task1_repeat = _suite_rows(
        task1_directory, task1_status, task1_resolved, task1_replica, "coin-heaven-repeat"
    )
    task1_values: dict[int, float] = {}
    for first, second in zip(task1_primary, task1_repeat, strict=True):
        deterministic &= (
            first["executed_action_sequence_sha256"] == second["executed_action_sequence_sha256"]
        )
        task1_values[first["world_seed"]] = (
            first["coins_collected"] / first["initially_available_coins"]
        )

    comparisons = {
        scenario: _compare(rows, scenario, "collection_fraction") for scenario in SCENARIOS
    }
    classic_crates = _compare(rows, "classic", "crates_destroyed")
    candidate_coin = _nested(rows, "candidate", "coin-heaven", "collection_fraction")
    task1_nested = {replica: dict(task1_values) for replica in candidate_coin}
    retention = paired_bootstrap(candidate_coin, task1_nested, resampler_seed=115).as_dict()

    model_hash_match: dict[str, dict[str, bool]] = {}
    training_rows: list[dict[str, Any]] = []
    training_summary: list[dict[str, Any]] = []
    for arm, diagnostic_id in DIAGNOSTIC_PLANS.items():
        diag_dir, diag_status, diag_resolved = _completed(plan_root, diagnostic_id)
        original_dir = plan_root / PLANS[arm]
        model_hash_match[arm] = {}
        grouped: dict[str, list[float]] = defaultdict(list)
        for replica in (item["replica_id"] for item in diag_resolved["replicas"]):
            model_hash_match[arm][replica] = _sha(
                diag_dir / "replicas" / replica / "agent" / "model.npz"
            ) == _sha(original_dir / "replicas" / replica / "agent" / "model.npz")
            for job in diag_resolved["jobs"]:
                if job["replica"] != replica:
                    continue
                attempt = diag_status["jobs"][job["run_id"]]["attempts"][-1]
                episode_rows = read_episodes_csv(diag_dir / attempt["output"] / "episodes.csv")
                for row in episode_rows:
                    value = row.get("useful_bombs")
                    if value is None:
                        raise ValueError(f"Missing useful_bombs in {job['run_id']}")
                    grouped[job["stage_or_suite"]].append(float(value))
                    training_rows.append(
                        {
                            "arm": arm,
                            "replica": replica,
                            "stage": job["stage_or_suite"],
                            "round": row["round"],
                            "useful_bombs": value,
                        }
                    )
        for stage, values in sorted(grouped.items()):
            training_summary.append(
                {
                    "arm": arm,
                    "stage": stage,
                    "episodes": len(values),
                    "mean_useful_bombs": fmean(values),
                    "total_useful_bombs": sum(values),
                }
            )
        all_values = [float(row["useful_bombs"]) for row in training_rows if row["arm"] == arm]
        training_summary.append(
            {
                "arm": arm,
                "stage": "all",
                "episodes": len(all_values),
                "mean_useful_bombs": fmean(all_values),
                "total_useful_bombs": sum(all_values),
            }
        )

    summaries = _summaries(rows)
    candidate_classic = _nested(rows, "candidate", "classic", "collection_fraction")
    control_classic = _nested(rows, "control", "classic", "collection_fraction")
    improving = sum(
        fmean(candidate_classic[r].values()) > fmean(control_classic[r].values())
        for r in candidate_classic
    )
    classic = comparisons["classic"]
    control_self = next(
        x["self_kill_rate"]
        for x in summaries
        if x["arm"] == "control" and x["scenario"] == "classic"
    )
    candidate_self = next(
        x["self_kill_rate"]
        for x in summaries
        if x["arm"] == "candidate" and x["scenario"] == "classic"
    )
    useful_means = {
        x["arm"]: x["mean_useful_bombs"] for x in training_summary if x["stage"] == "all"
    }
    timing_ok = all(
        float(row["decision_time_p95_ms"]) < 50 and float(row["decision_time_max_ms"]) < 100
        for row in rows
    )
    hashes_ok = all(all(values.values()) for values in model_hash_match.values())
    criteria = {
        "classic_mean_above_zero": classic["mean_difference"] > 0,
        "classic_ci_lower_above_zero": classic["ci_lower"] > 0,
        "at_least_four_replicas_improve": improving >= 4,
        "useful_bombs_increase": useful_means["candidate"] > useful_means["control"],
        "classic_crates_destroyed_increase": classic_crates["mean_difference"] > 0,
        "classic_self_kill_within_margin": candidate_self <= control_self + 0.02,
        "task1_retention_ci_lower_above_margin": retention["ci_lower"] > -0.05,
        "deterministic_repeats": deterministic,
        "timing_within_limits": timing_ok,
        "diagnostic_replay_models_match": hashes_ok,
    }
    result = {
        "schema_version": 1,
        "issue": 114,
        "primary_evaluation_episodes": len(rows),
        "comparisons": comparisons,
        "classic_crates_destroyed_candidate_minus_control": classic_crates,
        "task1_retention_candidate_minus_frozen": retention,
        "replicas_improving_classic": improving,
        "classic_self_kill_rates": {"control": control_self, "candidate": candidate_self},
        "mean_useful_bombs_per_training_episode": useful_means,
        "diagnostic_replay_model_hash_match": model_hash_match,
        "criteria": criteria,
        "decision": "adopt_useful_bomb_reward"
        if all(criteria.values())
        else "reject_useful_bomb_reward_for_this_configuration",
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "evidence.csv", rows)
    _write_csv(output / "training-summary.csv", training_summary)
    _write_csv(output / "summary.csv", summaries)
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _completed(root: Path, plan_id: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    directory = root / plan_id
    status, resolved = _json(directory / "status.json"), _json(directory / "resolved_plan.json")
    if status["status"] != "completed" or any(
        job["status"] != "completed" for job in status["jobs"].values()
    ):
        raise ValueError(f"Incomplete plan: {plan_id}")
    return directory, status, resolved


def _suite_rows(
    directory: Path, status: dict[str, Any], resolved: dict[str, Any], replica: str, suite: str
) -> list[dict[str, Any]]:
    jobs = [
        job
        for job in resolved["jobs"]
        if job["kind"] == "evaluation"
        and job["replica"] == replica
        and job["stage_or_suite"] == suite
    ]
    result = []
    for job in sorted(jobs, key=lambda item: item["world_seed"]):
        recorded = status["jobs"][job["run_id"]]
        rows = read_episodes_csv(directory / recorded["attempts"][-1]["output"] / "episodes.csv")
        if len(rows) != 1:
            raise ValueError(f"Expected one episode: {job['run_id']}")
        result.append({**rows[0], "world_seed": job["world_seed"], "agent_seed": job["agent_seed"]})
    if not result:
        raise ValueError(f"Missing suite {replica}/{suite}")
    return result


def _nested(
    rows: list[dict[str, Any]], arm: str, scenario: str, metric: str
) -> dict[str, dict[int, float]]:
    result: dict[str, dict[int, float]] = {}
    for row in rows:
        if row["arm"] == arm and row["scenario"] == scenario:
            result.setdefault(row["replica"], {})[row["world_seed"]] = float(row[metric])
    return result


def _compare(rows: list[dict[str, Any]], scenario: str, metric: str) -> dict[str, Any]:
    return paired_bootstrap(
        _nested(rows, "candidate", scenario, metric),
        _nested(rows, "control", scenario, metric),
        resampler_seed=114,
    ).as_dict()


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for arm in PLANS:
        for scenario in SCENARIOS:
            group = [row for row in rows if row["arm"] == arm and row["scenario"] == scenario]
            attempted = sum(row["attempted_actions"] for row in group)
            result.append(
                {
                    "arm": arm,
                    "scenario": scenario,
                    "episodes": len(group),
                    "mean_collection_fraction": fmean(row["collection_fraction"] for row in group),
                    "mean_crates_destroyed": fmean(row["crates_destroyed"] for row in group),
                    "self_kill_rate": sum(row["self_kills"] for row in group) / len(group),
                    "survival_rate": sum(row["survived"] for row in group) / len(group),
                    "invalid_action_rate": sum(row["invalid_actions"] for row in group) / attempted,
                    "decision_time_p95_ms_max": max(row["decision_time_p95_ms"] for row in group),
                    "decision_time_max_ms_max": max(row["decision_time_max_ms"] for row in group),
                }
            )
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-root", type=Path, default=PLAN_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(analyze(args.plan_root, args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
