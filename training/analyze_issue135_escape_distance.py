"""Analyze the preregistered Issue #135 compact-state experiment."""

from __future__ import annotations

import argparse
import csv
import json
import sys
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
    / "2026-09-11-task2-escape-distance-potential-DerKleineSprengstoffkapitalist"
)

PLAN_IDS = {
    "control": "issue135-compact-task2-control",
    "candidate": "issue135-compact-task2-escape-distance",
}

TREATMENT_SUITES = {
    "control": {
        "classic-primary": "classic",
        "coin-heaven-primary": "coin-heaven",
        "loot-crate-primary": "loot-crate",
    },
    "candidate": {
        "classic-primary": "classic",
        "coin-heaven-primary": "coin-heaven",
        "loot-crate-primary": "loot-crate",
    },
}
DETERMINISTIC_COLUMNS = (
    "executed_action_sequence_sha256",
    "episode_steps",
    "survival_steps",
    "score",
    "coins_collected",
    "initially_available_coins",
    "coins_found",
    "crates_destroyed",
    "bombs_dropped",
    "self_kills",
    "invalid_actions",
    "attempted_actions",
    "survived",
    "termination_reason",
    "action_up",
    "action_right",
    "action_down",
    "action_left",
    "action_wait",
    "action_bomb",
    "action_unknown",
    "evaluation_decisions",
    "evaluation_unseen_decisions",
    "evaluation_unseen_state_rate",
)


def analyze(plan_root: Path = PLAN_ROOT, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Require completed deterministic plans and evaluate registered gates."""
    plan_root = Path(plan_root).resolve()
    output = Path(output).resolve()
    rows: list[dict[str, Any]] = []
    training_diagnostics: list[dict[str, Any]] = []
    deterministic = True
    repeat_p95_ok = True
    repeat_max_ok = True

    for treatment, plan_id in PLAN_IDS.items():
        plan_directory = plan_root / plan_id
        status = _read_json(plan_directory / "status.json")
        resolved = _read_json(plan_directory / "resolved_plan.json")
        if status.get("status") != "completed":
            raise ValueError(f"Run plan is not completed: {plan_id}")
        replicas = [item["replica_id"] for item in resolved["replicas"]]
        for replica in replicas:
            model = replica
            training_diagnostics.append(
                _training_diagnostic(
                    plan_directory,
                    status,
                    replica,
                    treatment,
                )
            )
            for suite_id, scenario in TREATMENT_SUITES[treatment].items():
                primary_rows = _suite_rows(
                    plan_directory,
                    status,
                    resolved,
                    replica,
                    suite_id,
                )
                repeat_rows = _suite_rows(
                    plan_directory,
                    status,
                    resolved,
                    replica,
                    suite_id.replace("-primary", "-repeat"),
                )
                for primary, repeat in zip(primary_rows, repeat_rows, strict=True):
                    if primary["world_seed"] != repeat["world_seed"]:
                        raise ValueError(
                            f"Repeat seed mismatch for {treatment}/{replica}/{scenario}"
                        )
                    if any(
                        primary.get(name) is None or repeat.get(name) is None
                        for name in DETERMINISTIC_COLUMNS
                    ):
                        raise ValueError("Missing deterministic evidence in evaluation repeat")
                    if any(primary.get(name) != repeat.get(name) for name in DETERMINISTIC_COLUMNS):
                        deterministic = False
                    repeat_p95_ok &= (
                        repeat.get("decision_time_p95_ms") is not None
                        and repeat["decision_time_p95_ms"] < 50
                    )
                    repeat_max_ok &= (
                        repeat.get("decision_time_max_ms") is not None
                        and repeat["decision_time_max_ms"] < 100
                    )
                    if not isinstance(
                        primary.get("executed_action_sequence_sha256"), str
                    ) or not primary["executed_action_sequence_sha256"]:
                        raise ValueError("Missing action digest in primary evaluation")
                    if not isinstance(
                        repeat.get("executed_action_sequence_sha256"), str
                    ) or not repeat["executed_action_sequence_sha256"]:
                        raise ValueError("Missing action digest in repeat evaluation")
                    available = primary.get("initially_available_coins")
                    if not isinstance(available, int) or available <= 0:
                        raise ValueError(
                            f"Missing available-coin count for {treatment}/{replica}/{scenario}"
                        )
                    rows.append(
                        {
                            **primary,
                            "treatment": treatment,
                            "model": model,
                            "scenario": scenario,
                            "collection_fraction": primary["coins_collected"] / available,
                        }
                    )

    summaries = _summarize(rows)
    comparisons = _comparisons(rows)
    criteria = _criteria(
        rows,
        summaries,
        comparisons,
        training_diagnostics,
        deterministic,
        repeat_p95_ok,
        repeat_max_ok,
    )
    result = {
        "schema_version": 1,
        "issue": 135,
        "deterministic": deterministic,
        "primary_evaluation_episodes": len(rows),
        "repeat_evaluation_episodes": len(rows),
        "summaries": summaries,
        "comparisons": comparisons,
        "criteria": criteria,
        "passed": all(criteria.values()),
        "training_diagnostics": training_diagnostics,
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


def _training_diagnostic(
    plan_directory: Path,
    status: dict[str, Any],
    replica: str,
    treatment: str,
) -> dict[str, Any]:
    """Read final-checkpoint learning-efficiency diagnostics."""

    job_id = f"train-{replica}-classic-crates"
    job = status["jobs"].get(job_id)

    if not isinstance(job, dict):
        raise ValueError(f"Missing final training job: {job_id}")

    attempts = job.get("attempts", [])

    if job.get("status") != "completed" or not attempts:
        raise ValueError(f"Final training job is incomplete: {job_id}")

    run_directory = plan_directory / attempts[-1]["output"]
    episode_rows = read_episodes_csv(
        run_directory / "episodes.csv"
    )

    if not episode_rows:
        raise ValueError(
            f"Final training job contains no episodes: {job_id}"
        )

    final_row = max(
        episode_rows,
        key=lambda row: row["round"],
    )

    required_metrics = (
        "q_table_size",
        "total_state_visits",
        "mean_visits_per_state",
        "singleton_state_fraction",
    )

    if any(final_row.get(name) is None for name in required_metrics):
        raise ValueError(
            f"Final training diagnostics are incomplete: {job_id}"
        )

    artifact_metadata = job.get("artifact")

    if not isinstance(artifact_metadata, dict):
        raise ValueError(f"Missing final artifact metadata: {job_id}")

    artifact_path = plan_directory / artifact_metadata["path"]

    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)

    return {
        "treatment": treatment,
        "model": replica,
        "q_table_size": final_row["q_table_size"],
        "total_state_visits": final_row["total_state_visits"],
        "mean_visits_per_state": final_row[
            "mean_visits_per_state"
        ],
        "singleton_state_fraction": final_row[
            "singleton_state_fraction"
        ],
        "q_table_artifact_size_bytes": artifact_path.stat().st_size,
    }


def _suite_rows(
    plan_directory: Path,
    status: dict[str, Any],
    resolved: dict[str, Any],
    replica: str,
    suite_id: str,
) -> list[dict[str, Any]]:
    prefix = f"eval-{replica}-{suite_id}-seed-"
    keys = sorted(key for key in status["jobs"] if key.startswith(prefix))
    expected = {
        job["run_id"]: job
        for job in resolved["jobs"]
        if (
            job["kind"] == "evaluation"
            and job["replica"] == replica
            and job["stage_or_suite"] == suite_id
        )
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
        episode_rows = read_episodes_csv(run_directory / "episodes.csv")
        if len(episode_rows) != 1:
            raise ValueError(f"Expected one episode row for {key}")
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
        row = dict(episode_rows[0])
        row["world_seed"] = metadata["world_seed"]
        result.append(row)
    return result


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["treatment"], row["model"], row["scenario"])].append(row)
    summaries: list[dict[str, Any]] = []
    for (treatment, model, scenario), group in sorted(groups.items()):
        attempted = sum(row["attempted_actions"] for row in group)
        invalid = sum(row["invalid_actions"] for row in group)
        decision_p95 = [
            row["decision_time_p95_ms"] for row in group if row["decision_time_p95_ms"] is not None
        ]
        decision_max = [
            row["decision_time_max_ms"] for row in group if row["decision_time_max_ms"] is not None
        ]
        coins = sum(row["coins_collected"] for row in group)
        survival_steps = sum(row["survival_steps"] for row in group)
        bombs = sum(row["bombs_dropped"] for row in group)
        action_counts = {
            action: sum(row[action] for row in group)
            for action in (
                "action_up",
                "action_right",
                "action_down",
                "action_left",
                "action_wait",
                "action_bomb",
                "action_unknown",
            )
        }
        summaries.append(
            {
                "treatment": treatment,
                "model": model,
                "scenario": scenario,
                "episodes": len(group),
                "mean_collection_fraction": fmean(row["collection_fraction"] for row in group),
                "mean_coins": coins / len(group),
                "full_clear_rate": sum(
                    row["coins_collected"] == row["initially_available_coins"] for row in group
                )
                / len(group),
                "zero_coin_rate": sum(row["coins_collected"] == 0 for row in group) / len(group),
                "survival_rate": sum(row["survived"] for row in group) / len(group),
                "mean_survival_steps": survival_steps / len(group),
                "mean_episode_steps": fmean(row["episode_steps"] for row in group),
                "steps_per_coin": survival_steps / coins if coins else None,
                "coins_per_100_survival_steps": (
                    100.0 * coins / survival_steps if survival_steps else None
                ),
                "coins_found": sum(row["coins_found"] for row in group),
                "crates_destroyed": sum(row["crates_destroyed"] for row in group),
                "bombs_dropped": bombs,
                "crates_per_bomb": (
                    sum(row["crates_destroyed"] for row in group) / bombs if bombs else None
                ),
                "coins_found_per_bomb": (
                    sum(row["coins_found"] for row in group) / bombs if bombs else None
                ),
                "self_kills": sum(row["self_kills"] for row in group),
                "self_kill_rate": sum(row["self_kills"] for row in group) / len(group),
                "invalid_action_rate": invalid / attempted if attempted else None,
                **action_counts,
                "decision_time_p95_ms": max(decision_p95) if decision_p95 else None,
                "decision_time_max_ms": max(decision_max) if decision_max else None,
            }
        )
    return summaries


def _comparisons(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    collection: dict[
        tuple[str, str],
        dict[str, dict[int, float]],
    ] = defaultdict(dict)
    self_kills: dict[str, dict[str, dict[int, float]]] = defaultdict(dict)

    for row in rows:
        treatment = row["treatment"]
        scenario = row["scenario"]
        model = row["model"]
        world_seed = row["world_seed"]

        model_collection = collection[
            (treatment, scenario)
        ].setdefault(model, {})

        if world_seed in model_collection:
            raise ValueError(
                "Duplicate primary evaluation episode for "
                f"{treatment}/{model}/{scenario}/{world_seed}"
            )

        model_collection[world_seed] = row["collection_fraction"]

        if scenario == "classic":
            model_self_kills = self_kills[treatment].setdefault(model, {})
            model_self_kills[world_seed] = float(row["self_kills"])

    return {
        "classic_self_kill_candidate_minus_control": paired_bootstrap(
            self_kills["candidate"],
            self_kills["control"],
            resampler_seed=135,
        ).as_dict(),
        "classic_collection_candidate_minus_control": paired_bootstrap(
            collection[("candidate", "classic")],
            collection[("control", "classic")],
            resampler_seed=136,
        ).as_dict(),
    }


def _criteria(
    rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    comparisons: dict[str, Any],
    training_diagnostics: list[dict[str, Any]],
    deterministic: bool,
    repeat_p95_ok: bool,
    repeat_max_ok: bool,
) -> dict[str, bool]:
    """Evaluate every preregistered Issue #135 decision criterion."""

    self_kill_comparison = comparisons[
        "classic_self_kill_candidate_minus_control"
    ]
    collection_comparison = comparisons[
        "classic_collection_candidate_minus_control"
    ]
    replica_self_kill_rates: dict[str, dict[str, float]] = defaultdict(dict)

    for summary in summaries:
        if summary["scenario"] == "classic":
            replica_self_kill_rates[
                summary["treatment"]
            ][summary["model"]] = summary["self_kill_rate"]

    shared_models = sorted(
        set(replica_self_kill_rates["candidate"])
        & set(replica_self_kill_rates["control"])
    )

    if len(shared_models) != 5:
        raise ValueError(
            "Classic comparison must contain five paired replicas"
        )

    replicas_with_lower_self_kill = sum(
        replica_self_kill_rates["candidate"][model]
        < replica_self_kill_rates["control"][model]
        for model in shared_models
    )

    mean_visits = {
        treatment: fmean(
            item["mean_visits_per_state"]
            for item in training_diagnostics
            if item["treatment"] == treatment
        )
        for treatment in PLAN_IDS
    }

    if mean_visits["control"] <= 0:
        raise ValueError(
            "Control mean visits per state must be positive"
        )

    primary_p95_ok = all(
        summary["decision_time_p95_ms"] is not None
        and summary["decision_time_p95_ms"] < 50.0
        for summary in summaries
    )
    primary_max_ok = all(
        summary["decision_time_max_ms"] is not None
        and summary["decision_time_max_ms"] < 100.0
        for summary in summaries
    )

    return {
        "classic_self_kill_rate_below_control": (
            self_kill_comparison["mean_difference"] < 0.0
        ),
        "classic_self_kill_ci_upper_below_zero": (
            self_kill_comparison["ci_upper"] < 0.0
        ),
        "at_least_four_replicas_lower_self_kill": (
            replicas_with_lower_self_kill >= 4
        ),
        "classic_collection_decrease_within_0_05": (
            collection_comparison["mean_difference"] >= -0.05
        ),
        "mean_visits_per_state_ratio_at_least_0_90": (
            mean_visits["candidate"] / mean_visits["control"] >= 0.90
        ),
        "deterministic_repeats": deterministic,
        "decision_time_p95_within_limit": (
            primary_p95_ok and repeat_p95_ok
        ),
        "decision_time_max_within_limit": (
            primary_max_ok and repeat_max_ok
        ),
    }


def _evaluation_unseen_rate(
    rows: list[dict[str, Any]],
    treatment: str,
) -> float:
    """Return the ratio of unseen decisions across primary evaluations."""

    treatment_rows = [
        row for row in rows if row["treatment"] == treatment
    ]

    if not treatment_rows:
        raise ValueError(
            f"No evaluation rows for treatment {treatment}"
        )

    if any(
        row.get("evaluation_decisions") is None
        or row.get("evaluation_unseen_decisions") is None
        for row in treatment_rows
    ):
        raise ValueError(
            f"Missing evaluation state coverage for {treatment}"
        )

    decisions = sum(
        row["evaluation_decisions"] for row in treatment_rows
    )
    unseen_decisions = sum(
        row["evaluation_unseen_decisions"]
        for row in treatment_rows
    )

    if decisions <= 0:
        raise ValueError(
            f"No encoded evaluation decisions for {treatment}"
        )

    return unseen_decisions / decisions


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


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
