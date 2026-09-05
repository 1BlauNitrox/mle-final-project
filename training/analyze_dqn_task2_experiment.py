"""Validate and summarize the preregistered Issue #46 run-plan outputs."""

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
DEFAULT_OUTPUT = REPOSITORY_ROOT / "training_outputs" / "issue-46-analysis"
PLAN_IDS = {
    "trained": "issue46-dqn-task2-trained",
    "untrained": "issue46-dqn-task2-untrained",
    "task1": "issue46-dqn-task1-frozen",
}
PRIMARY_SUITES = {
    "classic-primary": "classic",
    "coin-heaven-primary": "coin-heaven",
    "loot-crate-primary": "loot-crate",
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
)


def analyze(plan_root: Path = PLAN_ROOT, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Require completed deterministic plans and evaluate registered gates."""
    plan_root = Path(plan_root).resolve()
    output = Path(output).resolve()
    rows: list[dict[str, Any]] = []
    deterministic = True

    for treatment, plan_id in PLAN_IDS.items():
        plan_directory = plan_root / plan_id
        status = _read_json(plan_directory / "status.json")
        resolved = _read_json(plan_directory / "resolved_plan.json")
        if status.get("status") != "completed":
            raise ValueError(f"Run plan is not completed: {plan_id}")
        replicas = [item["replica_id"] for item in resolved["replicas"]]
        for replica in replicas:
            model = replica if treatment == "trained" else treatment
            for suite_id, scenario in PRIMARY_SUITES.items():
                primary_rows = _suite_rows(
                    plan_directory,
                    status,
                    replica,
                    suite_id,
                )
                repeat_rows = _suite_rows(
                    plan_directory,
                    status,
                    replica,
                    suite_id.replace("-primary", "-repeat"),
                )
                for primary, repeat in zip(primary_rows, repeat_rows, strict=True):
                    if primary["world_seed"] != repeat["world_seed"]:
                        raise ValueError(
                            f"Repeat seed mismatch for {treatment}/{replica}/{scenario}"
                        )
                    if any(primary.get(name) != repeat.get(name) for name in DETERMINISTIC_COLUMNS):
                        deterministic = False
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
    criteria = _criteria(rows, summaries, comparisons, deterministic)
    result = {
        "schema_version": 1,
        "issue": 46,
        "deterministic": deterministic,
        "primary_evaluation_episodes": len(rows),
        "repeat_evaluation_episodes": len(rows),
        "summaries": summaries,
        "comparisons": comparisons,
        "criteria": criteria,
        "passed": all(criteria.values()),
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_summary(output / "summary.csv", summaries)
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _suite_rows(
    plan_directory: Path,
    status: dict[str, Any],
    replica: str,
    suite_id: str,
) -> list[dict[str, Any]]:
    prefix = f"eval-{replica}-{suite_id}-seed-"
    keys = sorted(key for key in status["jobs"] if key.startswith(prefix))
    if len(keys) != 40:
        raise ValueError(f"Expected 40 jobs for {prefix}, found {len(keys)}")
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


def _comparisons(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fractions: dict[tuple[str, str], dict[str, dict[int, float]]] = defaultdict(dict)
    for row in rows:
        key = (row["treatment"], row["scenario"])
        fractions[key].setdefault(row["model"], {})[row["world_seed"]] = row["collection_fraction"]
    trained_classic = fractions[("trained", "classic")]
    trained_coin = fractions[("trained", "coin-heaven")]
    untrained = next(iter(fractions[("untrained", "classic")].values()))
    task1 = next(iter(fractions[("task1", "coin-heaven")].values()))
    untrained_by_model = {model: dict(untrained) for model in trained_classic}
    task1_by_model = {model: dict(task1) for model in trained_coin}
    return {
        "classic_trained_minus_untrained": paired_bootstrap(
            trained_classic,
            untrained_by_model,
            resampler_seed=46,
        ).as_dict(),
        "coin_heaven_trained_minus_task1": paired_bootstrap(
            trained_coin,
            task1_by_model,
            resampler_seed=47,
        ).as_dict(),
    }


def _criteria(
    rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    comparisons: dict[str, Any],
    deterministic: bool,
) -> dict[str, bool]:
    trained_classic = [
        item
        for item in summaries
        if item["treatment"] == "trained" and item["scenario"] == "classic"
    ]
    trained_coin = [
        item
        for item in summaries
        if item["treatment"] == "trained" and item["scenario"] == "coin-heaven"
    ]
    untrained_classic = next(
        item
        for item in summaries
        if item["treatment"] == "untrained" and item["scenario"] == "classic"
    )
    task1_coin = next(
        item
        for item in summaries
        if item["treatment"] == "task1" and item["scenario"] == "coin-heaven"
    )
    total_attempted = sum(
        row["attempted_actions"]
        for row in rows
        if row["treatment"] == "trained" and row["scenario"] == "classic"
    )
    total_invalid = sum(
        row["invalid_actions"]
        for row in rows
        if row["treatment"] == "trained" and row["scenario"] == "classic"
    )
    trained_classic_rows = [
        row for row in rows if row["treatment"] == "trained" and row["scenario"] == "classic"
    ]
    all_trained = [item for item in summaries if item["treatment"] == "trained"]
    return {
        "task2_mean_fraction_at_least_0_30": fmean(
            item["mean_collection_fraction"] for item in trained_classic
        )
        >= 0.30,
        "task2_improvement_at_least_0_10": comparisons["classic_trained_minus_untrained"][
            "mean_difference"
        ]
        >= 0.10,
        "task2_improvement_ci_above_zero": comparisons["classic_trained_minus_untrained"][
            "ci_lower"
        ]
        > 0.0,
        "task2_four_of_five_improve": sum(
            item["mean_collection_fraction"] > untrained_classic["mean_collection_fraction"]
            for item in trained_classic
        )
        >= 4,
        "task2_self_kill_rate_at_most_0_20": sum(row["self_kills"] for row in trained_classic_rows)
        / len(trained_classic_rows)
        <= 0.20,
        "task2_invalid_rate_at_most_0_10": (
            total_invalid / total_attempted if total_attempted else 0.0
        )
        <= 0.10,
        "task2_each_invalid_rate_at_most_0_15": all(
            item["invalid_action_rate"] is not None and item["invalid_action_rate"] <= 0.15
            for item in trained_classic
        ),
        "task2_every_model_destroys_and_finds": all(
            item["crates_destroyed"] > 0 and item["coins_found"] > 0 for item in trained_classic
        ),
        "task1_retention_ci_above_minus_0_05": comparisons["coin_heaven_trained_minus_task1"][
            "ci_lower"
        ]
        > -0.05,
        "task1_four_of_five_within_0_10": sum(
            item["mean_collection_fraction"] >= task1_coin["mean_collection_fraction"] - 0.10
            for item in trained_coin
        )
        >= 4,
        "task1_invalid_below_0_01": all(
            item["invalid_action_rate"] is not None and item["invalid_action_rate"] < 0.01
            for item in trained_coin
        ),
        "task1_no_bombs": all(item["action_bomb"] == 0 for item in trained_coin),
        "deterministic_repeats": deterministic,
        "p95_below_50_ms": all(
            item["decision_time_p95_ms"] is not None and item["decision_time_p95_ms"] < 50.0
            for item in all_trained
        ),
        "maximum_below_100_ms": all(
            item["decision_time_max_ms"] is not None and item["decision_time_max_ms"] < 100.0
            for item in all_trained
        ),
    }


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
