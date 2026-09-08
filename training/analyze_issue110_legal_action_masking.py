"""Analyze the preregistered Issue #110 legal-action-masking experiment."""

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
    / "2026-09-07-task2-legal-action-masking-DerKleineSprengstoffkapitalist"
)

PLAN_IDS = {
    "unmasked": "issue110-tabular-task2-unmasked",
    "masked": "issue110-tabular-task2-masked",
    "task1": "issue110-tabular-task1-frozen",
}

EXPECTED_MASKING_MODES = {
    "unmasked": "none",
    "masked": "framework_legal",
    "task1": "none",
}

TREATMENT_SUITES = {
    "unmasked": {
        "classic-primary": "classic",
        "coin-heaven-primary": "coin-heaven",
        "loot-crate-primary": "loot-crate",
    },
    "masked": {
        "classic-primary": "classic",
        "coin-heaven-primary": "coin-heaven",
        "loot-crate-primary": "loot-crate",
    },
    "task1": {
        "coin-heaven-primary": "coin-heaven",
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
)


def analyze(
    plan_root: Path = PLAN_ROOT,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Validate outputs and evaluate the registered Issue #110 criteria."""

    plan_root = Path(plan_root).resolve()
    output = Path(output).resolve()

    primary_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    deterministic = True

    for treatment, plan_id in PLAN_IDS.items():
        plan_directory = plan_root / plan_id
        status = _read_json(plan_directory / "status.json")
        resolved = _read_json(plan_directory / "resolved_plan.json")

        if status.get("status") != "completed":
            raise ValueError(f"Run plan is not completed: {plan_id}")

        expected_mode = EXPECTED_MASKING_MODES[treatment]

        if resolved.get("action_masking") != expected_mode:
            raise ValueError(
                f"Unexpected action-masking mode for {plan_id}: {resolved.get('action_masking')!r}"
            )

        replicas = [item["replica_id"] for item in resolved["replicas"]]

        for replica in replicas:
            model = replica if treatment != "task1" else "task1"

            for suite_id, scenario in TREATMENT_SUITES[treatment].items():
                primary = _suite_rows(
                    plan_directory,
                    status,
                    resolved,
                    replica,
                    suite_id,
                )
                repeat = _suite_rows(
                    plan_directory,
                    status,
                    resolved,
                    replica,
                    suite_id.replace("-primary", "-repeat"),
                )

                for primary_row, repeat_row in zip(
                    primary,
                    repeat,
                    strict=True,
                ):
                    if primary_row["world_seed"] != repeat_row["world_seed"]:
                        raise ValueError(
                            f"Repeat seed mismatch for {treatment}/{replica}/{scenario}"
                        )

                    if primary_row["agent_seed"] != repeat_row["agent_seed"]:
                        raise ValueError(
                            f"Repeat agent-seed mismatch for {treatment}/{replica}/{scenario}"
                        )

                    for column in DETERMINISTIC_COLUMNS:
                        if primary_row.get(column) is None:
                            raise ValueError(f"Primary evaluation is missing {column}")

                        if repeat_row.get(column) is None:
                            raise ValueError(f"Repeat evaluation is missing {column}")

                        if primary_row.get(column) != repeat_row.get(column):
                            deterministic = False

                    available = primary_row.get("initially_available_coins")

                    if not isinstance(available, int) or available <= 0:
                        raise ValueError(
                            f"Missing available-coin count for {treatment}/{replica}/{scenario}"
                        )

                    common = {
                        "treatment": treatment,
                        "model": model,
                        "scenario": scenario,
                    }

                    prepared_primary = {
                        **primary_row,
                        **common,
                        "evaluation_pass": "primary",
                        "collection_fraction": (primary_row["coins_collected"] / available),
                    }

                    prepared_repeat = {
                        **repeat_row,
                        **common,
                        "evaluation_pass": "repeat",
                        "collection_fraction": (repeat_row["coins_collected"] / available),
                    }

                    primary_rows.append(prepared_primary)
                    evidence_rows.extend((prepared_primary, prepared_repeat))

    summaries = _summarize(primary_rows)
    comparisons = _comparisons(primary_rows)
    criteria = _criteria(
        primary_rows,
        summaries,
        comparisons,
        deterministic,
    )

    result = {
        "schema_version": 1,
        "issue": 110,
        "deterministic": deterministic,
        "primary_evaluation_episodes": len(primary_rows),
        "repeat_evaluation_episodes": len(primary_rows),
        "summaries": summaries,
        "comparisons": comparisons,
        "criteria": criteria,
        "passed": all(criteria.values()),
        "decision": (
            "adopt_framework_legal_action_masking"
            if all(criteria.values())
            else "retain_unmasked_baseline"
        ),
    }

    output.mkdir(parents=True, exist_ok=True)

    _write_csv(output / "summary.csv", summaries)
    _write_csv(output / "evidence.csv", evidence_rows)

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
    """Load and validate all rows belonging to one evaluation suite."""

    expected = {
        job["run_id"]: job
        for job in resolved["jobs"]
        if (
            job["kind"] == "evaluation"
            and job["replica"] == replica
            and job["stage_or_suite"] == suite_id
        )
    }

    actual_keys = {run_id for run_id in status["jobs"] if run_id in expected}

    if actual_keys != set(expected):
        raise ValueError(f"Evaluation matrix mismatch for {replica}/{suite_id}")

    rows: list[dict[str, Any]] = []

    for run_id in sorted(expected):
        job = status["jobs"][run_id]
        attempts = job.get("attempts", [])

        if job.get("status") != "completed" or not attempts:
            raise ValueError(f"Evaluation job is incomplete: {run_id}")

        run_directory = plan_directory / attempts[-1]["output"]
        episode_rows = read_episodes_csv(run_directory / "episodes.csv")

        if len(episode_rows) != 1:
            raise ValueError(f"Expected exactly one episode row: {run_id}")

        metadata = _read_json(run_directory / "metadata.json")
        registered = expected[run_id]

        for field in (
            "world_seed",
            "agent_seed",
            "scenario",
            "rounds",
            "opponents",
        ):
            if metadata.get(field) != registered[field]:
                raise ValueError(f"Metadata mismatch for {run_id}: {field}")

        if metadata.get("mode") != "evaluation":
            raise ValueError(f"Job is not an evaluation: {run_id}")

        row = dict(episode_rows[0])
        row["world_seed"] = metadata["world_seed"]
        row["agent_seed"] = metadata["agent_seed"]
        rows.append(row)

    return rows


def _summarize(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate primary results by treatment, model and scenario."""

    groups: dict[
        tuple[str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in rows:
        groups[
            (
                row["treatment"],
                row["model"],
                row["scenario"],
            )
        ].append(row)

    summaries: list[dict[str, Any]] = []

    for (treatment, model, scenario), group in sorted(groups.items()):
        attempted = sum(row["attempted_actions"] for row in group)
        invalid = sum(row["invalid_actions"] for row in group)
        coins = sum(row["coins_collected"] for row in group)
        survival_steps = sum(row["survival_steps"] for row in group)
        bombs = sum(row["bombs_dropped"] for row in group)

        decision_p95 = [
            row["decision_time_p95_ms"] for row in group if row["decision_time_p95_ms"] is not None
        ]
        decision_max = [
            row["decision_time_max_ms"] for row in group if row["decision_time_max_ms"] is not None
        ]

        summaries.append(
            {
                "treatment": treatment,
                "model": model,
                "scenario": scenario,
                "episodes": len(group),
                "mean_collection_fraction": fmean(row["collection_fraction"] for row in group),
                "mean_coins": coins / len(group),
                "full_clear_rate": (
                    sum(row["coins_collected"] == row["initially_available_coins"] for row in group)
                    / len(group)
                ),
                "zero_coin_rate": (sum(row["coins_collected"] == 0 for row in group) / len(group)),
                "survival_rate": (sum(row["survived"] for row in group) / len(group)),
                "self_kill_rate": (sum(row["self_kills"] for row in group) / len(group)),
                "invalid_action_rate": (invalid / attempted if attempted else 0.0),
                "mean_survival_steps": survival_steps / len(group),
                "steps_per_coin": (survival_steps / coins if coins else None),
                "mean_crates_destroyed": fmean(row["crates_destroyed"] for row in group),
                "mean_coins_found": fmean(row["coins_found"] for row in group),
                "bombs_dropped": bombs,
                "action_up": sum(row["action_up"] for row in group),
                "action_right": sum(row["action_right"] for row in group),
                "action_down": sum(row["action_down"] for row in group),
                "action_left": sum(row["action_left"] for row in group),
                "action_wait": sum(row["action_wait"] for row in group),
                "action_bomb": sum(row["action_bomb"] for row in group),
                "action_unknown": sum(row["action_unknown"] for row in group),
                "decision_time_p95_ms": (max(decision_p95) if decision_p95 else None),
                "decision_time_max_ms": (max(decision_max) if decision_max else None),
            }
        )

    return summaries


def _values_by_model_and_seed(
    rows: list[dict[str, Any]],
    *,
    treatment: str,
    scenario: str,
    metric: str,
) -> dict[str, dict[int, float]]:
    """Return values arranged for the two-stage paired bootstrap."""

    values: dict[str, dict[int, float]] = {}

    for row in rows:
        if row["treatment"] != treatment or row["scenario"] != scenario:
            continue

        model = row["model"]
        seed = int(row["world_seed"])

        if seed in values.setdefault(model, {}):
            raise ValueError(f"Duplicate row for {treatment}/{model}/{scenario}/{seed}")

        values[model][seed] = float(row[metric])

    if not values:
        raise ValueError(f"No rows for {treatment}/{scenario}/{metric}")

    return values


def _comparisons(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calculate all registered paired comparisons."""

    comparisons: dict[str, Any] = {}

    for scenario in ("classic", "coin-heaven", "loot-crate"):
        masked = _values_by_model_and_seed(
            rows,
            treatment="masked",
            scenario=scenario,
            metric="collection_fraction",
        )
        unmasked = _values_by_model_and_seed(
            rows,
            treatment="unmasked",
            scenario=scenario,
            metric="collection_fraction",
        )

        comparisons[f"{scenario}_masked_minus_unmasked"] = paired_bootstrap(
            masked,
            unmasked,
            resampler_seed=110,
        ).as_dict()

    masked_coin = _values_by_model_and_seed(
        rows,
        treatment="masked",
        scenario="coin-heaven",
        metric="collection_fraction",
    )
    task1_coin = _values_by_model_and_seed(
        rows,
        treatment="task1",
        scenario="coin-heaven",
        metric="collection_fraction",
    )

    task1_reference = next(iter(task1_coin.values()))
    task1_by_replica = {replica: dict(task1_reference) for replica in masked_coin}

    comparisons["coin_heaven_masked_minus_task1"] = paired_bootstrap(
        masked_coin,
        task1_by_replica,
        resampler_seed=111,
    ).as_dict()

    return comparisons


def _criteria(
    rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    comparisons: dict[str, Any],
    deterministic: bool,
) -> dict[str, bool]:
    """Evaluate the prospectively registered success criteria."""

    classic_comparison = comparisons["classic_masked_minus_unmasked"]
    retention_comparison = comparisons["coin_heaven_masked_minus_task1"]

    masked_classic = [
        item
        for item in summaries
        if (item["treatment"] == "masked" and item["scenario"] == "classic")
    ]
    unmasked_classic = {
        item["model"]: item
        for item in summaries
        if (item["treatment"] == "unmasked" and item["scenario"] == "classic")
    }

    improving_replicas = sum(
        item["mean_collection_fraction"]
        > unmasked_classic[item["model"]]["mean_collection_fraction"]
        for item in masked_classic
    )

    masked_rows = [row for row in rows if row["treatment"] == "masked"]
    masked_attempted = sum(row["attempted_actions"] for row in masked_rows)
    masked_invalid = sum(row["invalid_actions"] for row in masked_rows)
    masked_invalid_rate = masked_invalid / masked_attempted if masked_attempted else 0.0

    masked_classic_rows = [
        row for row in rows if (row["treatment"] == "masked" and row["scenario"] == "classic")
    ]
    unmasked_classic_rows = [
        row for row in rows if (row["treatment"] == "unmasked" and row["scenario"] == "classic")
    ]

    masked_self_kill_rate = sum(row["self_kills"] for row in masked_classic_rows) / len(
        masked_classic_rows
    )
    unmasked_self_kill_rate = sum(row["self_kills"] for row in unmasked_classic_rows) / len(
        unmasked_classic_rows
    )

    masked_coin_summaries = [
        item
        for item in summaries
        if (item["treatment"] == "masked" and item["scenario"] == "coin-heaven")
    ]

    timed_rows = [
        row
        for row in rows
        if (
            row.get("decision_time_p95_ms") is not None
            and row.get("decision_time_max_ms") is not None
        )
    ]

    return {
        "classic_mean_improvement_above_zero": (classic_comparison["mean_difference"] > 0.0),
        "classic_ci_lower_above_zero": (classic_comparison["ci_lower"] > 0.0),
        "at_least_four_of_five_replicas_improve": (improving_replicas >= 4),
        "masked_invalid_action_rate_below_0_01": (masked_invalid_rate < 0.01),
        "task1_retention_ci_above_minus_0_05": (retention_comparison["ci_lower"] > -0.05),
        "coin_heaven_has_no_bomb_actions": all(
            item["action_bomb"] == 0 for item in masked_coin_summaries
        ),
        "self_kill_increase_at_most_0_02": (
            masked_self_kill_rate <= unmasked_self_kill_rate + 0.02
        ),
        "decision_time_p95_below_50_ms": (
            bool(timed_rows) and all(row["decision_time_p95_ms"] < 50.0 for row in timed_rows)
        ),
        "decision_time_max_below_100_ms": (
            bool(timed_rows) and all(row["decision_time_max_ms"] < 100.0 for row in timed_rows)
        ),
        "deterministic_repeats": deterministic,
    }


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """Write rows with a stable union of all available columns."""

    if not rows:
        raise ValueError(f"No rows to write: {path}")

    fieldnames: list[str] = []

    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    """Read and validate a JSON object."""

    if not path.is_file():
        raise FileNotFoundError(path)

    value = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")

    return value


def parse_arguments(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-root",
        type=Path,
        default=PLAN_ROOT,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the Issue #110 analysis."""

    arguments = parse_arguments(argv)

    try:
        result = analyze(
            arguments.plan_root,
            arguments.output,
        )
    except Exception as error:
        print(f"Analysis failed: {error}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            result["comparisons"],
            indent=2,
            sort_keys=True,
        )
    )
    print(
        json.dumps(
            result["criteria"],
            indent=2,
            sort_keys=True,
        )
    )
    print(f"Overall pass: {result['passed']}")
    print(f"Decision: {result['decision']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
