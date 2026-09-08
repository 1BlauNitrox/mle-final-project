"""Validate, analyze, and select the Issue #107 Task 2 factorial outcome."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

from training.aggregate import read_episodes_csv
from training.run_experiment import REPOSITORY_ROOT
from training.run_issue107_campaign import (
    CELL_TREATMENTS,
    CPU_HOURS_MAX,
    MEMORY_GIB_MAX,
    PLAN_PATHS,
    WALL_CLOCK_HOURS_MAX,
    validate_protocol,
)
from training.run_plan import load_plan

PLAN_ROOT = REPOSITORY_ROOT / "training_outputs" / "run-plans"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "training_outputs" / "issue107-analysis"
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
ACTION_COLUMNS = (
    "action_up",
    "action_right",
    "action_down",
    "action_left",
    "action_wait",
    "action_bomb",
    "action_unknown",
)
CONTRAST_SEEDS = {
    "escape_b_minus_a": 10_701,
    "escape_d_minus_c": 10_702,
    "replay_c_minus_a": 10_703,
    "replay_d_minus_b": 10_704,
    "interaction": 10_705,
}


def analyze(plan_root: Path = PLAN_ROOT, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Require complete evidence and apply the prospective factorial decision rule."""
    validate_protocol()
    plan_root = Path(plan_root).resolve()
    output = Path(output).resolve()
    campaign = _validate_campaign_execution(plan_root)
    rows: list[dict[str, Any]] = []
    deterministic = True
    repeat_latency_ok = True
    resource_rows: list[dict[str, Any]] = []
    cell_fingerprints: list[dict[str, Any]] = []

    for treatment in ("A", "B", "C", "D", "untrained", "frozen_task1"):
        plan_id = PLAN_PATHS[treatment].stem
        plan_directory = plan_root / plan_id
        status = _read_json(plan_directory / "status.json")
        resolved = _read_json(plan_directory / "resolved_plan.json")
        if status.get("status") != "completed":
            raise ValueError(f"Run plan is not completed: {plan_id}")
        _validate_registered_plan(treatment, resolved)
        if treatment in CELL_TREATMENTS:
            cell_fingerprints.append(resolved["fingerprints"])
            resource_rows.extend(
                _training_resources(treatment, plan_directory, status, resolved, campaign)
            )

        replicas = [item["replica_id"] for item in resolved["replicas"]]
        suites = (
            PRIMARY_SUITES
            if treatment != "frozen_task1"
            else {"coin-heaven-primary": "coin-heaven"}
        )
        for replica in replicas:
            model = replica if treatment in CELL_TREATMENTS else treatment
            artifact_hashes: set[str] = set()
            for suite_id, scenario in suites.items():
                primary_rows, primary_hashes = _suite_rows(
                    plan_directory,
                    status,
                    resolved,
                    replica,
                    suite_id,
                    treatment,
                    campaign,
                )
                repeat_rows, repeat_hashes = _suite_rows(
                    plan_directory,
                    status,
                    resolved,
                    replica,
                    suite_id.replace("-primary", "-repeat"),
                    treatment,
                    campaign,
                )
                artifact_hashes.update(primary_hashes | repeat_hashes)
                for primary, repeat in zip(primary_rows, repeat_rows, strict=True):
                    if (
                        primary["world_seed"],
                        primary["agent_seed"],
                    ) != (repeat["world_seed"], repeat["agent_seed"]):
                        raise ValueError(
                            f"Repeat seed mismatch for {treatment}/{replica}/{scenario}"
                        )
                    if any(
                        primary.get(name) is None or repeat.get(name) is None
                        for name in DETERMINISTIC_COLUMNS
                    ):
                        raise ValueError("Missing deterministic evidence in evaluation repeat")
                    deterministic &= all(
                        primary[name] == repeat[name] for name in DETERMINISTIC_COLUMNS
                    )
                    repeat_latency_ok &= _latency_ok(repeat)
                    available = primary.get("initially_available_coins")
                    if not isinstance(available, int) or available <= 0:
                        raise ValueError(
                            f"Missing available-coin count for {treatment}/{replica}/{scenario}"
                        )
                    rows.append(
                        {
                            **primary,
                            "cell": treatment,
                            "model": model,
                            "scenario": scenario,
                            "collection_fraction": primary["coins_collected"] / available,
                        }
                    )
            if len(artifact_hashes) != 1:
                raise ValueError(f"Evaluation artifact changed for {treatment}/{replica}")

    _require_common_cell_fingerprints(cell_fingerprints)
    summaries = _summarize(rows)
    comparisons = _comparisons(rows)
    absolute_gates = {
        cell: _absolute_gates(cell, rows, summaries, comparisons, deterministic)
        for cell in CELL_TREATMENTS
    }
    eligibility = _eligibility(comparisons)
    selected_cell = _select_cell(eligibility, absolute_gates, summaries)
    selected_replica = _select_representative(selected_cell, summaries)
    selected_artifact = _selected_artifact(
        plan_root / PLAN_PATHS[selected_cell].stem, selected_replica
    )
    criteria = {
        "all_plans_complete": True,
        "deterministic_repeats": deterministic,
        "repeat_latency_within_limits": repeat_latency_ok,
        "evaluation_artifacts_immutable": True,
        "authorized_bounded_campaign": True,
        "all_twenty_final_artifacts_retained": len(
            {(item["cell"], item["model"]) for item in summaries if item["cell"] in CELL_TREATMENTS}
        )
        == 20,
    }
    result = {
        "schema_version": 1,
        "issue": 107,
        "scientific_result": True,
        "primary_evaluation_episodes": len(rows),
        "repeat_evaluation_episodes": len(rows),
        "summaries": summaries,
        "comparisons": comparisons,
        "absolute_gates": absolute_gates,
        "eligibility": eligibility,
        "selection": {
            "cell": selected_cell,
            "replica": selected_replica,
            "artifact": selected_artifact,
            "task2_complete": absolute_gates[selected_cell]["overall_passed"],
            "task3_status": (
                "eligible_predecessor"
                if absolute_gates[selected_cell]["overall_passed"]
                else "exploratory_predecessor_task2_not_complete"
            ),
        },
        "resources": _resource_summary(resource_rows),
        "criteria": criteria,
        "analysis_valid": all(criteria.values()),
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "summary.csv", summaries)
    _write_csv(output / "training-resources.csv", resource_rows)
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _validate_registered_plan(treatment: str, resolved: dict[str, Any]) -> None:
    """Bind evidence to every field and fingerprint in the reviewed plan."""
    registered = load_plan(PLAN_PATHS[treatment]).to_dict()
    portable_resolved = _without_parent_locations(resolved)
    portable_registered = _without_parent_locations(registered)
    if portable_resolved != portable_registered:
        changed = sorted(
            key
            for key in set(portable_resolved) | set(portable_registered)
            if portable_resolved.get(key) != portable_registered.get(key)
        )
        raise ValueError(
            f"Resolved plan does not match registered plan for {treatment}: {changed}"
        )

    expected = CELL_TREATMENTS.get(treatment)
    if (
        expected is not None
        and (
            resolved.get("escape_continuations"),
            resolved.get("replay_treatment"),
        )
        != expected
    ):
        raise ValueError(f"Resolved treatment mismatch for cell {treatment}")
    if treatment != "frozen_task1" and resolved.get("action_masking") != "none":
        raise ValueError(f"Action masking must remain off for {treatment}")


def _without_parent_locations(plan: dict[str, Any]) -> dict[str, Any]:
    """Ignore only machine-local parent paths while retaining their hashes."""
    return {
        **plan,
        "replicas": [
            {**replica, "parent_artifact": None}
            for replica in plan.get("replicas", [])
        ],
    }


def _validate_campaign_execution(plan_root: Path) -> dict[str, Any]:
    """Require the unique authorization and bounded resource record."""
    record_root = plan_root.parent
    authorizations = sorted(record_root.glob("issue107-campaign-authorization-*.json"))
    resources = sorted(record_root.glob("issue107-campaign-resources-*.json"))
    if len(authorizations) != 1 or len(resources) != 1:
        raise ValueError("Analysis requires unique campaign authorization and resource records")

    authorization = _read_json(authorizations[0])
    resource = _read_json(resources[0])
    reviewed_commit = authorization.get("reviewed_commit")
    expected_limits = {
        "cpu_seconds": CPU_HOURS_MAX * 60 * 60,
        "wall_seconds": WALL_CLOCK_HOURS_MAX * 60 * 60,
        "memory_bytes": MEMORY_GIB_MAX * 1024**3,
    }
    expected_campaign = {
        "schema_version": 1,
        "issue": 107,
        "reviewed_commit": reviewed_commit,
        "authorized_at": authorization.get("authorized_at"),
        "cpu_hours_max": CPU_HOURS_MAX,
        "wall_clock_hours_max": WALL_CLOCK_HOURS_MAX,
        "memory_gib_max": MEMORY_GIB_MAX,
        "max_training_workers": 4,
        "evaluation_workers": 1,
    }
    authorization_fields = {
        "issue": 107,
        "compute_authorized": True,
        "cpu_hours_max": CPU_HOURS_MAX,
        "wall_clock_hours_max": WALL_CLOCK_HOURS_MAX,
        "memory_gib_max": MEMORY_GIB_MAX,
        "max_training_workers": 4,
        "evaluation_workers": 1,
    }
    if any(
        authorization.get(key) != value
        for key, value in authorization_fields.items()
    ):
        raise ValueError("Campaign authorization does not match the registered protocol")
    if not str(authorization.get("authorized_by", "")).strip() or not str(
        authorization.get("hardware_description", "")
    ).strip():
        raise ValueError("Campaign authorization is missing its human or hardware record")
    if not isinstance(reviewed_commit, str) or len(reviewed_commit) != 40:
        raise ValueError("Campaign authorization has no full reviewed commit")
    if authorizations[0].stem != f"issue107-campaign-authorization-{reviewed_commit}":
        raise ValueError("Campaign authorization filename does not match its reviewed commit")
    if resources[0].stem != f"issue107-campaign-resources-{reviewed_commit}":
        raise ValueError("Campaign resource filename does not match the reviewed commit")
    if resource.get("authorized_at") != authorization.get("authorized_at"):
        raise ValueError("Campaign resource record belongs to another authorization")
    if resource.get("limits") != expected_limits or resource.get("limit_reached") is not None:
        raise ValueError("Campaign resource record is breached or uses different limits")
    usage = (
        ("cpu_seconds_consumed", expected_limits["cpu_seconds"]),
        ("wall_seconds_elapsed", expected_limits["wall_seconds"]),
        ("peak_memory_bytes", expected_limits["memory_bytes"]),
    )
    if any(
        not isinstance(resource.get(key), (int, float)) or resource[key] > limit
        for key, limit in usage
    ):
        raise ValueError("Campaign resource usage exceeds the registered limits")
    if resource.get("active_root_pids") != []:
        raise ValueError("Campaign resource record still has active processes")
    return expected_campaign


def _validate_job_campaign(
    metadata: dict[str, Any], expected_campaign: dict[str, Any], job_id: str
) -> None:
    if metadata.get("git_commit") != expected_campaign["reviewed_commit"]:
        raise ValueError(f"Reviewed commit mismatch for {job_id}")
    if metadata.get("run_plan", {}).get("campaign") != expected_campaign:
        raise ValueError(f"Campaign authorization metadata mismatch for {job_id}")


def _suite_rows(
    plan_directory: Path,
    status: dict[str, Any],
    resolved: dict[str, Any],
    replica: str,
    suite_id: str,
    treatment: str,
    campaign: dict[str, Any],
) -> tuple[list[dict[str, Any]], set[str]]:
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
    artifact_hashes: set[str] = set()
    for key in keys:
        job = status["jobs"][key]
        attempts = job.get("attempts", [])
        if job.get("status") != "completed" or not attempts:
            raise ValueError(f"Evaluation job is not complete: {key}")
        artifact = job.get("artifact")
        if not isinstance(artifact, dict) or not artifact.get("sha256"):
            raise ValueError(f"Evaluation artifact record is missing: {key}")
        artifact_hashes.add(artifact["sha256"])
        run_directory = plan_directory / attempts[-1]["output"]
        episode_rows = read_episodes_csv(run_directory / "episodes.csv")
        if len(episode_rows) != 1:
            raise ValueError(f"Expected one episode row for {key}")
        metadata = _read_json(run_directory / "metadata.json")
        _validate_job_campaign(metadata, campaign, key)
        registered = expected[key]
        for field in ("world_seed", "agent_seed", "scenario", "rounds"):
            if metadata.get(field) != registered[field]:
                raise ValueError(f"Metadata mismatch for {key}: {field}")
        if metadata.get("mode") != "evaluation" or metadata.get("opponents") != []:
            raise ValueError(f"Metadata mismatch for {key}: evaluation conditions")
        run_plan = metadata.get("run_plan", {})
        expected_escape, expected_replay = CELL_TREATMENTS.get(
            treatment, (resolved["escape_continuations"], resolved["replay_treatment"])
        )
        if (
            run_plan.get("action_masking"),
            run_plan.get("escape_continuations"),
            run_plan.get("replay_treatment"),
            run_plan.get("processes"),
            run_plan.get("artifact_writable"),
        ) != ("none", expected_escape, expected_replay, 1, False):
            raise ValueError(f"Treatment or execution metadata mismatch for {key}")
        row = dict(episode_rows[0])
        row["world_seed"] = metadata["world_seed"]
        row["agent_seed"] = metadata["agent_seed"]
        result.append(row)
    return result, artifact_hashes


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["cell"], row["model"], row["scenario"])].append(row)
    summaries: list[dict[str, Any]] = []
    for (cell, model, scenario), group in sorted(groups.items()):
        attempted = sum(row["attempted_actions"] for row in group)
        invalid = sum(row["invalid_actions"] for row in group)
        coins = sum(row["coins_collected"] for row in group)
        survival_steps = sum(row["survival_steps"] for row in group)
        bombs = sum(row["bombs_dropped"] for row in group)
        summaries.append(
            {
                "cell": cell,
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
                "survival_rate": sum(bool(row["survived"]) for row in group) / len(group),
                "mean_survival_steps": survival_steps / len(group),
                "mean_episode_steps": fmean(row["episode_steps"] for row in group),
                "steps_per_coin": survival_steps / coins if coins else None,
                "coins_per_100_survival_steps": 100.0 * coins / survival_steps
                if survival_steps
                else None,
                "coins_found": sum(row["coins_found"] for row in group),
                "crates_destroyed": sum(row["crates_destroyed"] for row in group),
                "bombs_dropped": bombs,
                "crates_per_bomb": sum(row["crates_destroyed"] for row in group) / bombs
                if bombs
                else None,
                "self_kill_rate": sum(row["self_kills"] for row in group) / len(group),
                "invalid_action_rate": invalid / attempted if attempted else None,
                **{name: sum(row[name] for row in group) for name in ACTION_COLUMNS},
                "decision_time_p95_ms": max(row["decision_time_p95_ms"] for row in group),
                "decision_time_max_ms": max(row["decision_time_max_ms"] for row in group),
            }
        )
    return summaries


def _comparisons(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = _metric_values(rows)
    comparisons: dict[str, Any] = {}
    efficacy = {
        "escape_b_minus_a": ("B", "A", "classic", "self_kill", -1.0),
        "escape_d_minus_c": ("D", "C", "classic", "self_kill", -1.0),
        "replay_c_minus_a": ("C", "A", "coin-heaven", "collection", 1.0),
        "replay_d_minus_b": ("D", "B", "coin-heaven", "collection", 1.0),
    }
    for name, (treatment, baseline, scenario, metric, direction) in efficacy.items():
        comparisons[name] = _contrast(
            values, treatment, baseline, scenario, metric, CONTRAST_SEEDS[name], direction
        )

    for treatment, baseline in (("B", "A"), ("C", "A"), ("D", "C"), ("D", "B")):
        guards: dict[str, Any] = {}
        for scenario in PRIMARY_SUITES.values():
            for metric in ("collection", "survival"):
                name = f"{treatment.lower()}_minus_{baseline.lower()}_{scenario}_{metric}"
                guards[f"{scenario}_{metric}"] = _contrast(
                    values,
                    treatment,
                    baseline,
                    scenario,
                    metric,
                    _stable_seed(name),
                    1.0,
                )
        comparisons[f"guards_{treatment.lower()}_minus_{baseline.lower()}"] = guards

    comparisons["interaction_d_minus_b_minus_c_plus_a"] = _interaction(
        values, CONTRAST_SEEDS["interaction"]
    )
    for cell in CELL_TREATMENTS:
        comparisons[f"{cell.lower()}_classic_minus_untrained"] = _contrast(
            values, cell, "untrained", "classic", "collection", 11_000 + ord(cell), 1.0
        )
        comparisons[f"{cell.lower()}_coin_heaven_minus_task1"] = _contrast(
            values,
            cell,
            "frozen_task1",
            "coin-heaven",
            "collection",
            12_000 + ord(cell),
            1.0,
        )
    return comparisons


def _metric_values(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, dict[int, float]]]:
    values: dict[tuple[str, str, str], dict[str, dict[int, float]]] = defaultdict(dict)
    for row in rows:
        for metric, value in (
            ("collection", row["collection_fraction"]),
            ("survival", float(bool(row["survived"]))),
            ("self_kill", float(row["self_kills"])),
        ):
            values[(row["cell"], row["scenario"], metric)].setdefault(row["model"], {})[
                row["world_seed"]
            ] = float(value)
    return values


def _contrast(
    values: dict[tuple[str, str, str], dict[str, dict[int, float]]],
    treatment: str,
    baseline: str,
    scenario: str,
    metric: str,
    resampler_seed: int,
    direction: float,
) -> dict[str, Any]:
    treatment_values = values[(treatment, scenario, metric)]
    baseline_values = values[(baseline, scenario, metric)]
    if len(baseline_values) == 1 and len(treatment_values) > 1:
        source = next(iter(baseline_values.values()))
        baseline_values = {model: dict(source) for model in treatment_values}
    differences = _paired_differences(treatment_values, baseline_values) * direction
    return _bootstrap_result(differences, resampler_seed)


def _interaction(
    values: dict[tuple[str, str, str], dict[str, dict[int, float]]],
    resampler_seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for scenario in PRIMARY_SUITES.values():
        for metric in ("collection", "survival", "self_kill"):
            arrays = [
                _ordered_values(values[(cell, scenario, metric)]) for cell in ("A", "B", "C", "D")
            ]
            if any(array.shape != arrays[0].shape for array in arrays[1:]):
                raise ValueError("Factorial interaction cells are not fully paired")
            interaction = arrays[3] - arrays[1] - arrays[2] + arrays[0]
            result[f"{scenario}_{metric}"] = _bootstrap_result(interaction, resampler_seed)
    return result


def _paired_differences(
    treatment: dict[str, dict[int, float]], baseline: dict[str, dict[int, float]]
) -> np.ndarray:
    if set(treatment) != set(baseline):
        raise ValueError("Contrasts require the same paired replica IDs")
    rows: list[list[float]] = []
    for model in sorted(treatment):
        if set(treatment[model]) != set(baseline[model]):
            raise ValueError("Contrasts require identical paired world seeds")
        rows.append(
            [treatment[model][seed] - baseline[model][seed] for seed in sorted(treatment[model])]
        )
    return np.asarray(rows, dtype=float)


def _ordered_values(values: dict[str, dict[int, float]]) -> np.ndarray:
    return np.asarray(
        [[values[model][seed] for seed in sorted(values[model])] for model in sorted(values)],
        dtype=float,
    )


def _bootstrap_result(differences: np.ndarray, resampler_seed: int) -> dict[str, Any]:
    if differences.ndim != 2 or not differences.size:
        raise ValueError("Bootstrap requires a non-empty replica-by-seed matrix")
    generator = np.random.default_rng(resampler_seed)
    model_count, seed_count = differences.shape
    replicates = np.empty(10_000, dtype=float)
    for index in range(len(replicates)):
        models = generator.integers(0, model_count, size=model_count)
        seeds = generator.integers(0, seed_count, size=(model_count, seed_count))
        replicates[index] = differences[models[:, None], seeds].mean()
    lower95, upper95 = np.percentile(replicates, (2.5, 97.5))
    lower_adjusted, upper_adjusted = np.percentile(replicates, (0.625, 99.375))
    return {
        "mean_difference": float(differences.mean()),
        "ci95_lower": float(lower95),
        "ci95_upper": float(upper95),
        "bonferroni_98_75_lower": float(lower_adjusted),
        "bonferroni_98_75_upper": float(upper_adjusted),
        "resamples": 10_000,
        "resampler_seed": resampler_seed,
        "paired_models": model_count,
        "paired_seeds_per_model": seed_count,
    }


def _eligibility(comparisons: dict[str, Any]) -> dict[str, bool]:
    escape_b = comparisons["escape_b_minus_a"]
    escape_d = comparisons["escape_d_minus_c"]
    replay_c = comparisons["replay_c_minus_a"]
    replay_d = comparisons["replay_d_minus_b"]
    return {
        "A": True,
        "B": _efficacy(escape_b, 0.15) and _guards(comparisons["guards_b_minus_a"]),
        "C": _efficacy(replay_c, 0.10) and _guards(comparisons["guards_c_minus_a"]),
        "D": _efficacy(escape_d, 0.15)
        and _efficacy(replay_d, 0.10)
        and _guards(comparisons["guards_d_minus_c"])
        and _guards(comparisons["guards_d_minus_b"]),
    }


def _efficacy(comparison: dict[str, Any], minimum: float) -> bool:
    return comparison["mean_difference"] >= minimum and comparison["bonferroni_98_75_lower"] > 0.0


def _guards(guards: dict[str, Any]) -> bool:
    return all(item["ci95_lower"] > -0.05 for item in guards.values())


def _absolute_gates(
    cell: str,
    rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    comparisons: dict[str, Any],
    deterministic: bool,
) -> dict[str, bool]:
    classic = _summary_group(summaries, cell, "classic")
    coin = _summary_group(summaries, cell, "coin-heaven")
    untrained = _summary_group(summaries, "untrained", "classic")[0]
    task1 = _summary_group(summaries, "frozen_task1", "coin-heaven")[0]
    classic_rows = [row for row in rows if row["cell"] == cell and row["scenario"] == "classic"]
    attempted = sum(row["attempted_actions"] for row in classic_rows)
    invalid = sum(row["invalid_actions"] for row in classic_rows)
    task2 = {
        "mean_fraction_at_least_0_30": fmean(item["mean_collection_fraction"] for item in classic)
        >= 0.30,
        "improvement_at_least_0_10": comparisons[f"{cell.lower()}_classic_minus_untrained"][
            "mean_difference"
        ]
        >= 0.10,
        "improvement_ci_above_zero": comparisons[f"{cell.lower()}_classic_minus_untrained"][
            "ci95_lower"
        ]
        > 0.0,
        "four_of_five_improve": sum(
            item["mean_collection_fraction"] > untrained["mean_collection_fraction"]
            for item in classic
        )
        >= 4,
        "self_kill_rate_at_most_0_20": sum(row["self_kills"] for row in classic_rows)
        / len(classic_rows)
        <= 0.20,
        "invalid_rate_at_most_0_10": invalid / attempted <= 0.10,
        "each_invalid_rate_at_most_0_15": all(
            item["invalid_action_rate"] is not None and item["invalid_action_rate"] <= 0.15
            for item in classic
        ),
        "every_replica_destroys_and_finds": all(
            item["crates_destroyed"] > 0 and item["coins_found"] > 0 for item in classic
        ),
    }
    coin_attempted = sum(
        row["attempted_actions"]
        for row in rows
        if row["cell"] == cell and row["scenario"] == "coin-heaven"
    )
    coin_invalid = sum(
        row["invalid_actions"]
        for row in rows
        if row["cell"] == cell and row["scenario"] == "coin-heaven"
    )
    task1_gates = {
        "retention_ci_above_minus_0_05": comparisons[f"{cell.lower()}_coin_heaven_minus_task1"][
            "ci95_lower"
        ]
        > -0.05,
        "four_of_five_within_0_10": sum(
            item["mean_collection_fraction"] >= task1["mean_collection_fraction"] - 0.10
            for item in coin
        )
        >= 4,
        "aggregate_invalid_below_0_01": coin_invalid / coin_attempted < 0.01,
        "each_invalid_below_0_01": all(
            item["invalid_action_rate"] is not None and item["invalid_action_rate"] < 0.01
            for item in coin
        ),
        "no_bombs": all(item["action_bomb"] == 0 for item in coin),
    }
    compatibility = {
        "deterministic": deterministic,
        "p95_below_50_ms": all(
            item["decision_time_p95_ms"] < 50.0 for item in summaries if item["cell"] == cell
        ),
        "maximum_below_100_ms": all(
            item["decision_time_max_ms"] < 100.0 for item in summaries if item["cell"] == cell
        ),
    }
    return {
        **{f"task2_{name}": value for name, value in task2.items()},
        **{f"task1_{name}": value for name, value in task1_gates.items()},
        **{f"compatibility_{name}": value for name, value in compatibility.items()},
        "task2_passed": all(task2.values()),
        "task1_passed": all(task1_gates.values()),
        "compatibility_passed": all(compatibility.values()),
        "overall_passed": all(task2.values())
        and all(task1_gates.values())
        and all(compatibility.values()),
    }


def _select_cell(
    eligibility: dict[str, bool],
    absolute_gates: dict[str, dict[str, bool]],
    summaries: list[dict[str, Any]],
) -> str:
    eligible_treatments = [cell for cell in ("B", "C", "D") if eligibility[cell]]
    if not eligible_treatments:
        return "A"

    def rank(cell: str) -> tuple[Any, ...]:
        classic = _aggregate_summary(summaries, cell, "classic")
        coin = _aggregate_summary(summaries, cell, "coin-heaven")
        gates = absolute_gates[cell]
        return (
            gates["overall_passed"],
            gates["task2_passed"],
            gates["task1_passed"],
            classic["mean_collection_fraction"],
            -classic["self_kill_rate"],
            coin["mean_collection_fraction"],
            -ord(cell),
        )

    return max(eligible_treatments, key=rank)


def _select_representative(cell: str, summaries: list[dict[str, Any]]) -> str:
    classic = sorted(
        _summary_group(summaries, cell, "classic"),
        key=lambda item: (item["mean_collection_fraction"], item["model"]),
    )
    if len(classic) != 5:
        raise ValueError(f"Cell {cell} does not contain five representative candidates")
    return str(classic[2]["model"])


def _selected_artifact(plan_directory: Path, replica: str) -> dict[str, Any]:
    status = _read_json(plan_directory / "status.json")
    key = f"train-{replica}-classic-crates"
    artifact = status["jobs"].get(key, {}).get("artifact")
    if not isinstance(artifact, dict) or not artifact.get("sha256"):
        raise ValueError(f"Final artifact is missing for {replica}")
    path = plan_directory / artifact["path"]
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    if digest != artifact["sha256"]:
        raise ValueError(f"Selected artifact checksum mismatch for {replica}")
    return {**artifact, "size_bytes": path.stat().st_size}


def _training_resources(
    cell: str,
    plan_directory: Path,
    status: dict[str, Any],
    resolved: dict[str, Any],
    campaign: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job in resolved["jobs"]:
        if job["kind"] != "training":
            continue
        record = status["jobs"][job["run_id"]]
        attempt = record["attempts"][-1]
        metadata = _read_json(plan_directory / attempt["metadata"])
        _validate_job_campaign(metadata, campaign, job["run_id"])
        rows.append(
            {
                "cell": cell,
                "replica": job["replica"],
                "stage": job["stage_or_suite"],
                "episodes": job["rounds"],
                "duration_seconds": metadata["duration_seconds"],
                "git_commit": metadata["git_commit"],
                "python_version": metadata["python_version"],
            }
        )
    return rows


def _resource_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    commits = sorted({row["git_commit"] for row in rows})
    return {
        "training_jobs": len(rows),
        "training_episodes": sum(row["episodes"] for row in rows),
        "sum_process_duration_seconds": sum(row["duration_seconds"] for row in rows),
        "git_commits": commits,
        "single_immutable_commit": len(commits) == 1,
    }


def _require_common_cell_fingerprints(fingerprints: list[dict[str, Any]]) -> None:
    for field in ("source", "framework", "agent", "dependencies_sha256"):
        values = [item[field] for item in fingerprints]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"Factorial cells differ in protected {field} fingerprint")


def _summary_group(
    summaries: list[dict[str, Any]], cell: str, scenario: str
) -> list[dict[str, Any]]:
    return [item for item in summaries if item["cell"] == cell and item["scenario"] == scenario]


def _aggregate_summary(
    summaries: list[dict[str, Any]], cell: str, scenario: str
) -> dict[str, float]:
    group = _summary_group(summaries, cell, scenario)
    return {
        "mean_collection_fraction": fmean(item["mean_collection_fraction"] for item in group),
        "self_kill_rate": fmean(item["self_kill_rate"] for item in group),
    }


def _latency_ok(row: dict[str, Any]) -> bool:
    return (
        row.get("decision_time_p95_ms") is not None
        and row.get("decision_time_max_ms") is not None
        and row["decision_time_p95_ms"] < 50
        and row["decision_time_max_ms"] < 100
    )


def _stable_seed(value: str) -> int:
    return sum((index + 1) * ord(character) for index, character in enumerate(value))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty evidence table: {path}")
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
    parser.add_argument(
        "--validate-protocol",
        action="store_true",
        help="Validate the prospective matrix without reading results or writing output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    try:
        if arguments.validate_protocol:
            print(json.dumps(validate_protocol(), indent=2, sort_keys=True))
            return 0
        result = analyze(arguments.plan_root, arguments.output)
    except Exception as error:
        print(f"Analysis failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result["selection"], indent=2, sort_keys=True))
    print(f"Analysis valid: {result['analysis_valid']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
