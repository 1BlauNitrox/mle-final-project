"""Validate and summarize the preregistered Issue #85 evaluation plans."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

from training.aggregate import read_episodes_csv
from training.run_experiment import REPOSITORY_ROOT

PLAN_ROOT = REPOSITORY_ROOT / "training_outputs" / "run-plans"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "training_outputs" / "issue85-analysis"
ARMS = {
    "frozen_parent": {
        "plan_id": "issue85-dqn-frozen-parent",
        "agent": "DagobertDuckDQN",
        "artifact": "checkpoint.pt",
        "sha256": "eb08e3f67b620ac2a253a2af4db3d5b4c6ea9e667a2aaf1d91e3fccf4ba8b05e",
    },
    "old_migration": {
        "plan_id": "issue85-dqn-old-migration",
        "agent": "DagobertDuckDQNTask2",
        "artifact": "checkpoint.pt",
        "sha256": "44cd337001b27b8596eaed985cfae1d7f30ecaf0b6b0328b35185395b7b81b6e",
    },
    "corrected_migration": {
        "plan_id": "issue85-dqn-corrected-migration",
        "agent": "DagobertDuckDQNTask2",
        "artifact": "checkpoint-issue85-zero-suffix.pt",
        "sha256": "3edb2e7196030fcb52af6c7dc9ee69d9fc1259898ea674002fe06fbe93468015",
    },
}
PRIMARY = "primary"
REPEAT = "deterministic-repeat"
DETERMINISTIC_COLUMNS = (
    "executed_action_sequence_sha256",
    "episode_steps",
    "survival_steps",
    "score",
    "coins_collected",
    "invalid_actions",
    "attempted_actions",
    "bombs_dropped",
    "survived",
    "termination_reason",
    "decision_time_median_ms",
    "decision_time_p95_ms",
    "decision_time_max_ms",
)
SUMMARY_COLUMNS = (
    "arm",
    "episodes",
    "mean_coins",
    "coins_standard_deviation",
    "mean_collection_fraction",
    "collection_fraction_standard_deviation",
    "invalid_action_rate",
    "bombs_dropped",
    "decision_time_median_ms",
    "decision_time_p95_ms",
    "decision_time_max_ms",
)


def analyze(plan_root: Path = PLAN_ROOT, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Reject incomplete/mismatched evidence and aggregate the primary pass."""
    rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    artifacts: dict[str, dict[str, str]] = {}
    deterministic = True
    for arm, expected in ARMS.items():
        primary, repeated, artifact = _load_arm(Path(plan_root), arm, expected)
        rows_by_arm[arm] = primary
        artifacts[arm] = artifact
        for left, right in zip(primary, repeated, strict=True):
            if any(left[field] != right[field] for field in DETERMINISTIC_COLUMNS):
                deterministic = False

    summaries = [_summary(arm, rows) for arm, rows in rows_by_arm.items()]
    result = {
        "schema_version": 1,
        "issue": 85,
        "primary_evaluation_episodes": sum(len(rows) for rows in rows_by_arm.values()),
        "repeat_evaluation_episodes": sum(len(rows) for rows in rows_by_arm.values()),
        "artifacts": artifacts,
        "deterministic_repeats": deterministic,
        "summaries": summaries,
    }
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "summary.csv", summaries)
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _load_arm(
    plan_root: Path, arm: str, expected: dict[str, str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    directory = plan_root / expected["plan_id"]
    status = _read_json(directory / "status.json")
    resolved = _read_json(directory / "resolved_plan.json")
    if status.get("status") != "completed":
        raise ValueError(f"Run plan is not completed: {expected['plan_id']}")
    if (
        resolved.get("agent") != expected["agent"]
        or resolved.get("artifact_path") != expected["artifact"]
    ):
        raise ValueError(f"Registered agent/artifact mismatch for {arm}")

    expected_jobs = [job for job in resolved.get("jobs", []) if job.get("kind") == "evaluation"]
    if len(expected_jobs) != 20:
        raise ValueError(f"Expected 20 registered evaluation jobs for {arm}")
    rows: dict[str, dict[str, Any]] = {}
    artifact_record: dict[str, str] | None = None
    for job in expected_jobs:
        job_id = job["run_id"]
        record = status.get("jobs", {}).get(job_id, {})
        attempts = record.get("attempts", [])
        if record.get("status") != "completed" or len(attempts) != 1:
            raise ValueError(f"Evaluation job is incomplete or rerun: {job_id}")
        artifact = record.get("artifact")
        if not isinstance(artifact, dict) or artifact.get("sha256") != expected["sha256"]:
            raise ValueError(f"Artifact checksum mismatch for {job_id}")
        selected = {"path": artifact.get("path"), "sha256": artifact.get("sha256")}
        if artifact_record is None:
            artifact_record = selected
        elif artifact_record != selected:
            raise ValueError(f"Artifact selection changed within {arm}")
        run_directory = directory / attempts[0]["output"]
        metadata = _read_json(run_directory / "metadata.json")
        for field in ("world_seed", "agent_seed", "scenario", "rounds"):
            if metadata.get(field) != job[field]:
                raise ValueError(f"Metadata mismatch for {job_id}: {field}")
        if metadata.get("mode") != "evaluation" or metadata.get("opponents") != job["opponents"]:
            raise ValueError(f"Evaluation conditions mismatch for {job_id}")
        run_plan = metadata.get("run_plan", {})
        if run_plan.get("evaluation_checkpoint") != expected["artifact"]:
            raise ValueError(f"Checkpoint selection mismatch for {job_id}")
        episode_rows = read_episodes_csv(run_directory / "episodes.csv")
        if len(episode_rows) != 1:
            raise ValueError(f"Expected one retained episode row for {job_id}")
        row = dict(episode_rows[0])
        if any(row.get(field) is None for field in DETERMINISTIC_COLUMNS):
            raise ValueError(f"Missing deterministic evidence for {job_id}")
        row["world_seed"] = job["world_seed"]
        row["agent_seed"] = job["agent_seed"]
        rows[job_id] = row

    primary = _suite_rows(rows, expected_jobs, PRIMARY)
    repeated = _suite_rows(rows, expected_jobs, REPEAT)
    for left, right in zip(primary, repeated, strict=True):
        if (left["world_seed"], left["agent_seed"]) != (right["world_seed"], right["agent_seed"]):
            raise ValueError(f"Repeat seed mismatch for {arm}")
    if artifact_record is None:
        raise ValueError(f"No artifact record for {arm}")
    return primary, repeated, artifact_record


def _suite_rows(
    rows: dict[str, dict[str, Any]], jobs: list[dict[str, Any]], suite: str
) -> list[dict[str, Any]]:
    selected = [job for job in jobs if job.get("stage_or_suite") == suite]
    if len(selected) != 10:
        raise ValueError(f"Expected ten {suite} jobs")
    return [rows[job["run_id"]] for job in selected]


def _summary(arm: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    coins = [row["coins_collected"] for row in rows]
    fractions = [row["coins_collected"] / row["initially_available_coins"] for row in rows]
    attempted = sum(row["attempted_actions"] for row in rows)
    return {
        "arm": arm,
        "episodes": len(rows),
        "mean_coins": fmean(coins),
        "coins_standard_deviation": stdev(coins),
        "mean_collection_fraction": fmean(fractions),
        "collection_fraction_standard_deviation": stdev(fractions),
        "invalid_action_rate": sum(row["invalid_actions"] for row in rows) / attempted
        if attempted
        else None,
        "bombs_dropped": sum(row["bombs_dropped"] for row in rows),
        "decision_time_median_ms": fmean(row["decision_time_median_ms"] for row in rows),
        "decision_time_p95_ms": max(row["decision_time_p95_ms"] for row in rows),
        "decision_time_max_ms": max(row["decision_time_max_ms"] for row in rows),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-root", type=Path, default=PLAN_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)
    try:
        result = analyze(arguments.plan_root, arguments.output)
    except Exception as error:
        print(f"Issue #85 analysis failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
