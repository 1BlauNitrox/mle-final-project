"""Tests for exporting committable evidence from a completed run plan."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

from training.export_evidence import export_plan_evidence

EPISODE_HEADER = [
    "schema_version",
    "round",
    "agent",
    "mode",
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
    "invalid_action_rate",
    "survived",
    "termination_reason",
    "executed_action_sequence_sha256",
    "action_up",
    "action_right",
    "action_down",
    "action_left",
    "action_wait",
    "action_bomb",
    "action_unknown",
    "decision_time_median_ms",
    "decision_time_p95_ms",
    "decision_time_max_ms",
]


def _write_job(
    plan_directory: Path,
    *,
    run_id: str,
    kind: str,
    replica: str,
    stage_or_suite: str,
    world_seed: int,
    agent_seed: int,
    rows: int,
) -> None:
    run_directory = plan_directory / "jobs" / run_id / "attempt-001"
    run_directory.mkdir(parents=True)

    with (run_directory / "episodes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(EPISODE_HEADER)
        for round_number in range(1, rows + 1):
            writer.writerow(
                [
                    1,
                    round_number,
                    "agent",
                    kind,
                    5,
                    5,
                    1,
                    1,
                    9,
                    0,
                    0,
                    0,
                    0,
                    0,
                    5,
                    0.0,
                    "True",
                    "step_limit",
                    "deadbeef",
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    0.5,
                    0.6,
                    0.7,
                ]
            )

    (run_directory / "metadata.json").write_text(
        json.dumps({"git_commit": "abc123", "git_dirty": False, "started_at": "now"}),
        encoding="utf-8",
    )

    plan_directory.mkdir(parents=True, exist_ok=True)
    status_path = plan_directory / "status.json"
    status = json.loads(status_path.read_text()) if status_path.is_file() else {"jobs": {}}
    status["status"] = "completed"
    status["jobs"][run_id] = {
        "status": "completed",
        "attempts": [{"output": f"jobs/{run_id}/attempt-001"}],
    }
    status_path.write_text(json.dumps(status), encoding="utf-8")

    resolved_path = plan_directory / "resolved_plan.json"
    resolved = (
        json.loads(resolved_path.read_text())
        if resolved_path.is_file()
        else {
            "plan_id": "test-plan",
            "action_masking": "none",
            "reward_variant": "control",
            "fingerprints": {"configuration": "fp"},
            "jobs": [],
        }
    )
    resolved["jobs"].append(
        {
            "run_id": run_id,
            "kind": kind,
            "replica": replica,
            "stage_or_suite": stage_or_suite,
            "world_seed": world_seed,
            "agent_seed": agent_seed,
        }
    )
    resolved_path.write_text(json.dumps(resolved), encoding="utf-8")


def test_export_produces_tagged_training_and_evaluation_csvs_and_a_manifest(
    tmp_path: Path,
) -> None:
    plan_directory = tmp_path / "plan"
    _write_job(
        plan_directory,
        run_id="train-r1",
        kind="training",
        replica="r1",
        stage_or_suite="classic-direct",
        world_seed=1,
        agent_seed=2,
        rows=3,
    )
    _write_job(
        plan_directory,
        run_id="eval-r1-classic-primary-seed-001",
        kind="evaluation",
        replica="r1",
        stage_or_suite="classic-primary",
        world_seed=71001,
        agent_seed=81001,
        rows=1,
    )

    output_directory = tmp_path / "evidence"
    manifest = export_plan_evidence(plan_directory, output_directory)

    assert manifest["plan_id"] == "test-plan"
    assert manifest["reward_variant"] == "control"
    assert len(manifest["jobs"]) == 2
    assert manifest["evidence_files"]["training-episodes.csv.gz"]["size_bytes"] > 0

    with gzip.open(output_directory / "training-episodes.csv.gz", "rt", newline="") as handle:
        training_rows = list(csv.DictReader(handle))
    assert len(training_rows) == 3
    assert training_rows[0]["replica"] == "r1"
    assert training_rows[0]["stage_or_suite"] == "classic-direct"
    assert training_rows[0]["world_seed"] == "1"

    evaluation_path = output_directory / "evaluation-episodes.csv"
    assert b"\r\n" not in evaluation_path.read_bytes()
    with evaluation_path.open(encoding="utf-8", newline="") as handle:
        evaluation_rows = list(csv.DictReader(handle))
    assert len(evaluation_rows) == 1
    assert evaluation_rows[0]["stage_or_suite"] == "classic-primary"
    assert evaluation_rows[0]["world_seed"] == "71001"
    evaluation_record = manifest["evidence_files"]["evaluation-episodes.csv"]
    evaluation_bytes = evaluation_path.read_bytes()
    assert evaluation_record == {
        "sha256": hashlib.sha256(evaluation_bytes).hexdigest(),
        "size_bytes": len(evaluation_bytes),
    }


def test_export_rejects_an_incomplete_plan(tmp_path: Path) -> None:
    plan_directory = tmp_path / "plan"
    plan_directory.mkdir()
    (plan_directory / "status.json").write_text(json.dumps({"status": "running", "jobs": {}}))
    (plan_directory / "resolved_plan.json").write_text(
        json.dumps({"plan_id": "p", "fingerprints": {}, "jobs": []})
    )

    try:
        export_plan_evidence(plan_directory, tmp_path / "evidence")
        raise AssertionError("expected ValueError for an incomplete plan")
    except ValueError as error:
        assert "not completed" in str(error)
