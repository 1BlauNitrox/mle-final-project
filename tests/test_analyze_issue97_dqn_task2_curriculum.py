"""Tests for Issue #97's curriculum analysis, focused on its direct-arm-only
degradation path (the staged arm's raw evidence is an external archive not
always available locally -- see STAGED_ARM_MISSING_NOTE)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from training.analyze_issue97_dqn_task2_curriculum import (
    DIRECT_PLAN_ID,
    _summaries,
    rows_from_evidence,
)
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


def _write_evaluation_job(
    plan_directory: Path,
    *,
    run_id: str,
    stage_or_suite: str,
    world_seed: int,
) -> None:
    run_directory = plan_directory / "jobs" / run_id / "attempt-001"
    run_directory.mkdir(parents=True)
    with (run_directory / "episodes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(EPISODE_HEADER)
        writer.writerow(
            [1, 1, "agent", "evaluation", 5, 5, 1, 1, 9, 0, 0, 0, 0, 0, 5, 0.0, "True",
             "step_limit", "deadbeef", 1, 1, 1, 1, 1, 0, 0, 0.5, 0.6, 0.7]
        )
    (run_directory / "metadata.json").write_text(
        json.dumps({"git_commit": "abc123", "git_dirty": False, "started_at": "now"}),
        encoding="utf-8",
    )

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
            "plan_id": DIRECT_PLAN_ID,
            "action_masking": "none",
            "reward_variant": None,
            "fingerprints": {"configuration": "fp"},
            "jobs": [],
        }
    )
    resolved["jobs"].append(
        {
            "run_id": run_id,
            "kind": "evaluation",
            "replica": "r1",
            "stage_or_suite": stage_or_suite,
            "world_seed": world_seed,
            "agent_seed": world_seed + 10000,
        }
    )
    resolved_path.write_text(json.dumps(resolved), encoding="utf-8")


def _export_minimal_direct_arm_evidence(tmp_path: Path) -> Path:
    plan_directory = tmp_path / "plan"
    _write_evaluation_job(
        plan_directory,
        run_id="eval-r1-classic-primary-seed-001",
        stage_or_suite="classic-primary",
        world_seed=71001,
    )
    _write_evaluation_job(
        plan_directory,
        run_id="eval-r1-classic-repeat-seed-001",
        stage_or_suite="classic-repeat",
        world_seed=71001,
    )
    evidence_root = tmp_path / "evidence"
    export_plan_evidence(plan_directory, evidence_root / DIRECT_PLAN_ID)
    return evidence_root


def _row(**overrides: object) -> dict[str, object]:
    base = {
        "scenario": "classic",
        "arm": "direct",
        "attempted_actions": 10,
        "coins_collected": 1,
        "collection_fraction": 0.1,
        "survival_steps": 20,
        "survived": True,
        "self_kills": 0,
        "invalid_actions": 1,
        "crates_destroyed": 2,
        "action_up": 1,
        "action_right": 1,
        "action_down": 1,
        "action_left": 1,
        "action_wait": 1,
        "action_bomb": 1,
        "action_unknown": 0,
        "decision_time_median_ms": 0.5,
        "decision_time_p95_ms": 1.0,
        "decision_time_max_ms": 2.0,
    }
    return {**base, **overrides}


def test_summaries_skip_the_absent_staged_arm_without_error() -> None:
    """With only "direct" rows present, the missing "staged" group is skipped
    rather than raising or producing an empty/NaN entry -- this is exactly
    what happens when the staged arm's raw archive is not available locally.
    """
    rows = [_row(), _row(scenario="coin-heaven"), _row(scenario="loot-crate")]

    summaries = _summaries(rows)

    assert {entry["scenario"] for entry in summaries} == {"classic", "coin-heaven", "loot-crate"}
    assert all(entry["arm"] == "direct" for entry in summaries)


def test_summaries_include_both_arms_once_staged_rows_exist() -> None:
    rows = [_row(), _row(arm="staged")]

    summaries = _summaries(rows)

    assert {entry["arm"] for entry in summaries} == {"direct", "staged"}


def test_rows_from_evidence_reads_freshly_exported_evidence(tmp_path: Path) -> None:
    evidence_root = _export_minimal_direct_arm_evidence(tmp_path)

    rows, deterministic_repeats, staged_arm_available = rows_from_evidence(evidence_root)

    assert len(rows) == 1
    assert rows[0]["scenario"] == "classic"
    assert rows[0]["arm"] == "direct"
    assert deterministic_repeats is True
    assert staged_arm_available is False


def test_rows_from_evidence_rejects_evidence_that_no_longer_matches_its_manifest(
    tmp_path: Path,
) -> None:
    """Regression test: a manifest fingerprinted before Git's own eol=lf
    normalization must not be trusted silently -- rows_from_evidence has to
    detect a CSV that no longer matches its manifest record, not just read it.
    """
    evidence_root = _export_minimal_direct_arm_evidence(tmp_path)
    (evidence_root / DIRECT_PLAN_ID / "evaluation-episodes.csv").write_bytes(b"tampered\n")

    try:
        rows_from_evidence(evidence_root)
        raise AssertionError("expected ValueError for a manifest/evidence mismatch")
    except ValueError as error:
        assert "does not match its manifest record" in str(error)
