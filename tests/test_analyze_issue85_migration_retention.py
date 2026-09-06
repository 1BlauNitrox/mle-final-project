"""Contracts for Issue #85's registered evaluation analyzer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import training.analyze_issue85_migration_retention as issue85


def _episode_row() -> dict[str, object]:
    return {
        "executed_action_sequence_sha256": "a" * 64,
        "episode_steps": 10,
        "survival_steps": 10,
        "score": 1,
        "coins_collected": 1,
        "initially_available_coins": 2,
        "invalid_actions": 0,
        "attempted_actions": 10,
        "bombs_dropped": 0,
        "survived": True,
        "termination_reason": "survived",
        "decision_time_median_ms": 1.0,
        "decision_time_p95_ms": 2.0,
        "decision_time_max_ms": 3.0,
    }


def _write_registered_evidence(root: Path) -> None:
    for arm, expected in issue85.ARMS.items():
        directory = root / expected["plan_id"]
        replica = arm.replace("_", "-")
        jobs: list[dict[str, object]] = []
        status_jobs: dict[str, object] = {}
        for suite in (issue85.PRIMARY, issue85.REPEAT):
            for index in range(10):
                job_id = f"{arm}-{suite}-{index}"
                jobs.append(
                    {
                        "run_id": job_id,
                        "kind": "evaluation",
                        "stage_or_suite": suite,
                        "world_seed": 36001 + index,
                        "agent_seed": 46001 + index,
                        "scenario": "coin-heaven",
                        "rounds": 1,
                        "opponents": [],
                    }
                )
                output = f"jobs/{job_id}/attempt-001"
                status_jobs[job_id] = {
                    "status": "completed",
                    "artifact": {
                        "path": f"replicas/{replica}/agent/{expected['artifact']}",
                        "sha256": expected["sha256"],
                    },
                    "attempts": [{"output": output}],
                }
                run = directory / output
                run.mkdir(parents=True)
                (run / "metadata.json").write_text(
                    json.dumps(
                        {
                            "world_seed": 36001 + index,
                            "agent_seed": 46001 + index,
                            "scenario": "coin-heaven",
                            "rounds": 1,
                            "mode": "evaluation",
                            "opponents": [],
                            "run_plan": {"evaluation_checkpoint": expected["artifact"]},
                        }
                    ),
                    encoding="utf-8",
                )
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "status.json").write_text(
            json.dumps({"status": "completed", "jobs": status_jobs}), encoding="utf-8"
        )
        (directory / "resolved_plan.json").write_text(
            json.dumps(
                {"agent": expected["agent"], "artifact_path": expected["artifact"], "jobs": jobs}
            ),
            encoding="utf-8",
        )


def test_analyzer_requires_registered_artifacts_and_checkpoint_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_registered_evidence(tmp_path)
    monkeypatch.setattr(issue85, "read_episodes_csv", lambda _: [_episode_row()])

    result = issue85.analyze(tmp_path, tmp_path / "analysis")

    assert result["primary_evaluation_episodes"] == 30
    assert result["repeat_evaluation_episodes"] == 30
    assert result["deterministic_repeats"] is True
    assert (tmp_path / "analysis" / "summary.csv").is_file()

    path = tmp_path / issue85.ARMS["corrected_migration"]["plan_id"] / "status.json"
    status = json.loads(path.read_text(encoding="utf-8"))
    first_job = next(iter(status["jobs"].values()))
    first_job["artifact"]["sha256"] = "changed"
    path.write_text(json.dumps(status), encoding="utf-8")

    with pytest.raises(ValueError, match="Artifact checksum mismatch"):
        issue85.analyze(tmp_path, tmp_path / "analysis-rejected")
