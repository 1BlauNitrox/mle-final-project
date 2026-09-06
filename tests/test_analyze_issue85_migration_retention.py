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
        "coins_found": 0,
        "crates_destroyed": 0,
        "invalid_actions": 0,
        "attempted_actions": 10,
        "bombs_dropped": 0,
        "self_kills": 0,
        "survived": True,
        "termination_reason": "survived",
        "action_up": 2,
        "action_right": 2,
        "action_down": 2,
        "action_left": 2,
        "action_wait": 2,
        "action_bomb": 0,
        "action_unknown": 0,
        "decision_time_median_ms": 1.0,
        "decision_time_p95_ms": 2.0,
        "decision_time_max_ms": 3.0,
    }


def _write_registered_evidence(root: Path) -> None:
    for arm, expected in issue85.ARMS.items():
        directory = root / expected["plan_id"]
        registered = issue85.load_plan(issue85.RUN_PLAN_PATHS[arm]).to_dict()
        jobs = registered["jobs"]
        status_jobs: dict[str, object] = {}
        for job in jobs:
            job_id = job["run_id"]
            output = f"jobs/{job_id}/attempt-001"
            status_jobs[job_id] = {
                "status": "completed",
                "artifact": {
                    "path": f"replicas/{job['replica']}/agent/{expected['artifact']}",
                    "sha256": expected["sha256"],
                },
                "attempts": [{"output": output}],
            }
            run = directory / output
            run.mkdir(parents=True)
            (run / "metadata.json").write_text(
                json.dumps(
                    {
                        "world_seed": job["world_seed"],
                        "agent_seed": job["agent_seed"],
                        "scenario": job["scenario"],
                        "rounds": job["rounds"],
                        "mode": "evaluation",
                        "opponents": job["opponents"],
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
            json.dumps(registered),
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


def test_analyzer_allows_timing_variation_but_rejects_changed_action_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_registered_evidence(tmp_path)

    def timing_variation(path: Path) -> list[dict[str, object]]:
        row = _episode_row()
        if issue85.REPEAT in str(path):
            row["decision_time_p95_ms"] = 99.0
        return [row]

    monkeypatch.setattr(issue85, "read_episodes_csv", timing_variation)
    issue85.analyze(tmp_path, tmp_path / "timing-variation")

    def changed_action(path: Path) -> list[dict[str, object]]:
        row = timing_variation(path)
        if issue85.REPEAT in str(path):
            row[0]["executed_action_sequence_sha256"] = "b" * 64
        return row

    monkeypatch.setattr(issue85, "read_episodes_csv", changed_action)
    with pytest.raises(ValueError, match="Deterministic repeat mismatch"):
        issue85.analyze(tmp_path, tmp_path / "changed-action")
    assert not (tmp_path / "changed-action").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("world_seed", 999),
        ("scenario", "classic"),
        ("population", "confirmation"),
        ("stage_or_suite", "unregistered-suite"),
    ),
)
def test_analyzer_rejects_unregistered_job_matrix(
    field: str, value: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_registered_evidence(tmp_path)
    monkeypatch.setattr(issue85, "read_episodes_csv", lambda _: [_episode_row()])
    directory = tmp_path / issue85.ARMS["old_migration"]["plan_id"]
    resolved_path = directory / "resolved_plan.json"
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    resolved["jobs"][0][field] = value
    resolved_path.write_text(json.dumps(resolved), encoding="utf-8")

    with pytest.raises(ValueError, match="Registered job matrix mismatch"):
        issue85.analyze(tmp_path, tmp_path / "unregistered-job")


def test_analyzer_rejects_changed_configuration_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_registered_evidence(tmp_path)
    monkeypatch.setattr(issue85, "read_episodes_csv", lambda _: [_episode_row()])
    path = tmp_path / issue85.ARMS["old_migration"]["plan_id"] / "resolved_plan.json"
    resolved = json.loads(path.read_text(encoding="utf-8"))
    resolved["fingerprints"]["configuration"] = "changed"
    path.write_text(json.dumps(resolved), encoding="utf-8")

    with pytest.raises(ValueError, match="configuration fingerprint mismatch"):
        issue85.analyze(tmp_path, tmp_path / "changed-configuration")
