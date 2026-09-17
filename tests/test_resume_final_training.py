"""Recovery of the final-training run: stage order, audits and the detached orchestrator."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import psutil
import pytest

from scripts import resume_final_training as resume


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def complete(root, stage):
    write(root / f"{stage}-state.json", {"status": "completed", "active": []})


def finish_results(root, size=12):
    (root / resume.EVIDENCE).write_bytes(b"x" * size)
    write(root / f"{resume.EVIDENCE}.manifest.json", {"size_bytes": size})


@pytest.fixture
def root(tmp_path):
    write(tmp_path / "config.json", {"minimum_free_memory_bytes": 0})
    return tmp_path


def test_stages_are_resumed_in_the_order_the_chain_runs_them(root):
    assert resume.first_incomplete(root) == ("train", "training")
    complete(root, "training")
    assert resume.first_incomplete(root) == ("evaluate", "evaluation")
    complete(root, "evaluation")
    complete(root, "latency")
    assert resume.first_incomplete(root) == ("results", None)
    finish_results(root)
    assert resume.first_incomplete(root) is None


def test_results_are_complete_only_when_the_manifest_matches_the_archive(root):
    (root / resume.EVIDENCE).write_bytes(b"partial")
    assert not resume.stage_complete(root, "results", None)
    write(root / f"{resume.EVIDENCE}.manifest.json", {"size_bytes": 999})
    assert not resume.stage_complete(root, "results", None)
    finish_results(root)
    assert resume.stage_complete(root, "results", None)


def test_a_run_with_no_live_orchestrator_needs_recovery(root):
    write(root / "chain-state.json", {"status": "failed", "active_pid": None})
    verdict, _ = resume.status(root)
    assert verdict == "NEEDS_RECOVERY"


def test_a_finished_run_reports_completed(root):
    for stage in ("training", "evaluation", "latency"):
        complete(root, stage)
    finish_results(root)
    verdict, _ = resume.status(root)
    assert verdict == "COMPLETED"


def fake_runner(root, calls, fail_on=None):
    def run(command, cwd, env):
        mode = command[command.index("--root") - 1]
        calls.append({"mode": mode, "command": command, "env": env})
        if mode == fail_on:
            return SimpleNamespace(returncode=1)
        stage = dict(resume.STAGES)[mode]
        if stage is None:
            finish_results(root)
        else:
            complete(root, stage)
        return SimpleNamespace(returncode=0)

    return run


def test_the_orchestrator_carries_the_run_through_every_remaining_stage(root):
    complete(root, "training")
    calls = []
    assert resume.orchestrate(root, "final-training", runner=fake_runner(root, calls)) == 0

    assert [c["mode"] for c in calls] == ["evaluate", "latency", "results"]
    assert all(c["env"]["TASK4_AUTHORIZED"] == "yes" for c in calls)
    assert all(c["env"]["TASK4_PROFILE"] == "final-training" for c in calls)
    assert not any("--resume" in c["command"] for c in calls)
    assert calls[-1]["command"][-2:] == ["--output", str(root / resume.EVIDENCE)]
    record = json.loads((root / resume.RECOVERY_STATE).read_text(encoding="utf-8"))
    assert record["status"] == "completed"
    assert record["completed"] == ["train", "evaluate", "latency", "results"]


def test_an_existing_stage_is_resumed_and_a_failure_stops_the_orchestrator(root):
    complete(root, "training")
    write(root / "evaluation-state.json", {"status": "running", "active": []})
    calls = []
    assert resume.orchestrate(root, "final-training", runner=fake_runner(root, calls, "evaluate")) == 1

    assert [c["mode"] for c in calls] == ["evaluate"]
    assert "--resume" in calls[0]["command"]
    record = json.loads((root / resume.RECOVERY_STATE).read_text(encoding="utf-8"))
    assert record["status"] == "failed"


def test_a_partial_archive_is_set_aside_rather_than_blocking_the_results_stage(root):
    for stage in ("training", "evaluation", "latency"):
        complete(root, stage)
    (root / resume.EVIDENCE).write_bytes(b"truncated")
    calls = []
    assert resume.orchestrate(root, "final-training", runner=fake_runner(root, calls)) == 0
    assert list(root.glob(f"{resume.EVIDENCE}.partial-*"))


def test_the_audit_refuses_while_a_recorded_worker_is_alive(root):
    me = psutil.Process()
    write(
        root / "training-state.json",
        {"status": "running", "active": [{"pid": os.getpid(), "created": me.create_time()}]},
    )
    with pytest.raises(RuntimeError):
        resume.audit(root, "training")
    assert not list(root.glob("resume-audit-*.json"))


def test_the_audit_records_an_abandoned_roster_before_clearing_it(root):
    write(
        root / "training-state.json",
        {"status": "failed", "active": [{"pid": 999999999, "created": 0.0}], "cpu_seconds": 5.0},
    )
    audited = resume.audit(root, "training")
    record = json.loads(audited.read_text(encoding="utf-8"))
    assert record["unreaped_workers"] == [{"pid": 999999999, "created": 0.0}]
    assert record["accounting_carried_forward"]["cpu_seconds"] == 5.0
    state = json.loads((root / "training-state.json").read_text(encoding="utf-8"))
    assert state["active"] == [] and state["status"] == "running"


def test_the_process_sweep_matches_every_spelling_of_the_run_root():
    assert resume.normalized("C:/Task4-Final-Training/") == resume.normalized("c:\\task4-final-training")
