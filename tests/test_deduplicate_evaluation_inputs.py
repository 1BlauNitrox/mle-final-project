"""Recovery preserves contents/history and never touches live or failed snapshots."""

import json
import os
import shutil
import stat

import pytest

from scripts.deduplicate_evaluation_inputs import digest, recover


def fixture(root):
    plan = root / "plans/example"
    jobs, paths = {}, []
    for index, (kind, status) in enumerate(
        [
            ("evaluation", "completed"),
            ("evaluation", "completed"),
            ("evaluation", "running"),
            ("training", "completed"),
        ]
    ):
        path = plan / f"jobs/job{index}/attempt-001-input-agent/checkpoint.pt"
        path.parent.mkdir(parents=True)
        if paths:
            shutil.copy2(paths[0], path)
        else:
            path.write_bytes(b"checkpoint-content" * 1000)
        paths.append(path)
        jobs[f"job{index}"] = {
            "kind": kind,
            "status": status,
            "replica": "r1",
            "artifact": {
                "selection": "immutable evaluation input",
                "sha256": digest(path),
                "path": "artifacts/checkpoint.pt",
            },
            "attempts": [{"status": status, "output": f"jobs/job{index}/attempt-001"}],
        }
    for relative in ("artifacts/checkpoint.pt", "replicas/r1/agent/checkpoint.pt"):
        protected = plan / relative
        protected.parent.mkdir(parents=True)
        shutil.copy2(paths[0], protected)
    (plan / "status.json").write_text(json.dumps({"jobs": jobs}))
    (root / "resources.json").write_text('{"active_root_pids": []}')
    return paths


def test_dry_run_apply_preservation_and_idempotence(tmp_path):
    root = tmp_path / "runs"
    paths = fixture(root)
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    attributes = {
        p: (stat.S_IMODE(p.stat().st_mode), p.stat().st_mtime_ns)
        for p in before
    }
    result = recover(root)
    assert result["replacement_paths"] == 1
    assert result["estimated_reclaimed_bytes"] == paths[0].stat().st_size
    assert not os.path.samefile(*paths[:2])
    audit = tmp_path / "audit.jsonl"
    recover(root, audit=audit)
    assert os.path.samefile(*paths[:2])
    assert not os.path.samefile(paths[0], paths[2])
    assert not os.path.samefile(paths[0], paths[3])
    assert all(p.read_bytes() == data for p, data in before.items())
    assert all(
        (stat.S_IMODE(p.stat().st_mode), p.stat().st_mtime_ns) == value
        for p, value in attributes.items()
    )
    assert set(before) == {p for p in root.rglob("*") if p.is_file()}
    assert recover(root)["replacement_paths"] == 0
    assert json.loads(audit.read_text().splitlines()[-1])["phase"] == "complete"
    with pytest.raises(ValueError, match="previous audit"):
        recover(root, audit=audit)


def test_hash_mismatch_aborts_before_any_mutation(tmp_path):
    paths = fixture(tmp_path)
    paths[1].write_bytes(b"damaged checkpoint")
    with pytest.raises(ValueError, match="hash mismatch"):
        recover(tmp_path, audit=tmp_path / "new-audit")
    assert not (tmp_path / "new-audit").exists()
    assert not os.path.samefile(*paths[:2])
    assert not (tmp_path / ".task3-campaign.lock").exists()


def test_active_lock_and_workers_rejected(tmp_path):
    fixture(tmp_path)
    lock = tmp_path / ".task3-campaign.lock"
    lock.write_text("Live campaign")
    with pytest.raises(ValueError, match="lock exists"):
        recover(tmp_path)
    lock.unlink()
    (tmp_path / "resources.json").write_text('{"active_root_pids": [123]}')
    with pytest.raises(ValueError, match="active workers"):
        recover(tmp_path)


def test_path_escape_rejected(tmp_path):
    fixture(tmp_path)
    status = tmp_path / "plans/example/status.json"
    data = json.loads(status.read_text())
    data["jobs"]["job0"]["attempts"][0]["output"] = "../../outside/attempt-001"
    status.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Unexpected attempt path"):
        recover(tmp_path)


def test_link_failure_preserves_original_and_records_attempt(tmp_path, monkeypatch):
    paths = fixture(tmp_path)
    before = paths[1].read_bytes()

    def fail(*args):
        raise OSError("Simulated no free directory space")

    monkeypatch.setattr(os, "link", fail)
    audit = tmp_path / "audit.jsonl"
    with pytest.raises(OSError, match="directory space"):
        recover(tmp_path, audit=audit)
    assert paths[1].read_bytes() == before
    assert json.loads(audit.read_text().splitlines()[-1])["phase"] == "before"
    assert not (tmp_path / ".task3-campaign.lock").exists()


def test_damaged_live_workspace_blocks_recovery_before_changes(tmp_path):
    paths = fixture(tmp_path)
    live = tmp_path / "plans/example/replicas/r1/agent/checkpoint.pt"
    live.write_bytes(b"truncated workspace")
    with pytest.raises(ValueError, match="artifact/workspace hash mismatch"):
        recover(tmp_path, audit=tmp_path / "audit.jsonl")
    assert not os.path.samefile(*paths[:2])
    assert live.read_bytes() == b"truncated workspace"
