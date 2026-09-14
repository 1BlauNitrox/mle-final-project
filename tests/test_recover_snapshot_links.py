import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

from scripts import recover_snapshot_links as recovery


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "quiescent", lambda root: None)
    root = tmp_path / "campaign"
    plan = root / "runs/plans/example"
    dump(plan / "status.json", {"jobs": {"eval-a": {"status": "pending", "attempts": []}}})
    dump(plan / "resolved_plan.json", {"jobs": [{"run_id": "eval-a", "kind": "evaluation"}]})
    dump(root / "runs/resources.json", {"active_root_pids": [], "cpu_seconds_consumed": 19})
    dump(root / "supervisor-resources.json", {"active_root_pids": []})
    dump(root / "runs/authorization.json", {"unchanged": True})
    checkpoint = plan / "replicas/r1/agent/checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"original trained checkpoint")
    (root / "supervisor.err.log").write_text("original WinError 1142", encoding="utf-8")
    data = b"immutable feature code"
    digest = hashlib.sha256(data).hexdigest()
    obj = root / "runs/input-objects" / digest
    obj.parent.mkdir(parents=True)
    obj.write_bytes(data)
    retained = root / "retained"
    retained.mkdir()
    for n in range(935):
        os.link(obj, retained / str(n))
    obj.chmod(stat.S_IREAD)
    partial = plan / "jobs/eval-a/attempt-001-input-agent.tmp"
    partial.mkdir(parents=True)
    os.link(obj, partial / "features.py")
    return root, obj, checkpoint, partial


def test_read_only_audit(campaign):
    root, obj, checkpoint, partial = campaign
    before = obj.stat().st_ino
    result = recovery.recover(root, root / "audit")
    assert result["objects"][0]["rotate"]
    assert obj.stat().st_ino == before
    assert partial.exists() and checkpoint.read_bytes() == b"original trained checkpoint"
    assert not (root / "audit").exists()


def test_real_links_rotate_preserving_old_bytes_and_pending_record(campaign):
    root, obj, checkpoint, partial = campaign
    inode = obj.stat().st_ino
    result = recovery.recover(root, root / "audit", True)
    assert result["rotated_objects"] == 1
    assert obj.stat().st_ino != inode
    assert (root / "retained/0").stat().st_ino == inode
    assert (root / "retained/0").read_bytes() == obj.read_bytes()
    assert (root / "audit" / obj.name).stat().st_ino == inode
    assert checkpoint.read_bytes() == b"original trained checkpoint"
    assert not partial.exists()
    assert partial.with_name(partial.name + ".failed-storage").exists()
    for n in range(100):
        os.link(obj, root / "retained" / f"new-{n}")
    before = json.loads((root / "audit/before.json").read_text())
    after = json.loads((root / "audit/after.json").read_text())
    assert before["protected"] == after["protected"]


def test_corrupt_object_fails_before_apply(campaign):
    root, obj, _, _ = campaign
    obj.chmod(stat.S_IREAD | stat.S_IWRITE)
    obj.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Invalid content"):
        recovery.recover(root, root / "audit", True)
    assert not (root / "audit").exists()


def test_interrupted_install_restores_original_object(campaign, monkeypatch):
    root, obj, _, partial = campaign
    original = Path.rename

    def fail_new(self, target):
        if self.name.endswith(".new"):
            raise OSError("simulated install failure")
        return original(self, target)

    monkeypatch.setattr(Path, "rename", fail_new)
    inode = obj.stat().st_ino
    with pytest.raises(OSError, match="simulated"):
        recovery.recover(root, root / "audit", True)
    assert obj.stat().st_ino == inode and partial.exists()
    assert "rotate_rolled_back" in (root / "audit/operations.jsonl").read_text()
    assert list((root / "audit").glob("*.new"))


def test_rejects_training_and_existing_audit(campaign):
    root, _, _, _ = campaign
    (root / "audit").mkdir()
    with pytest.raises(ValueError, match="fresh audit"):
        recovery.recover(root, root / "audit", True)
    dump(
        root / "runs/plans/example/resolved_plan.json",
        {"jobs": [{"run_id": "eval-a", "kind": "training"}]},
    )
    with pytest.raises(ValueError, match="only pending evaluation"):
        recovery.inventory(root)


def test_active_lock_rejected(tmp_path):
    (tmp_path / ".supervisor.lock").touch()
    with pytest.raises(ValueError, match="lock"):
        recovery.quiescent(tmp_path)


def test_escape_rejected(campaign):
    root, _, _, _ = campaign
    with pytest.raises(ValueError, match="escapes"):
        recovery.recover(root, root.parent / "outside", True)


def test_unknown_staging_is_preserved(campaign):
    root, _, _, partial = campaign
    unknown = partial.with_name("unrecognized.tmp")
    unknown.mkdir()
    with pytest.raises(ValueError, match="Unexpected incomplete"):
        recovery.inventory(root)
    assert unknown.exists()
