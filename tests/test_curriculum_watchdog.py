"""Recovery must preserve the experiment, budget, evidence and user ownership."""

import gzip
import hashlib
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import psutil
import pytest

from scripts import watch_hunting_curriculum as w


def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    runner = tmp_path / "tools/scripts/runner.py"
    runner.parent.mkdir(parents=True)
    runner.write_text("# immutable runner", encoding="utf-8")
    config = put(root, "config.json", {
        "devices": {"pc": {"cpu_seconds": 1000}},
        "limits": {"stop_utc": "2050-01-01T00:00:00+00:00", "wall_seconds": 1000},
    })
    put(root, "binding.json", {"device": "pc", "config_sha256": w.digest(config),
                              "tools": {"scripts/runner.py": w.digest(runner)}, "inputs": {}})
    put(root, "smoke.json", {"passed": True})
    put(root, "resources.json", {"cpu_seconds": 100, "first_start": time.time() - 10})
    return root, runner


@pytest.mark.parametrize("reason", ["CPU budget exhausted", "Elapsed budget exhausted",
    "Sunday absolute stop reached", "Memory limit reached", "Corrupt generation",
    "User stop requested", "Pilot safety failed", "Unexpected model shape"])
def test_hard_stops_never_recover(prepared, reason):
    root, _ = prepared
    put(root, "STOP.json", {"reason": reason})
    assert w.recovery_reason(root) is None


def test_only_sharing_failure_retries(prepared):
    root, _ = prepared
    put(root, "STOP.json", {"reason": "Worker failed: example; inspect its log"})
    (root / "example.log").write_text(
        "Traceback\nPermissionError: [WinError 32] File in use\n", encoding="utf-8")
    assert w.recovery_reason(root)
    (root / "STOP.request").touch()
    assert w.recovery_reason(root) is None


def test_integrity_and_generation(prepared):
    root, runner = prepared
    checkpoint = put(root, "pairs/r1/control/generation-1/checkpoint.pt", {"weights": 1})
    state = put(root, "pairs/r1/control/generation-1/state.json", {"episodes": 10})
    put(root, "pairs/r1/control/resume.json", {"generation": "generation-1",
        "checkpoint_sha256": w.digest(checkpoint), "state_sha256": w.digest(state)})
    w.validate(root, runner)
    state.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Corrupt recovery"):
        w.validate(root, runner)


def test_config_change_blocks(prepared):
    root, runner = prepared
    put(root, "config.json", {})
    with pytest.raises(ValueError, match="Integrity failure"):
        w.validate(root, runner)


def test_budget_counts_overhead_and_original_elapsed(prepared):
    root, _ = prepared
    assert w.budget_reason(root, 901).startswith("CPU")
    usage = w.read(root / "resources.json")
    usage["first_start"] -= 1000
    put(root, "resources.json", usage)
    assert w.budget_reason(root).startswith("Elapsed")


def test_pid_reuse_does_not_own_process():
    own = psutil.Process()
    assert w.process({"pid": own.pid, "created": own.create_time()}) is not None
    assert w.process({"pid": own.pid, "created": own.create_time() - 1}) is None


def test_stall_requires_missing_supervisor_heartbeat(prepared):
    root, _ = prepared
    own = psutil.Process()
    now = time.time()
    assert not w.stale_heartbeat(root, own, now)
    assert w.stale_heartbeat(root, own, now + 301)


def test_dead_process_during_observation_is_not_watcher_failure():
    class Dead:
        def children(self, recursive):
            raise psutil.NoSuchProcess(99999999)

    assert w.observe_family(Dead()) == []


def test_duplicate_watchdog_lock(prepared):
    root, _ = prepared
    with w.exclusive(root), pytest.raises(OSError), w.exclusive(root):
        pytest.fail("Second watcher acquired lock")


def test_restart_archives_stop_preserves_budget_and_limits(prepared, monkeypatch):
    root, runner = prepared
    put(root, "supervisor.json", {"pid": 99999999, "created": 1})
    put(root, "STOP.json", {"reason": "[WinError 32] Sharing violation"})
    original = w.read(root / "resources.json")
    calls = []

    def fake_launch(*args):
        calls.append(args)
        # An unrelated fatal failure must end the next iteration, not be retried.
        put(root, "STOP.json", {"reason": "Unknown fatal failure"})
        return {"pid": 99999999, "created": 1}

    monkeypatch.setattr(w, "launch", fake_launch)
    result = w.watch(root, runner, Path(sys.executable), interval=0)
    assert len(calls) == 1
    assert result["restarts"] == 1
    usage = w.read(root / "resources.json")
    assert usage["cpu_seconds"] >= original["cpu_seconds"] + 60
    assert usage["first_start"] == original["first_start"]
    assert list(root.glob("watchdog-recovery-*.STOP.json"))
    assert w.read(root / "STOP.json")["reason"] == "Unknown fatal failure"


def test_startup_crash_does_not_retry_forever(prepared, monkeypatch):
    root, runner = prepared
    monkeypatch.setattr(w, "launch", lambda *args: {"pid": 99999999, "created": 1})
    result = w.watch(root, runner, Path(sys.executable), interval=0)
    assert result["launches"] == 4
    assert result["status"] == "stopped"


def test_orphan_cleanup_ignores_wrong_creation_time(prepared):
    root, _ = prepared
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        created = psutil.Process(child.pid).create_time()
        w.stop_orphans(root, [{"pid": child.pid, "created": created - 1}])
        assert child.poll() is None
        w.stop_orphans(root, [{"pid": child.pid, "created": created}])
        child.wait(timeout=5)
        assert child.returncode is not None
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_result_commit_excludes_models_prose_and_unrelated_work(prepared, tmp_path):
    root, _ = prepared
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()

    git("init", "-b", "experiment/217-test-results")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    (repo / "README.md").write_text("human draft", encoding="utf-8")
    files = {"pairs/r1/decision.json": b'{"status":"stopped_safety"}',
             "pairs/r1/large-state.json": json.dumps({"rows": ["example"] * 20000}).encode(),
             "pairs/r1/checkpoint.pt": b"MODEL MUST NOT BE COMMITTED"}
    archive = root / "result.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        for name, content in files.items():
            stream.writestr(name, content)
        stream.writestr("manifest.json", json.dumps({name: {
            "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)
        } for name, content in files.items()}))
    put(root, "export.json", {"path": str(archive), "sha256": w.digest(archive),
                              "bytes": archive.stat().st_size})
    put(root, "complete.json", {"pipeline_complete": True})
    commit = w.evidence(root, repo)
    assert len(commit) == 40
    tracked = git("ls-files")
    assert "decision.json" in tracked
    assert "README" not in tracked
    assert "checkpoint" not in tracked
    assert "zip" not in tracked
    assert "large-state.json.gz" in tracked
    stored = repo / "experiments/2026-09-19-hunting-curriculum/evidence/pc"
    assert gzip.decompress((stored / "pairs/r1/large-state.json.gz").read_bytes()) == (
        files["pairs/r1/large-state.json"])
    assert w.read(stored / "evidence-index.json")["pairs/r1/large-state.json"]["sha256"] == (
        hashlib.sha256(files["pairs/r1/large-state.json"]).hexdigest())
