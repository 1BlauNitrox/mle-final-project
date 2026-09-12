"""Storage limits cover the matrix and fail before exhausting evidence storage."""

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from training import issue150_storage as storage
from training import run_plan
from training import task3_double_campaign as campaign
from training.run_issue107_campaign import CampaignLimits, CampaignResourceLimitExceeded


def plans():
    return {
        "arm": SimpleNamespace(
            agent="example",
            replicas=(),
            jobs=tuple(SimpleNamespace(kind="training") for _ in range(10))
            + tuple(SimpleNamespace(kind="evaluation") for _ in range(3520)),
        )
    }


def monitor(tmp_path, monkeypatch, **kwargs):
    monkeypatch.setattr(
        storage.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * storage.GIB)
    )
    return storage.StorageMonitor(
        source_root=tmp_path,
        state_path=tmp_path / "resources.json",
        authorized_at="2026-09-12T11:21:26+00:00",
        limits=CampaignLimits(86400, 36000, 8 * storage.GIB),
        time_fn=lambda: 1789212086.0,
        **kwargs,
    )


def test_budget_counts_all_copies_and_export():
    result = storage.storage_budget(plans())
    assert result["training_jobs"] == 10
    assert result["evaluation_jobs"] == 3520
    assert result["retained_snapshot_bytes"] == 3530 * 2 * storage.SNAPSHOT_BYTES
    assert result["export_analysis_bytes"] == 2 * (result["record_bytes"] + storage.WORKSPACE_BYTES)
    assert result["required_free_bytes"] == 91 * storage.GIB
    assert campaign.validate()[2]["storage"] == result


@pytest.mark.parametrize("free", [8 * storage.GIB, int(8.9 * storage.GIB), 77 * storage.GIB])
def test_insufficient_output_disk_is_read_only(tmp_path, monkeypatch, free):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    observed = []

    def disk(path):
        observed.append(path)
        return SimpleNamespace(free=free if path == target else 100 * storage.GIB)

    monkeypatch.setattr(storage.shutil, "disk_usage", disk)
    with pytest.raises(ValueError, match="8 GiB is insufficient"):
        storage.preflight(plans(), target / "not-created", source)
    assert observed == [target]
    assert list(target.iterdir()) == []


def test_sufficient_disk_and_separate_source_floor(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    monkeypatch.setattr(
        storage.shutil, "disk_usage", lambda _: SimpleNamespace(free=91 * storage.GIB)
    )
    result = storage.preflight(plans(), output, source)
    assert result["observed_free_bytes"] == result["required_free_bytes"]
    monkeypatch.setattr(
        storage.shutil,
        "disk_usage",
        lambda p: SimpleNamespace(free=5 * storage.GIB if p == source else 91 * storage.GIB),
    )
    with pytest.raises(ValueError, match="6 GiB"):
        storage.preflight(plans(), output, source)


def test_storage_breach_is_persistent_and_stops_process_forest(tmp_path, monkeypatch):
    guard = monitor(tmp_path, monkeypatch)
    stopped = []
    monkeypatch.setattr(guard, "_terminate", lambda processes: stopped.append(processes))
    monkeypatch.setattr(
        storage.shutil, "disk_usage", lambda _: SimpleNamespace(free=3 * storage.GIB)
    )
    with pytest.raises(CampaignResourceLimitExceeded, match="Storage reserve"):
        guard.check()
    record = json.loads((tmp_path / "resources.json").read_text())
    assert "Storage reserve" in record["limit_reached"]
    assert len(stopped) == 1
    monkeypatch.setattr(
        storage.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * storage.GIB)
    )
    with pytest.raises(CampaignResourceLimitExceeded, match="Storage reserve"):
        guard.check()


@pytest.mark.parametrize(
    "kind,limit", [("evaluation", storage.EVALUATION_BYTES), ("training", storage.TRAINING_BYTES)]
)
def test_attempt_output_guard_preserves_bytes(tmp_path, monkeypatch, kind, limit):
    guard = monitor(tmp_path, monkeypatch)
    attempt = tmp_path / "jobs/j/attempt-001"
    attempt.mkdir(parents=True)
    path = attempt / "framework_stats.json"
    with path.open("wb") as file:
        file.truncate(limit + 1)
    job = SimpleNamespace(run_id="j", kind=kind)
    with pytest.raises(CampaignResourceLimitExceeded, match="output exceeds"):
        guard.begin_job(job, tmp_path / "alias", attempt)
    assert path.stat().st_size == limit + 1


def test_oversized_checkpoint_stops_before_next_snapshot(tmp_path, monkeypatch):
    guard = monitor(tmp_path, monkeypatch)
    alias = tmp_path / "alias"
    alias.mkdir()
    path = alias / "checkpoint.pt"
    with path.open("wb") as file:
        file.truncate(storage.SNAPSHOT_BYTES + 1)
    with pytest.raises(CampaignResourceLimitExceeded, match="workspace exceeds"):
        guard.begin_job(
            SimpleNamespace(run_id="j", kind="evaluation"), alias, tmp_path / "jobs/j/attempt-001"
        )
    assert not (tmp_path / "jobs").exists()


def test_completed_job_is_not_recharged(tmp_path, monkeypatch):
    guard = monitor(tmp_path, monkeypatch)
    job = SimpleNamespace(run_id="j", kind="evaluation")
    guard.begin_job(job, tmp_path / "alias", tmp_path / "attempt")
    guard.end_job(job)
    assert guard.storage_jobs == {}


def test_equivalent_timestamp_resume_preserves_usage_and_limits(tmp_path, monkeypatch):
    guard = monitor(tmp_path, monkeypatch)
    guard.check()
    path = tmp_path / "resources.json"
    record = json.loads(path.read_text())
    record["cpu_seconds_consumed"] = 17
    path.write_text(json.dumps(record))
    before = path.read_bytes()
    restored = monitor(tmp_path, monkeypatch)
    assert restored._completed_cpu_seconds == 17
    assert path.read_bytes() == before
    record["limits"]["wall_seconds"] += 1
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="different limits"):
        monitor(tmp_path, monkeypatch)


def test_cli_run_cannot_bypass_storage_preflight(tmp_path, monkeypatch):
    config, resolved, report = campaign.validate()
    monkeypatch.setattr(campaign, "validate", lambda _: (config, resolved, report))
    monkeypatch.setattr(
        storage.shutil, "disk_usage", lambda _: SimpleNamespace(free=8 * storage.GIB)
    )
    monkeypatch.setattr(
        campaign.campaign, "execute_plan", lambda *a, **k: pytest.fail("job started")
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "task3_double_campaign",
            "run",
            "--binding-dir",
            str(tmp_path),
            "--output-root",
            str(tmp_path / "new-run"),
            "--allocation-hours",
            "12",
        ],
    )
    with pytest.raises(ValueError, match="8 GiB is insufficient"):
        campaign.main()
    assert not (tmp_path / "new-run").exists()


def test_shell_uses_same_output_filesystem_guard():
    launcher = (Path(campaign.ROOT) / "scripts/run_issue150_server.sh").read_text()
    assert 'storage-check --binding-dir "$ROOT/binding" --output-root "$ROOT/runs"' in launcher
    assert "8 GiB free disk for compact logging" not in launcher


def test_runner_rejects_low_storage_before_copying_workspace(tmp_path, monkeypatch):
    guard = monitor(tmp_path, monkeypatch)
    monkeypatch.setattr(
        storage.shutil, "disk_usage", lambda _: SimpleNamespace(free=3 * storage.GIB)
    )
    monkeypatch.setattr(run_plan, "_alias_directory", lambda *a: tmp_path / "alias")
    monkeypatch.setattr(
        run_plan, "_prepare_replica_workspace", lambda *a: pytest.fail("workspace copied")
    )
    job = SimpleNamespace(run_id="j", kind="evaluation", replica="r1")
    plan = SimpleNamespace(replicas=[SimpleNamespace(replica_id="r1")])
    status = {"jobs": {"j": {"status": "pending", "attempts": []}}}
    with pytest.raises(CampaignResourceLimitExceeded, match="Storage reserve"):
        run_plan._run_job(
            plan, job, tmp_path / "plan", status, tmp_path / "status.json",
            threading.Lock(), process_monitor=guard,
        )
    assert status["jobs"]["j"] == {"status": "pending", "attempts": []}
    assert not (tmp_path / "plan").exists()
