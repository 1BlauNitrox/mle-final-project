"""Exercise the real monitor's UTC round trip and fail-closed recovery guards."""

import json
import sys
from datetime import timedelta
from types import SimpleNamespace

import pytest

from scripts import resume_issue150 as recovery
from training.run_issue107_campaign import (
    CampaignLimits,
    CampaignResourceLimitExceeded,
    CampaignResourceMonitor,
)

START = "2026-09-12T11:21:26.793224+00:00"
NOW = recovery.instant(START) + timedelta(hours=7)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_real_monitor_roundtrip_keeps_bytes_usage_and_deadline(tmp_path):
    path = tmp_path / "resources.json"
    kwargs = dict(
        state_path=path,
        authorized_at=START,
        limits=CampaignLimits(**recovery.LIMITS),
        time_fn=lambda: NOW.timestamp(),
    )
    CampaignResourceMonitor(**kwargs)
    state = recovery.read(path)
    assert state["authorized_at"] == START.replace("+00:00", "Z")
    state["cpu_seconds_consumed"] = 29928.01
    write(path, state)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="another authorization"):
        CampaignResourceMonitor(**kwargs)
    adapted = recovery.monitor_adapter(CampaignResourceMonitor)
    monitor = adapted(**kwargs)
    assert path.read_bytes() == before
    monitor.check()
    after = recovery.read(path)
    assert after["cpu_seconds_consumed"] == 29928.01
    assert after["wall_seconds_elapsed"] == 7 * 3600
    assert after["authorized_at"] == state["authorized_at"]
    assert after["limits"] == state["limits"]
    kwargs["time_fn"] = lambda: (recovery.instant(START) + timedelta(hours=11)).timestamp()
    expired = adapted(**kwargs)
    with pytest.raises(CampaignResourceLimitExceeded, match="wall ceiling"):
        expired.check()


@pytest.mark.parametrize("timestamp", ["2026-09-12T11:21:26.793225Z", "2026-09-12T11:21:26.793224"])
def test_adapter_rejects_changed_or_naive_time(tmp_path, timestamp):
    path = tmp_path / "resources.json"
    write(path, {"authorized_at": timestamp})
    before = path.read_bytes()
    with pytest.raises(ValueError):
        recovery.monitor_adapter(CampaignResourceMonitor)(state_path=path, authorized_at=START)
    assert path.read_bytes() == before


def test_adapter_retains_original_limit_identity_check(tmp_path):
    path = tmp_path / "resources.json"
    write(path, {"authorized_at": START.replace("+00:00", "Z"), "limits": {}})
    with pytest.raises(ValueError, match="different limits"):
        recovery.monitor_adapter(CampaignResourceMonitor)(
            state_path=path, authorized_at=START, limits=CampaignLimits(**recovery.LIMITS)
        )


@pytest.fixture
def campaign(tmp_path):
    binding = tmp_path / "binding/binding.json"
    write(binding, {})
    auth = {
        "authorized_at": START,
        "identity": {
            "issue": 150,
            "reviewed_commit": recovery.SOURCE,
            "protocol_sha256": recovery.PROTOCOL,
            "binding_sha256": recovery.sha(binding),
        },
    }
    write(tmp_path / "runs/authorization.json", auth)
    write(
        tmp_path / "runs/resources.json",
        {
            "authorized_at": START.replace("+00:00", "Z"),
            "limits": recovery.LIMITS,
            "cpu_seconds_consumed": 29928.01,
            "active_root_pids": [],
            "limit_reached": None,
        },
    )
    for arm in ("reference", "control", "double"):
        directory = tmp_path / "runs/plans" / ("issue150-" + arm)
        jobs = {}
        for i in range(5 if arm != "reference" else 0):
            artifact = directory / "artifacts" / f"r{i}.pt"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(f"model-{arm}-{i}".encode())
            live = directory / "replicas" / f"r{i}" / "agent/checkpoint.pt"
            live.parent.mkdir(parents=True, exist_ok=True)
            live.write_bytes(artifact.read_bytes())
            jobs[str(i)] = {
                "kind": "training",
                "status": "completed",
                "replica": f"r{i}",
                "artifact": {"path": f"artifacts/r{i}.pt", "sha256": recovery.sha(artifact)},
            }
        jobs["evaluation"] = {"kind": "evaluation", "status": "running"}
        write(directory / "status.json", {"jobs": jobs})
    return tmp_path


def test_preflight_is_read_only_and_retains_interrupted_evaluation(campaign):
    before = {p: p.read_bytes() for p in campaign.rglob("*") if p.is_file()}
    _, report = recovery.check_records(campaign, NOW)
    assert report["remaining_wall_seconds"] == 3 * 3600
    assert report["remaining_cpu_seconds"] == 86400 - 29928.01
    assert report["jobs"]["issue150-double"] == {"completed": 5, "running": 1}
    assert all(p.read_bytes() == value for p, value in before.items())
    assert set(before) == {p for p in campaign.rglob("*") if p.is_file()}


@pytest.mark.parametrize(
    "field,value",
    [
        ("authorized_at", "2026-09-12T12:21:26.793224Z"),
        ("limits", {}),
        ("active_root_pids", [42]),
        ("limit_reached", "CPU ceiling exceeded"),
        ("cpu_seconds_consumed", 86400),
        ("cpu_seconds_consumed", float("nan")),
    ],
)
def test_preflight_rejects_bad_resource_state(campaign, field, value):
    path = campaign / "runs/resources.json"
    state = recovery.read(path)
    state[field] = value
    write(path, state)
    with pytest.raises(ValueError):
        recovery.check_records(campaign, NOW)


@pytest.mark.parametrize(
    "problem", ["lock", "expired", "checkpoint", "training", "source", "binding"]
)
def test_preflight_rejects_unsafe_resume(campaign, problem):
    now = NOW
    if problem == "lock":
        (campaign / "runs/.task3-campaign.lock").touch()
    elif problem == "expired":
        now += timedelta(hours=4)
    elif problem == "checkpoint":
        next((campaign / "runs/plans").glob("*/replicas/*/agent/checkpoint.pt")).write_bytes(b"bad")
    elif problem == "training":
        path = campaign / "runs/plans/issue150-double/status.json"
        status = recovery.read(path)
        status["jobs"]["0"]["status"] = "pending"
        write(path, status)
    elif problem == "source":
        path = campaign / "runs/authorization.json"
        auth = recovery.read(path)
        auth["identity"]["reviewed_commit"] = "0" * 40
        write(path, auth)
    else:
        (campaign / "binding/binding.json").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError):
        recovery.check_records(campaign, now)


def test_cli_adapter_restores_original_monitor_even_on_failure(campaign, monkeypatch):
    import training

    run_task3_campaign = SimpleNamespace(CampaignResourceMonitor=CampaignResourceMonitor)
    monkeypatch.setattr(training, "run_task3_campaign", run_task3_campaign, raising=False)

    original = run_task3_campaign.CampaignResourceMonitor
    argv = sys.argv
    auth = recovery.read(campaign / "runs/authorization.json")
    auth["identity"].update(
        authorized_by="Julius", hardware_description="same server", available_memory_gib=8
    )

    def fake_cli():
        assert "--resume" in sys.argv
        assert sys.argv[sys.argv.index("--reviewed-commit") + 1] == recovery.SOURCE
        assert sys.argv[sys.argv.index("--hardware-description") + 1] == "same server"
        monitor = run_task3_campaign.CampaignResourceMonitor(
            state_path=campaign / "runs/resources.json",
            authorized_at=START,
            limits=CampaignLimits(**recovery.LIMITS),
            time_fn=lambda: NOW.timestamp(),
        )
        monitor.check()
        raise RuntimeError("Retain failed attempt")

    monkeypatch.setattr(
        training, "task3_double_campaign", SimpleNamespace(main=fake_cli), raising=False
    )
    with pytest.raises(RuntimeError, match="Retain failed attempt"):
        recovery.run_campaign(campaign, auth)
    assert run_task3_campaign.CampaignResourceMonitor is original
    assert sys.argv is argv
    assert recovery.read(campaign / "runs/resources.json")["cpu_seconds_consumed"] == 29928.01


@pytest.mark.parametrize("execute,failed", [(False, False), (True, False), (True, True)])
def test_entrypoint_audit_marker_and_original_export(campaign, monkeypatch, execute, failed):
    import shutil

    import psutil

    repo = campaign / "repo"
    repo.mkdir()
    monkeypatch.chdir(campaign)
    monkeypatch.setattr(
        recovery.subprocess,
        "check_output",
        lambda command, **kw: recovery.SOURCE if "rev-parse" in command else "",
    )
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=16 * 1024**3))
    monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=16 * 1024**3))
    checked_auth, report = recovery.check_records(campaign, NOW)
    monkeypatch.setattr(recovery, "check_records", lambda *args: (checked_auth, report))
    calls = []

    def run(root, auth):
        calls.append("campaign")
        if failed:
            raise ValueError("original validator rejected")

    monkeypatch.setattr(recovery, "run_campaign", run)
    monkeypatch.setattr(recovery.subprocess, "run", lambda *a, **kw: calls.append((a, kw)))
    monkeypatch.setenv("TASK150_AUTHORIZED", "yes")
    monkeypatch.setattr(
        sys, "argv", ["resume.py", "--root", str(campaign)] + (["--execute"] if execute else [])
    )
    before = (campaign / "runs/authorization.json").read_bytes()
    if failed:
        with pytest.raises(ValueError, match="original validator rejected"):
            recovery.main()
    else:
        recovery.main()
    assert (campaign / "runs/authorization.json").read_bytes() == before
    assert (campaign / "run-completed").exists() == (execute and not failed)
    audits = list(campaign.glob("resume-*.jsonl"))
    if execute:
        entries = [json.loads(line) for line in audits[0].read_text().splitlines()]
        assert entries[0]["phase"] == "before"
        assert entries[-1]["phase"] == ("failed" if failed else "campaign_complete")
        assert len(calls) == (1 if failed else 2)
        if not failed:
            assert calls[-1][0][0] == [
                "bash",
                str(repo / "scripts/run_issue150_server.sh"),
                str(campaign),
                recovery.SOURCE,
            ]
    else:
        assert not audits and not calls
