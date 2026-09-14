"""Small fixtures test evidence rejection; no downloaded campaign or games required."""

import hashlib
import io
import json
import tarfile

import pytest

from scripts import verify_issue150_results as verify


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@pytest.mark.parametrize(
    "name,kind",
    [
        ("../escape", "file"),
        ("/escape", "file"),
        ("C:/escape", "file"),
        ("nested\\escape", "file"),
        ("link", "link"),
        ("twice", "duplicate"),
    ],
)
def test_extract_rejects_unsafe_members(tmp_path, name, kind):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w") as archive:
        for _ in range(2 if kind == "duplicate" else 1):
            entry = tarfile.TarInfo(name)
            if kind == "link":
                entry.type = tarfile.SYMTYPE
                entry.linkname = "elsewhere"
                archive.addfile(entry)
            else:
                entry.size = 1
                archive.addfile(entry, io.BytesIO(b"x"))
    data.seek(0)
    with tarfile.open(fileobj=data) as archive, pytest.raises(ValueError):
        verify.extract(archive, tmp_path)


def test_tree_detects_tampering_missing_and_extra_files(tmp_path):
    p = tmp_path / "data"
    p.write_bytes(b"good")
    records = {"data": {"sha256": verify.sha(p), "size_bytes": 4}}
    verify.verify_tree(tmp_path, records)
    p.write_bytes(b"evil")
    with pytest.raises(ValueError, match="hash/size"):
        verify.verify_tree(tmp_path, records)
    p.unlink()
    with pytest.raises(ValueError, match="inventory"):
        verify.verify_tree(tmp_path, records)
    p.write_bytes(b"good")
    (tmp_path / "extra").touch()
    with pytest.raises(ValueError, match="inventory"):
        verify.verify_tree(tmp_path, records)


@pytest.fixture
def recovery_fixture(tmp_path):
    evidence, recovery = tmp_path / "evidence", tmp_path / "recovery"
    recovery.mkdir()
    root = evidence / "campaign"
    timestamp = "2026-09-12T00:00:00Z"
    resources = {
        "authorized_at": timestamp,
        "limits": {"wall_seconds": 36000},
        "cpu_seconds_consumed": 40,
        "wall_seconds_elapsed": 200,
    }
    previous = dict(resources, cpu_seconds_consumed=20, wall_seconds_elapsed=50)
    write(root / "authorization.json", {"authorized_at": timestamp})
    write(root / "resources.json", resources)
    marker = evidence / "handoff/run-completed"
    marker.parent.mkdir()
    marker.write_text(verify.EXECUTION)
    canonical_hash = hashlib.sha256(b"model").hexdigest()
    statuses = {}
    old_hashes = {}
    for name, count in (("issue150-control", 2), ("issue150-reference", 1), ("issue150-double", 1)):
        directory = root / "plans" / name
        directory.mkdir(parents=True)
        (directory / "checkpoint.pt").write_bytes(b"model")
        jobs = {}
        for i in range(count):
            attempt = {
                "output": f"jobs/eval-{i}/attempt-001",
                "status": "completed",
                "started_at": "2026-09-12T00:00:01Z",
                "finished_at": "2026-09-12T00:00:02Z",
            }
            attempts = [attempt]
            if name == "issue150-double":
                attempts = [
                    dict(attempt, status="interrupted", finished_at="2026-09-12T00:01:41Z"),
                    dict(
                        attempt,
                        output=f"jobs/eval-{i}/attempt-002",
                        started_at="2026-09-12T00:01:41Z",
                        finished_at="2026-09-12T00:01:42Z",
                    ),
                ]
            jobs[f"eval-{i}"] = {
                "status": "completed",
                "kind": "evaluation",
                "attempts": attempts,
                "artifact": {"path": "checkpoint.pt", "sha256": canonical_hash},
            }
        path = directory / "status.json"
        status = {"jobs": jobs, "updated_at": "old"}
        write(path, status)
        old_hashes[f"runs/plans/{name}/status.json"] = verify.sha(path)
        write(path, dict(status, updated_at="new"))
        statuses[name] = jobs
    prefix = "/home/julius/task3-issue150/runs"
    controls = {
        prefix + "/plans/issue150-control/checkpoint.pt": canonical_hash,
        prefix + "/resources.json": hashlib.sha256(
            (json.dumps(previous, indent=2, sort_keys=True) + "\n").encode()
        ).hexdigest(),
    }
    controls.update(
        {prefix + "/" + key.removeprefix("runs/"): value for key, value in old_hashes.items()}
    )
    header = {
        "apply": True,
        "root": prefix,
        "controls": controls,
        "replacement_paths": 1,
        "verified_inputs": 2,
        "estimated_reclaimed_bytes": 5,
    }
    source = prefix + "/plans/issue150-control/jobs/eval-0/attempt-001-input-agent/checkpoint.pt"
    target = prefix + "/plans/issue150-control/jobs/eval-1/attempt-001-input-agent/checkpoint.pt"
    storage = [
        header,
        {
            "phase": "before",
            "source": source,
            "target": target,
            "sha256": canonical_hash,
            "size_bytes": 5,
        },
        {"phase": "verified", "target": target, "sha256": canonical_hash},
        {"phase": "complete"},
    ]
    before = {
        "phase": "before",
        "source": verify.EXECUTION,
        "helper_sha256": verify.HELPERS["resume.py"],
        "authorization_time": timestamp,
        "resource_time": timestamp,
        "retained_resources": previous,
        "remaining_wall_seconds": 35900,
        "before_sha256": {
            **old_hashes,
            "runs/authorization.json": verify.sha(root / "authorization.json"),
        },
        "jobs": {
            "issue150-control": {"completed": 2},
            "issue150-reference": {"completed": 1},
            "issue150-double": {"running": 1},
        },
    }
    resumed = [before, {"phase": "campaign_complete", "resources": resources}]
    for name, rows in (("recovery-test.jsonl", storage), ("resume-test.jsonl", resumed)):
        (recovery / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    return evidence, recovery


def test_recovery_allows_expected_status_rewrites_but_preserves_accounting(recovery_fixture):
    report = verify.verify_recovery(*recovery_fixture)
    assert report["storage_journal_verified"] and report["resume_journal_verified"]
    assert report["replacement_paths"] == 1 and report["protected_checkpoint_files"] == 1


@pytest.mark.parametrize(
    "change", ["model", "authorization", "journal", "counts", "budget", "retry"]
)
def test_recovery_rejects_changes(recovery_fixture, change):
    evidence, recovery = recovery_fixture
    if change == "model":
        (evidence / "campaign/plans/issue150-control/checkpoint.pt").write_bytes(b"wrong")
    elif change == "authorization":
        write(evidence / "campaign/authorization.json", {"authorized_at": "2026-09-13T00:00:00Z"})
    elif change == "retry":
        path = evidence / "campaign/plans/issue150-double/status.json"
        status = verify.read(path)
        status["jobs"]["eval-0"]["attempts"].pop()
        write(path, status)
    else:
        path = recovery / ("recovery-test.jsonl" if change == "journal" else "resume-test.jsonl")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if change == "journal":
            rows[2]["sha256"] = "0" * 64
        elif change == "counts":
            rows[0]["jobs"]["issue150-control"]["completed"] = 3
        else:
            rows[0]["retained_resources"]["limits"] = {"wall_seconds": 1}
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        verify.verify_recovery(evidence, recovery)
