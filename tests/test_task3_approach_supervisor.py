"""The approach pilot's worker pool runs jobs concurrently and accounts for them."""

from __future__ import annotations

import json

import pytest

from scripts import pilot_task3_approach as pilot

JOBS = 6
WORKERS = 3


@pytest.fixture
def supervised(tmp_path, monkeypatch):
    """A supervisor wired to trivial self-test workers instead of real games."""
    root = tmp_path / "root"
    root.mkdir()
    config = {
        "stage_workers": {"training": WORKERS},
        "training_limits": {
            "wall_seconds": 600,
            "cpu_seconds": 600,
            "memory_bytes": 4 * 1024**3,
        },
        "minimum_free_memory_bytes": 0,
        "minimum_free_disk_bytes": 0,
    }
    monkeypatch.setenv("TASK3_APPROACH_AUTHORIZED", "yes")
    monkeypatch.delenv("TASK3_APPROACH_SELFTEST_FAIL", raising=False)
    monkeypatch.setattr(pilot, "verify", lambda _root: {})
    monkeypatch.setattr(pilot, "config", lambda: config)
    monkeypatch.setattr(
        pilot,
        "stage_jobs",
        lambda _root, _stage, _cfg: [(f"job-{index}", ["_selftest"]) for index in range(JOBS)],
    )
    return root


def read_state(root):
    return json.loads((root / "training-state.json").read_text(encoding="utf-8"))


def test_authorization_is_required_before_any_worker_starts(supervised, monkeypatch):
    monkeypatch.delenv("TASK3_APPROACH_AUTHORIZED", raising=False)
    with pytest.raises(ValueError, match="Explicit owner authorization required"):
        pilot.supervise(supervised, "training")
    assert not (supervised / "training-state.json").exists()


def test_every_job_completes_and_resources_are_accumulated(supervised):
    pilot.supervise(supervised, "training")

    state = read_state(supervised)
    assert state["status"] == "completed"
    assert set(state["completed"]) == {f"job-{index}" for index in range(JOBS)}
    assert state["active"] == []
    assert state["workers"] == WORKERS
    assert state["cpu_seconds"] > 0
    assert state["wall_seconds"] > 0
    assert state["peak_memory_bytes"] > 0
    assert not (supervised / "training.lock").exists()

    # Concurrency is the point of the pool: with three workers and six
    # one-second jobs the wall time must be far below the summed CPU time, and
    # at least one pair of jobs must genuinely overlap in time.
    windows = []
    for relative in state["completed"].values():
        result = json.loads((supervised / relative / "result.json").read_text(encoding="utf-8"))
        windows.append((result["started_unix"], result["finished_unix"]))
    overlaps = sum(
        first[0] < second[1] and second[0] < first[1]
        for index, first in enumerate(windows)
        for second in windows[index + 1 :]
    )
    assert overlaps >= WORKERS - 1
    assert state["wall_seconds"] < state["cpu_seconds"]


def test_a_failed_worker_stops_the_stage_and_retains_the_failure(supervised, monkeypatch):
    monkeypatch.setenv("TASK3_APPROACH_SELFTEST_FAIL", "yes")
    with pytest.raises(RuntimeError, match="Pilot worker failed"):
        pilot.supervise(supervised, "training")

    state = read_state(supervised)
    assert state["status"] == "failed"
    assert state["completed"] == {}
    assert state["active"] == []
    assert all(attempt["status"] == "failed" for attempt in state["attempts"])
    assert not (supervised / "training.lock").exists()


def test_a_completed_stage_is_not_silently_rerun(supervised):
    pilot.supervise(supervised, "training")
    with pytest.raises(ValueError, match="Existing stage"):
        pilot.supervise(supervised, "training")
    with pytest.raises(ValueError, match="Stage already complete"):
        pilot.supervise(supervised, "training", resume=True)
