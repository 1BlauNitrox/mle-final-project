"""The overnight monitor picks up complete, settled, not-yet-evaluated milestones."""

from __future__ import annotations

import json
import os
import time

from scripts import monitor_warm_lineup as monitor


def write_milestones(root, episodes, jobs=monitor.JOBS, age=monitor.SETTLE_SECONDS + 60):
    for job in jobs:
        folder = root / "training-resume" / job
        folder.mkdir(parents=True, exist_ok=True)
        for episode in episodes:
            path = folder / f"milestone-{episode:06d}.pt"
            path.write_bytes(b"checkpoint")
            stamp = time.time() - age
            os.utime(path, (stamp, stamp))
    return root


def test_only_episodes_every_job_has_are_complete(tmp_path):
    write_milestones(tmp_path, [1000, 2000])
    # one job is still working on 3000
    write_milestones(tmp_path, [3000], jobs=monitor.JOBS[:5])
    assert monitor.complete_episodes(tmp_path) == [1000, 2000]


def test_a_milestone_still_being_written_is_not_complete(tmp_path):
    write_milestones(tmp_path, [1000])
    write_milestones(tmp_path, [2000], age=5)  # written five seconds ago
    assert monitor.complete_episodes(tmp_path) == [1000]


def test_nothing_is_complete_before_the_first_milestone(tmp_path):
    for job in monitor.JOBS:
        (tmp_path / "training-resume" / job).mkdir(parents=True)
    assert monitor.complete_episodes(tmp_path) == []


def test_evaluated_episodes_come_from_the_games_log(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    rows = [
        {"artifact": "reference", "episode": 0, "world_seed": 1},
        {"artifact": "control-r1@1000", "episode": 1000, "world_seed": 1},
        {"artifact": "hard-r3@1000", "episode": 1000, "world_seed": 1},
        {"artifact": "control-r1@2000", "episode": 2000, "world_seed": 1},
    ]
    (out / "games.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert monitor.evaluated_episodes(out) == {1000, 2000}


def test_an_empty_output_folder_has_evaluated_nothing(tmp_path):
    assert monitor.evaluated_episodes(tmp_path) == set()


def test_staging_copies_the_reference_and_only_the_named_episodes(tmp_path):
    root = write_milestones(tmp_path / "run", [1000, 2000])
    (root / "reference.pt").write_bytes(b"reference")
    staging = tmp_path / "staging"
    monitor.stage(root, staging, [2000])
    assert (staging / "reference.pt").read_bytes() == b"reference"
    for job in monitor.JOBS:
        assert (staging / "training-resume" / job / "milestone-002000.pt").is_file()
        assert not (staging / "training-resume" / job / "milestone-001000.pt").is_file()


def test_staging_is_idempotent(tmp_path):
    root = write_milestones(tmp_path / "run", [1000])
    (root / "reference.pt").write_bytes(b"reference")
    staging = tmp_path / "staging"
    monitor.stage(root, staging, [1000])
    monitor.stage(root, staging, [1000])
    assert (staging / "training-resume" / monitor.JOBS[0] / "milestone-001000.pt").is_file()


def test_the_registration_it_defaults_to_is_the_hundred_world_suite():
    cfg = json.loads(monitor.REGISTRATION.read_text(encoding="utf-8"))
    assert cfg["suite"]["name"] == "classic-monitoring-100"
    assert len(cfg["suite"]["world_seeds"]) == 100
