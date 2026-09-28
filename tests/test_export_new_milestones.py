"""New milestones get bundled once, and only when every job's file is fully written."""

from __future__ import annotations

import json
import os
import time
import zipfile

from scripts import export_new_milestones as export

JOBS = ["control-r1", "halved-r1"]


def run_root(tmp_path, episodes, fresh=()):
    for job in JOBS:
        folder = tmp_path / "training-resume" / job
        folder.mkdir(parents=True, exist_ok=True)
        for ep in episodes:
            path = folder / f"milestone-{ep:06d}.pt"
            path.write_bytes(b"x")
            age = 5 if (job, ep) in fresh else 3600
            os.utime(path, (time.time() - age, time.time() - age))
    return tmp_path


def test_an_episode_counts_only_when_every_job_has_it(tmp_path):
    root = run_root(tmp_path, [2000, 4000])
    (root / "training-resume" / "halved-r1" / "milestone-004000.pt").unlink()
    assert export.complete_episodes(root, time.time()) == [2000]


def test_a_milestone_still_being_written_is_not_ready(tmp_path):
    root = run_root(tmp_path, [2000, 4000], fresh={("control-r1", 4000)})
    assert export.complete_episodes(root, time.time()) == [2000]


def test_an_episode_is_exported_only_if_a_bundle_holds_all_of_its_files(tmp_path):
    bundles = tmp_path / "bundles"
    bundles.mkdir()
    full = {f"training-resume/{job}/milestone-002000.pt": {} for job in JOBS}
    partial = {"training-resume/control-r1/milestone-004000.pt": {}}
    for name, files in (("a.zip", full), ("b.zip", partial)):
        with zipfile.ZipFile(bundles / name, "w") as zf:
            zf.writestr("MANIFEST.json", json.dumps({"files": {**files, "reference.pt": {}}}))
    (bundles / "broken.zip").write_bytes(b"not a zip")
    assert export.exported_episodes(bundles, JOBS) == {2000}
