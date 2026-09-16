"""A run measured in days must survive a blackout without losing its trajectory."""

from __future__ import annotations

import gzip
import json

import pytest
import torch

from scripts.pilot_task4_competition import (
    _durable_replace,
    _write,
    read_zip_json,
    resume_state,
    write_checkpoint,
    zip_json,
)


def trail(path, count):
    zip_json(path / "episodes.json.gz", [{"episode": i + 1} for i in range(count)])


def checkpoint(path, completed):
    write_checkpoint({"completed_episodes": completed}, path / "resume.pt")


def test_nothing_to_resume_starts_from_the_registered_initialization(tmp_path):
    assert resume_state(tmp_path) == (0, [])
    trail(tmp_path, 12)
    # A trail without its checkpoint is not a resume point: the weights are gone.
    assert resume_state(tmp_path) == (0, [])


def test_the_trail_is_truncated_to_the_checkpoint_it_must_agree_with(tmp_path):
    # The agent persists its checkpoint at the end of an episode and the trail is
    # written afterwards, so a crash leaves the trail ahead. The extra episodes
    # are replayed rather than trusted.
    trail(tmp_path, 40)
    checkpoint(tmp_path, 25)
    completed, rows = resume_state(tmp_path)
    assert completed == 25
    assert [row["episode"] for row in rows] == list(range(1, 26))


def test_a_checkpoint_ahead_of_its_trail_refuses_to_resume(tmp_path):
    # The dangerous direction: the audit chain for those episodes cannot be
    # reconstructed, so the replica must restart rather than silently continue
    # with a hole in the evidence.
    trail(tmp_path, 10)
    checkpoint(tmp_path, 11)
    with pytest.raises(ValueError, match="audit chain"):
        resume_state(tmp_path)


def test_an_exactly_consistent_pair_resumes_where_it_stopped(tmp_path):
    trail(tmp_path, 250)
    checkpoint(tmp_path, 250)
    completed, rows = resume_state(tmp_path)
    assert completed == 250 and len(rows) == 250


def test_progress_writes_land_whole_or_not_at_all(tmp_path):
    # Durability is the point, but atomicity is what the reader depends on: no
    # partial file is ever visible under the real name.
    target = tmp_path / "state.json"
    _write(target, {"status": "running", "completed": 3})
    assert json.loads(target.read_text(encoding="utf-8"))["completed"] == 3
    assert not list(tmp_path.glob("*.tmp"))

    zip_json(tmp_path / "rows.json.gz", [{"a": 1}])
    assert read_zip_json(tmp_path / "rows.json.gz") == [{"a": 1}]
    with gzip.open(tmp_path / "rows.json.gz", "rb") as handle:
        assert json.loads(handle.read().decode()) == [{"a": 1}]
    assert not list(tmp_path.glob("*.tmp"))

    write_checkpoint({"completed_episodes": 7}, tmp_path / "c.pt")
    assert torch.load(tmp_path / "c.pt", weights_only=True)["completed_episodes"] == 7
    assert not list(tmp_path.glob("*.tmp"))


def test_a_directory_that_cannot_be_fsynced_does_not_fail_the_write(tmp_path):
    # Windows cannot open a directory for fsync; the file's own fsync carries
    # the guarantee there, and the write must still complete.
    source = tmp_path / "payload.tmp"
    source.write_text("content", encoding="utf-8")
    _durable_replace(source, tmp_path / "payload")
    assert (tmp_path / "payload").read_text(encoding="utf-8") == "content"
