"""The working copy of the episode trail rides along with the durable write.

Rewriting the whole gzipped trail after every episode made the final run
quadratic: by episode 9,000 the file was 38 MB and each episode spent about
20 seconds re-zipping it, against 3 seconds of actual game. The durable copy
under training-resume is the one a restart reads and the one we analyse; the
copy in the stage output directory is a convenience. It now follows the same
every-25-episodes cadence, and still ends complete.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import pilot_task4_competition as pilot


@pytest.fixture
def stub(monkeypatch, tmp_path):
    """A training job whose games are free, so only the writes are observable."""
    episodes = 100
    resume_every = 25
    cfg = {
        "arms": ["control"],
        "episodes_per_replica_arm": episodes,
        "arm_settings": {"control": {
            "trainable_scope": "all", "kill_reward": 5.0, "potential_scale": 1.0,
            "update_every": 1, "training_opponents": ["rule_based_agent"],
        }},
        "training_world_seeds": [[900000 + i for i in range(episodes)]],
        "replica_agent_seeds": [1],
        "checkpoint_milestones": [50, 100],
        "resume_checkpoint_every": resume_every,
    }
    monkeypatch.setattr(pilot, "config", lambda: cfg)
    monkeypatch.setattr(pilot, "sha", lambda path: "0" * 64)
    monkeypatch.setattr(pilot, "environment", lambda: {})

    played = {"n": 0}

    def fake_play(root, checkpoint, world_seed, seed, scenario, opponents, train,
                  epsilon, output, **kwargs):
        played["n"] += 1
        Path(checkpoint).write_bytes(b"checkpoint")
        return {
            "completed_episodes": played["n"],
            "world_seed": world_seed,
            "optimizer_updates": played["n"] * 10,
            "online_before_sha256": "a" * 64,
            "online_after_sha256": "b" * 64,
        }

    monkeypatch.setattr(pilot, "play", fake_play)

    writes = []
    original = pilot.zip_json

    def counting_zip_json(path, data):
        writes.append((Path(path).parent.name, len(data)))
        original(path, data)

    monkeypatch.setattr(pilot, "zip_json", counting_zip_json)

    root = tmp_path / "run"
    (root / "training-resume").mkdir(parents=True)
    (root / "initial-control.pt").write_bytes(b"initial")
    output = tmp_path / "out"
    return root, output, writes, episodes, resume_every


def test_the_working_copy_is_written_once_per_durable_point(stub):
    root, output, writes, episodes, resume_every = stub
    pilot.train_job(root, output, "control", 0)
    working = [w for w in writes if w[0] == output.name]
    assert len(working) == episodes // resume_every
    assert len(working) < episodes  # the point of the change


def test_the_durable_trail_keeps_its_cadence(stub):
    root, output, writes, episodes, resume_every = stub
    pilot.train_job(root, output, "control", 0)
    durable = [w for w in writes if w[0] == "control-r1"]
    assert len(durable) == episodes // resume_every
    assert [count for _, count in durable] == [25, 50, 75, 100]


def test_the_working_copy_holds_every_episode_when_the_job_ends(stub):
    root, output, writes, episodes, _ = stub
    pilot.train_job(root, output, "control", 0)
    rows = pilot.read_zip_json(output / "episodes.json.gz")
    assert len(rows) == episodes
    assert rows[-1]["completed_episodes"] == episodes


def test_a_job_that_does_not_end_on_a_boundary_still_writes_the_rest(stub, monkeypatch):
    root, output, writes, _, _ = stub
    cfg = pilot.config()
    cfg["training_world_seeds"] = [cfg["training_world_seeds"][0][:30]]
    pilot.train_job(root, output, "control", 0)
    rows = pilot.read_zip_json(output / "episodes.json.gz")
    assert len(rows) == 30


def test_the_result_records_every_episode(stub):
    root, output, _, episodes, _ = stub
    pilot.train_job(root, output, "control", 0)
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert result["episodes"] == episodes
