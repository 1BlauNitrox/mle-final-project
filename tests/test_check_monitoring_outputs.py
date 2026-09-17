"""Monitoring outputs pass only when every checkpoint has every world, exactly once."""

from __future__ import annotations

import json

from scripts import check_monitoring_outputs as check


def make_root(tmp_path, milestones=("control-r1@2000",)):
    (tmp_path / "reference.pt").write_bytes(b"x")
    for artifact in milestones:
        job, episode = artifact.split("@")
        folder = tmp_path / "training-resume" / job
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"milestone-{int(episode):06d}.pt").write_bytes(b"x")
    return tmp_path


def fill(root, artifacts, drop=None, agent="Bomb-omb"):
    for name, (folder, seeds) in check.suites().items():
        out = root / folder
        out.mkdir(parents=True, exist_ok=True)
        rows = [
            {"artifact": a, "world_seed": s, "agent": agent, "suite": name}
            for a in artifacts for s in seeds
            if (a, s, name) != drop
        ]
        (out / "games.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        (out / "summary.csv").write_text("x", encoding="utf-8")
        (out / "paired.csv").write_text("x", encoding="utf-8")


def test_complete_clean_outputs_pass(tmp_path):
    root = make_root(tmp_path)
    fill(root, ["reference", "control-r1@2000"])
    ok, _ = check.check(root)
    assert ok


def test_one_missing_game_fails(tmp_path):
    root = make_root(tmp_path)
    seed = check.suites()["coin-heaven"][1][0]
    fill(root, ["reference", "control-r1@2000"], drop=("control-r1@2000", seed, "coin-heaven"))
    ok, lines = check.check(root)
    assert not ok
    assert any(line.startswith("FAIL coin-heaven") for line in lines)


def test_games_from_another_agent_fail(tmp_path):
    root = make_root(tmp_path)
    fill(root, ["reference", "control-r1@2000"], agent="Bomb-omb-loopguard")
    ok, _ = check.check(root)
    assert not ok


def test_a_new_milestone_without_games_fails_until_it_is_evaluated(tmp_path):
    root = make_root(tmp_path, milestones=("control-r1@2000", "control-r1@4000"))
    fill(root, ["reference", "control-r1@2000"])
    ok, _ = check.check(root)
    assert not ok


def test_the_monitored_classic_worlds_are_the_registered_100(tmp_path):
    name, (folder, seeds) = next(iter(check.suites().items()))
    assert name == "classic-monitoring-100"
    assert folder == "milestone-evaluation-classic-monitoring-100"
    assert len(seeds) == len(set(seeds)) == 100
