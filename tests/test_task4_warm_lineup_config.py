"""The warm-started line-up registration says what we think it says.

This run continues from the final run's own agent rather than from the Task 3
incumbent, and varies only the training line-up. The checks below are the ones
that would silently ruin it: a warm start that is not the artifact we verified, a
second factor drifting between the arms, or development worlds the previous run
already played.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
WARM = ROOT / "experiments/2026-09-18-task4-warm-lineup/config.json"
FINAL = ROOT / "experiments/2026-09-17-task4-final-training/config.json"
WARM_START_SHA = "df7ff283a4c559eaf2180e2edc5bdb537601eafd734a5f2df4b53d69c477d3ad"


@pytest.fixture(scope="module")
def cfg():
    return json.loads(WARM.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def final():
    return json.loads(FINAL.read_text(encoding="utf-8"))


def test_the_pilot_accepts_the_registration(monkeypatch):
    monkeypatch.setenv("TASK4_PROFILE", "warm-lineup")
    import importlib

    from scripts import pilot_task4_competition as pilot

    importlib.reload(pilot)
    try:
        loaded = pilot.config()
        assert loaded["profile"] == "warm-lineup"
        assert pilot.canonical_sha(loaded) == pilot.PROFILE_HASHES["warm-lineup"]
    finally:
        monkeypatch.delenv("TASK4_PROFILE", raising=False)
        importlib.reload(pilot)


def test_it_starts_from_the_artifact_we_migrated_and_verified(cfg):
    assert cfg["initial_sha256"] == WARM_START_SHA
    assert cfg["warm_start"]["initial_sha256"] == WARM_START_SHA
    assert "milestone-008000" in cfg["warm_start"]["source_artifact"]


def test_only_the_line_up_differs_between_the_arms(cfg):
    control, hard = cfg["arm_settings"]["control"], cfg["arm_settings"]["hard"]
    differing = {key for key in set(control) | set(hard) if control.get(key) != hard.get(key)}
    assert differing == {"training_opponents"}
    assert hard["training_opponents"] == ["rule_based_agent"] * 3
    assert control["training_opponents"] == [
        "rule_based_agent",
        "coin_collector_agent",
        "peaceful_agent",
    ]


def test_the_control_arm_reproduces_what_the_previous_run_trained_against(cfg, final):
    assert (
        cfg["arm_settings"]["control"]["training_opponents"]
        == final["arm_settings"]["control"]["training_opponents"]
    )


def test_both_arms_use_the_gentler_learning_rate(cfg):
    assert {s["learning_rate"] for s in cfg["arm_settings"].values()} == {0.0001}
    assert {s["kill_reward"] for s in cfg["arm_settings"].values()} == {5.0}


def test_no_world_is_reused_from_the_previous_run(cfg, final):
    def worlds(config):
        seeds = {s for replica in config["training_world_seeds"] for s in replica}
        seeds |= {s for suite in config["evaluation_suites"].values() for s in suite["world_seeds"]}
        return seeds

    assert not worlds(cfg) & worlds(final)


def test_the_held_out_suite_is_not_carried_into_this_run(cfg):
    assert not any("holdout" in name for name in cfg["evaluation_suites"])


def test_the_declared_totals_match_the_seeds(cfg):
    assert (
        cfg["training_episodes"]
        == len(cfg["arms"]) * cfg["replicas"] * cfg["episodes_per_replica_arm"]
    )
    for replica in cfg["training_world_seeds"]:
        assert len(replica) == cfg["episodes_per_replica_arm"]
    assert len(cfg["training_world_seeds"]) == cfg["replicas"]


def test_milestones_are_frequent_enough_to_stop_anywhere(cfg):
    milestones = cfg["checkpoint_milestones"]
    assert milestones[0] == 1000 and milestones[-1] == cfg["episodes_per_replica_arm"]
    assert all(b - a == 1000 for a, b in zip(milestones, milestones[1:], strict=False))


def test_the_seed_audit_passed(cfg):
    audit = json.loads((WARM.parent / "seed-audit.json").read_text(encoding="utf-8"))
    assert audit["passed"] is True
    assert audit["colliding_files"] == []
    assert audit["cross_profile_separation"]["shared_seed_count"] == 0
