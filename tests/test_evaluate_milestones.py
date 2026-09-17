"""Milestone monitoring reads per-round records and pairs candidates on shared worlds."""

from __future__ import annotations

import pytest

from scripts import evaluate_milestones as evaluate


def stats(agent_entry, extra_rounds=0):
    rounds = {f"Round {i:02d}": {"agents": {"Bomb-omb_0": agent_entry}} for i in range(1 + extra_rounds)}
    return {"by_agent": {"Bomb-omb_0": {"score": 99}}, "by_round": rounds}


ENTRY = {
    "score": 7, "kills": 1, "self_kills": 0, "survived": True, "survival_steps": 400,
    "coins": 2, "initially_available_coins": 8, "invalid": 3, "termination_reason": "step_limit",
}


def test_survival_is_read_from_the_round_not_the_lifetime_totals():
    # Lifetime totals carry no survival key at all; reading them reported every
    # agent as never surviving.
    record = evaluate.round_record(stats(ENTRY), "Bomb-omb")
    assert record["survived"] == 1
    assert record["score"] == 7
    assert record["collection_fraction"] == 0.25


def test_one_game_per_world_is_enforced():
    with pytest.raises(ValueError):
        evaluate.round_record(stats(ENTRY, extra_rounds=1), "Bomb-omb")


def test_candidates_are_paired_with_the_reference_world_by_world():
    games = [
        {"artifact": "reference", "world_seed": 1, "score": 1},
        {"artifact": "reference", "world_seed": 2, "score": 5},
        {"artifact": "control-r1@2000", "world_seed": 1, "score": 3},
        {"artifact": "control-r1@2000", "world_seed": 2, "score": 5},
        {"artifact": "control-r2@2000", "world_seed": 1, "score": 5},
        {"artifact": "control-r2@2000", "world_seed": 2, "score": 7},
        {"artifact": "control-r2@2000", "world_seed": 3, "score": 100},  # no reference game: unpaired
    ]
    single = evaluate.paired_difference(games, ["control-r1@2000"], "score", resamples=200)
    assert single["worlds"] == 2 and single["mean_difference"] == 1.0
    pooled = evaluate.paired_difference(games, ["control-r1@2000", "control-r2@2000"], "score", resamples=200)
    assert pooled["worlds"] == 2 and pooled["mean_difference"] == 2.0
    assert pooled["ci_low"] <= pooled["mean_difference"] <= pooled["ci_high"]


def test_artifacts_carry_their_arm_and_episode():
    assert evaluate.arm_and_episode("halved-r3@4000") == ("halved", 4000)
    assert evaluate.arm_and_episode("reference") == ("reference", 0)
