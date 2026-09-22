"""Milestone monitoring reads per-round records and pairs candidates on shared worlds."""

from __future__ import annotations

import pytest

from scripts import evaluate_milestones as evaluate


def stats(agent_entry, extra_rounds=0):
    rounds = {
        f"Round {i:02d}": {"agents": {"Bomb-omb_0": agent_entry}} for i in range(1 + extra_rounds)
    }
    return {"by_agent": {"Bomb-omb_0": {"score": 99}}, "by_round": rounds}


ENTRY = {
    "score": 7,
    "kills": 1,
    "self_kills": 0,
    "survived": True,
    "survival_steps": 400,
    "coins": 2,
    "initially_available_coins": 8,
    "invalid": 3,
    "termination_reason": "step_limit",
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
        {
            "artifact": "control-r2@2000",
            "world_seed": 3,
            "score": 100,
        },  # no reference game: unpaired
    ]
    single = evaluate.paired_difference(games, ["control-r1@2000"], "score", resamples=200)
    assert single["worlds"] == 2 and single["mean_difference"] == 1.0
    pooled = evaluate.paired_difference(
        games, ["control-r1@2000", "control-r2@2000"], "score", resamples=200
    )
    assert pooled["worlds"] == 2 and pooled["mean_difference"] == 2.0
    assert pooled["ci_low"] <= pooled["mean_difference"] <= pooled["ci_high"]


def test_each_game_is_played_in_its_suites_scenario(monkeypatch):
    # main.py defaults to classic, so an opponent-free retention suite would
    # otherwise be played as a classic board without anyone noticing.
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        path = command[command.index("--save-stats") + 1]
        with open(path, "w", encoding="utf-8") as handle:
            import json

            json.dump(stats(ENTRY), handle)
        from types import SimpleNamespace

        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(evaluate.subprocess, "run", fake_run)
    evaluate.play("Bomb-omb", "eval-x.pt", 930102001, [], "coin-heaven")
    command = seen["command"]
    assert command[command.index("--scenario") + 1] == "coin-heaven"
    assert command[command.index("--seed") + 1] == "930102001"


def test_artifacts_carry_their_arm_and_episode():
    assert evaluate.arm_and_episode("halved-r3@4000") == ("halved", 4000)
    assert evaluate.arm_and_episode("reference") == ("reference", 0)


def test_two_guard_cells_cannot_share_one_output_folder():
    # A variant is switched on by an environment variable, so a guarded and an
    # unguarded cell are indistinguishable in the output unless we label them.
    guarded = [{"variant": "both"}, {"variant": "both"}]
    assert evaluate.variant_conflict(guarded, "both") == []
    assert evaluate.variant_conflict(guarded, "control") == ["both"]
    assert evaluate.variant_conflict(guarded, "") == ["both"]


def test_rows_written_before_variants_existed_count_as_unlabelled():
    monitoring = [{"artifact": "reference"}, {"artifact": "control-r1@2000"}]
    assert evaluate.variant_conflict(monitoring, "") == []
    assert evaluate.variant_conflict(monitoring, "control") == [""]
