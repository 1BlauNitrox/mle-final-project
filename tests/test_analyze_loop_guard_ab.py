"""The loop-guard decision follows the rule we registered, and nothing else."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts import analyze_loop_guard_ab as ab
from scripts import evaluate_milestones

ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / "experiments/2026-09-17-loop-guard-classic-ab/config.json"
RULE = json.loads(REGISTRATION.read_text(encoding="utf-8"))["decision_rule"]


def interval(mean, low, high):
    return {"mean_difference": mean, "low": low, "high": high}


def test_a_neutral_guard_with_a_tight_interval_ships():
    ships, _ = ab.decide(interval(0.05, -0.20, 0.30), interval(0.01, -0.05, 0.06), RULE)
    assert ships


def test_a_guard_that_might_cost_too_much_score_does_not_ship():
    ships, checks = ab.decide(interval(0.10, -0.45, 0.65), interval(0.0, -0.05, 0.05), RULE)
    assert not ships
    assert checks[0].startswith("FAIL")


def test_a_negative_point_estimate_does_not_ship_even_inside_the_margin():
    ships, _ = ab.decide(interval(-0.05, -0.25, 0.15), interval(0.0, -0.05, 0.05), RULE)
    assert not ships


def test_too_many_extra_self_kills_block_shipping():
    ships, checks = ab.decide(interval(0.20, -0.10, 0.50), interval(0.06, 0.01, 0.12), RULE)
    assert not ships
    assert checks[2].startswith("FAIL")


def test_a_missing_game_refuses_a_decision():
    games = [{"artifact": "control-r1@4000", "world_seed": 1, "score": 3}]
    with pytest.raises(ValueError):
        ab.matrix(games, ["control-r1@4000"], [1, 2], "score")


def test_the_interval_brackets_a_constant_difference_exactly():
    result = ab.hierarchical_interval(np.full((6, 100), 0.5), resamples=200, seed=1, percent=95)
    assert result["low"] == result["mean_difference"] == result["high"] == 0.5


def test_the_registered_worlds_are_fresh_and_never_held_out():
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    seeds = registration["suite"]["world_seeds"]
    assert len(seeds) == len(set(seeds)) == 100
    final = json.loads(
        (ROOT / "experiments/2026-09-17-task4-final-training/config.json").read_text(
            encoding="utf-8"
        )
    )
    registered = {s for suite in final["evaluation_suites"].values() for s in suite["world_seeds"]}
    registered |= {s for replica in final["training_world_seeds"] for s in replica}
    assert not set(seeds) & registered


def test_the_evaluator_refuses_a_registration_that_touches_held_out_worlds(tmp_path, monkeypatch):
    final = json.loads(
        (ROOT / "experiments/2026-09-17-task4-final-training/config.json").read_text(
            encoding="utf-8"
        )
    )
    held = final["evaluation_suites"]["holdout-rule-based"]["world_seeds"][:2]
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "suite": {
                    "name": "sneaky",
                    "scenario": "classic",
                    "opponents": [],
                    "world_seeds": [960000001, *held],
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys, "argv", ["evaluate_milestones.py", "--root", str(tmp_path), "--registration", str(bad)]
    )
    with pytest.raises(SystemExit, match="held-out"):
        evaluate_milestones.main()
