"""The confirmation decides on point estimates fixed in advance, not on significance."""

from __future__ import annotations

import json

from scripts import analyze_survival_guard_confirmation as confirm


def stat(mean, low, high):
    return {"mean": mean, "ci_low": low, "ci_high": high}


def test_the_registration_says_what_this_script_implements():
    cfg = json.loads(confirm.REGISTRATION.read_text(encoding="utf-8"))
    rule = cfg["decision_rule"]
    assert rule["type"].startswith("decision under uncertainty")
    assert rule["interval"]["role"] == "reported, not a gate"
    assert len(rule["ship_if_all"]) == 3
    assert cfg["suite"]["world_seeds"][0] == 990000001
    assert len(cfg["suite"]["world_seeds"]) == 250


def test_a_positive_but_not_significant_result_ships():
    # This is the whole point of the rule: the factorial's own numbers, which
    # fail a significance test, are a ship under a decision rule.
    passed, _ = confirm.decide(stat(+0.18, -0.02, +0.38), stat(-0.04, -0.09, +0.00))
    assert passed


def test_a_score_loss_does_not_ship():
    passed, _ = confirm.decide(stat(-0.05, -0.30, +0.20), stat(-0.10, -0.20, +0.00))
    assert not passed


def test_a_deep_score_floor_does_not_ship_even_with_a_positive_point_estimate():
    passed, _ = confirm.decide(stat(+0.05, -0.40, +0.50), stat(-0.10, -0.20, +0.00))
    assert not passed


def test_more_self_kills_do_not_ship_even_when_score_looks_better():
    passed, lines = confirm.decide(stat(+0.30, +0.05, +0.55), stat(+0.02, -0.03, +0.07))
    assert not passed
    assert lines[-1].strip().startswith("FAIL")


def test_exactly_neutral_is_a_ship():
    # Zero is inside every condition, so a guard that changes nothing ships.
    # The registration accepts that explicitly under known_limits.
    passed, _ = confirm.decide(stat(0.0, -0.10, +0.10), stat(0.0, -0.05, +0.05))
    assert passed
