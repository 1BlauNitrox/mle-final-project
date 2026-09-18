"""The attack-rule decision demands kills first, then that they pay for themselves."""

from __future__ import annotations

import json

from scripts import analyze_attack_rule as analyze


def stat(mean, low, high):
    return {"mean": mean, "ci_low": low, "ci_high": high}


def test_the_registration_says_what_this_script_implements():
    cfg = json.loads(analyze.REGISTRATION.read_text(encoding="utf-8"))
    rule = cfg["decision_rule"]
    assert rule["type"].startswith("decision under uncertainty")
    assert rule["interval"]["role"] == "reported, not a gate"
    assert rule["primary_contrast"] == "attack minus control"
    assert [c["name"] for c in cfg["design"]["cells"]] == ["control", "attack", "attack-guard"]
    assert len(cfg["suite"]["world_seeds"]) == 150


def test_kills_that_pay_for_themselves_ship():
    passed, _ = analyze.decide(stat(+0.35, +0.02, +0.70), stat(+0.06, +0.01, +0.11))
    assert passed


def test_no_kills_means_no_reason_to_exist():
    # Score could drift up on noise alone; without kills the rule did nothing.
    passed, lines = analyze.decide(stat(+0.20, -0.05, +0.45), stat(0.0, -0.03, +0.03))
    assert not passed
    assert lines[0].strip().startswith("FAIL")


def test_kills_bought_with_score_do_not_ship():
    passed, _ = analyze.decide(stat(-0.30, -0.70, +0.10), stat(+0.09, +0.03, +0.15))
    assert not passed


def test_a_deep_score_floor_does_not_ship_even_with_kills_and_a_positive_estimate():
    passed, _ = analyze.decide(stat(+0.05, -0.40, +0.50), stat(+0.08, +0.02, +0.14))
    assert not passed


def test_the_cell_names_match_the_registration():
    cfg = json.loads(analyze.REGISTRATION.read_text(encoding="utf-8"))
    names = {c["name"] for c in cfg["design"]["cells"]}
    assert {analyze.CONTROL, analyze.ATTACK, analyze.STACKED} == names


def test_every_cell_declares_all_three_switches():
    # A leftover switch from a previous cell is the one mistake that would
    # silently produce the wrong comparison, so the registration spells each
    # cell's full environment out.
    cfg = json.loads(analyze.REGISTRATION.read_text(encoding="utf-8"))
    for cell, environment in cfg["design"]["environment"].items():
        assert set(environment) == {
            "BOMBERMAN_ATTACK_RULE",
            "BOMBERMAN_SURVIVAL_GUARD_BOMB",
            "BOMBERMAN_SURVIVAL_GUARD_MOVE",
        }, cell
        assert set(environment.values()) <= {"on", "off"}, cell
