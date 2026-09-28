"""The factorial analysis pairs correctly and applies the registered rule literally."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripts import analyze_survival_guard_factorial as analyze

THRESHOLDS = {
    "self_kills_upper_bound_below": 0.0,
    "score_lower_bound_above": -0.15,
    "score_point_estimate_at_least": 0.0,
}


def game(artifact, world, variant, score=1.0, self_kills=0.0):
    return {
        "artifact": artifact,
        "world_seed": world,
        "variant": variant,
        "score": score,
        "self_kills": self_kills,
        "survived": 1,
        "kills": 0,
        "coins": 1.0,
        "collection_fraction": 0.1,
        "invalid": 0,
    }


def test_registration_rule_matches_what_the_analysis_expects():
    cfg = json.loads(analyze.REGISTRATION.read_text(encoding="utf-8"))
    rule = cfg["decision_rule"]
    assert set(rule["thresholds"]) == set(THRESHOLDS)
    assert rule["primary_contrast"].split()[0] == "both"
    assert [cell["name"] for cell in cfg["design"]["cells"]][0] == analyze.CONTROL


def test_only_games_present_in_both_cells_are_paired():
    control = [
        game("control-r1@2000", 1, "control", score=1.0),
        game("control-r1@2000", 2, "control", score=1.0),
    ]
    cell = [game("control-r1@2000", 1, "both", score=3.0)]
    matrix = analyze.paired(cell, control, "score", ["control-r1@2000"], [1, 2])
    assert matrix.shape == (1, 1)
    assert matrix[0, 0] == pytest.approx(2.0)


def test_an_interrupted_cell_is_analysed_on_the_pairs_it_has():
    # World 3 was never reached and checkpoint r2 never started. Neither may
    # turn the whole interval into nothing.
    control = [
        game(a, w, "control", score=1.0)
        for a in ("control-r1@2000", "control-r2@2000")
        for w in (1, 2, 3)
    ]
    cell = [game("control-r1@2000", w, "both", score=2.0) for w in (1, 2)]
    matrix = analyze.paired(
        cell, control, "score", ["control-r1@2000", "control-r2@2000"], [1, 2, 3]
    )
    assert matrix.shape == (1, 2)
    stat = analyze.bootstrap(matrix, resamples=500, seed=980001)
    assert stat["pairs"] == 2
    assert stat["resamples_used"] == 500
    assert stat["mean"] == pytest.approx(1.0)


def test_the_reference_is_excluded_from_the_judged_checkpoints():
    control = [game("reference", 1, "control", score=9.0), game("control-r1@2000", 1, "control")]
    cell = [game("reference", 1, "both", score=0.0), game("control-r1@2000", 1, "both")]
    matrix = analyze.paired(cell, control, "score", ["control-r1@2000"], [1])
    assert matrix.shape == (1, 1)
    assert matrix[0, 0] == pytest.approx(0.0)  # the reference's -9 never enters


def test_a_cell_labelled_as_another_is_refused(tmp_path):
    folder = tmp_path / "survival-guard-both"
    folder.mkdir()
    (folder / "games.jsonl").write_text(
        json.dumps(game("control-r1@2000", 1, "control")) + "\n", encoding="utf-8"
    )
    with pytest.raises(SystemExit):
        analyze.load(folder, "both")


def test_a_clear_win_ships():
    score = {"mean": 0.4, "ci_low": 0.1, "ci_high": 0.7}
    self_kills = {"mean": -0.2, "ci_low": -0.3, "ci_high": -0.1}
    passed, lines = analyze.verdict(score, self_kills, THRESHOLDS)
    assert passed
    assert all(line.strip().startswith("PASS") for line in lines)


def test_fewer_self_kills_do_not_ship_if_score_pays_for_them():
    score = {"mean": -0.4, "ci_low": -0.8, "ci_high": 0.0}
    self_kills = {"mean": -0.3, "ci_low": -0.4, "ci_high": -0.2}
    passed, _ = analyze.verdict(score, self_kills, THRESHOLDS)
    assert not passed


def test_a_self_kill_interval_that_only_allows_an_improvement_does_not_ship():
    # The registered rule demands a reduction that clears zero, not one that is
    # merely plausible, because the whole claim is that we stop suicides.
    score = {"mean": 0.2, "ci_low": 0.0, "ci_high": 0.4}
    self_kills = {"mean": -0.1, "ci_low": -0.25, "ci_high": 0.05}
    passed, _ = analyze.verdict(score, self_kills, THRESHOLDS)
    assert not passed


def test_the_bootstrap_resamples_both_levels():
    # One checkpoint is much worse than the others, so resampling checkpoints
    # has to widen the interval well beyond the world-only spread.
    matrix = np.vstack([np.full((5, 40), 0.2), np.full((1, 40), -3.0)])
    stat = analyze.bootstrap(matrix, resamples=2000, seed=980001)
    assert stat["checkpoints"] == 6
    assert stat["worlds"] == 40
    assert stat["ci_low"] < -0.5 < stat["ci_high"]
