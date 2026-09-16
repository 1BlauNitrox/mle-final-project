"""The approach analyzer's statistics and derived endpoints behave as registered."""

from __future__ import annotations

import pytest

from scripts.analyze_task3_approach import behavioral, interval, metrics

BOOTSTRAP = {"replicas": 5, "worlds": 40, "seed": 185185, "resamples": 2000}


def row(**overrides):
    agent = {
        "score": 5,
        "coins": 4,
        "initially_available_coins": 9,
        "crates_destroyed": 7,
        "kills": 1,
        "survived": True,
        "survival_steps": 200,
        "self_kills": 0,
        "invalid": 0,
        "action_bomb": 6,
        "decision_times_ms": [1.0, 2.0],
    }
    agent.update(overrides.pop("agent", {}))
    return {
        "native": {"agents": {"DagobertDuckDQNTask3": agent, "peaceful_agent": {"score": 2}}},
        "attack_steps": overrides.pop("attack_steps", 10),
        "opponent_present_steps": overrides.pop("opponent_present_steps", 160),
        "opponent_distance_sum": overrides.pop("opponent_distance_sum", 1600),
    }


def test_exposure_endpoints_are_rates_not_raw_counts():
    values = metrics(row())
    assert values["attack_opportunities_per_100_steps"] == pytest.approx(5.0)
    assert values["opponent_present_fraction"] == pytest.approx(0.8)
    assert values["mean_opponent_distance"] == pytest.approx(10.0)
    assert values["collection_fraction"] == pytest.approx(4 / 9)
    assert values["strict_win"] == 1


def test_exposure_is_defined_without_opponents_and_without_survival():
    values = metrics(row(opponent_present_steps=0, opponent_distance_sum=0))
    assert values["opponent_present_fraction"] == 0.0
    assert values["mean_opponent_distance"] != values["mean_opponent_distance"]  # NaN
    zero = metrics(row(agent={"survival_steps": 0}, attack_steps=0))
    assert zero["attack_opportunities_per_100_steps"] == 0.0


def test_missing_collection_denominator_is_refused():
    with pytest.raises(ValueError, match="collection denominator"):
        metrics(row(agent={"initially_available_coins": 0}))


def test_behavioral_comparison_ignores_decision_times_only():
    native = {"agents": {"a": {"score": 1, "decision_times_ms": [1.0], "decision_time_max": 2.0}}}
    assert behavioral(native) == {"agents": {"a": {"score": 1}}}
    assert behavioral([{"decision_times_ms": []}, {"kept": 3}]) == [{}, {"kept": 3}]


def test_interval_is_deterministic_and_narrows_with_a_wider_family():
    matrix = [[0.1 * ((r + w) % 4) for w in range(40)] for r in range(5)]
    first = interval(matrix, percent=95.0, **BOOTSTRAP)
    assert interval(matrix, percent=95.0, **BOOTSTRAP) == first
    adjusted = interval(matrix, percent=97.5, **BOOTSTRAP)
    # A Bonferroni-adjusted interval covers more, so it must be at least as wide.
    assert adjusted[0] <= first[0] and adjusted[1] >= first[1]
    assert first[0] < sum(sum(r) for r in matrix) / 200 < first[1]


def test_a_constant_difference_has_a_degenerate_interval():
    assert interval([[0.25] * 40] * 5, percent=95.0, **BOOTSTRAP) == pytest.approx([0.25, 0.25])


def test_interval_enforces_the_registered_paired_shape():
    with pytest.raises(ValueError, match="registered paired replica and world counts"):
        interval([[0.0] * 39] * 5, percent=95.0, **BOOTSTRAP)
    with pytest.raises(ValueError, match="registered paired replica and world counts"):
        interval([[0.0] * 40] * 4, percent=95.0, **BOOTSTRAP)


def test_a_non_finite_difference_yields_no_interval_instead_of_a_wrong_one():
    matrix = [[float("nan")] * 40] * 5
    low, high = interval(matrix, percent=95.0, **BOOTSTRAP)
    assert low != low and high != high
