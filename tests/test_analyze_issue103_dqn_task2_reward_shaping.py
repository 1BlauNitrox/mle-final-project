"""Decision-rule tests for the preregistered Issue #103 analysis."""

from __future__ import annotations

from training.analyze_issue103_dqn_task2_reward_shaping import _criteria


def _timing_row() -> dict[str, object]:
    return {"decision_time_p95_ms": 1.0, "decision_time_max_ms": 2.0}


def test_registered_gates_require_every_scenario_and_classic_survival() -> None:
    passing_collection = {
        "classic": {"ci_lower": -0.04},
        "coin-heaven": {"ci_lower": -0.03},
        "loot-crate": {"ci_lower": -0.02},
    }
    passing_survival = {"classic": {"ci_lower": 0.01}}

    assert all(
        _criteria([_timing_row()], passing_collection, passing_survival, True).values()
    )


def test_a_single_scenario_regression_rejects_collection_non_regression() -> None:
    collection = {
        "classic": {"ci_lower": -0.06},
        "coin-heaven": {"ci_lower": -0.01},
        "loot-crate": {"ci_lower": -0.01},
    }
    survival = {"classic": {"ci_lower": 0.01}}

    criteria = _criteria([_timing_row()], collection, survival, True)

    assert not criteria["collection_non_regression"]


def test_a_negative_classic_survival_bound_rejects_the_primary_gate() -> None:
    collection = {
        scenario: {"ci_lower": 0.0} for scenario in ("classic", "coin-heaven", "loot-crate")
    }
    survival = {"classic": {"ci_lower": -0.01}}

    criteria = _criteria([_timing_row()], collection, survival, True)

    assert not criteria["classic_survival_improved"]


def test_timing_gate_requires_both_limits_across_every_row() -> None:
    collection = {
        scenario: {"ci_lower": 0.0} for scenario in ("classic", "coin-heaven", "loot-crate")
    }
    survival = {"classic": {"ci_lower": 0.0}}
    slow_row = {"decision_time_p95_ms": 60.0, "decision_time_max_ms": 2.0}

    criteria = _criteria([_timing_row(), slow_row], collection, survival, True)

    assert not criteria["timing_within_limits"]
