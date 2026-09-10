"""Tests for the Issue #133 potential-safety analysis."""

from __future__ import annotations

import pytest

from training.analyze_issue133_potential_safety import (
    _comparisons,
    _criteria,
)


def make_summary(
    treatment: str,
    model: str,
    *,
    collection_fraction: float,
    self_kill_rate: float,
) -> dict:
    return {
        "treatment": treatment,
        "model": model,
        "scenario": "classic",
        "mean_collection_fraction": collection_fraction,
        "self_kill_rate": self_kill_rate,
        "decision_time_p95_ms": 1.0,
        "decision_time_max_ms": 2.0,
    }


def make_training_diagnostics(
    *,
    control_visits: float = 2.0,
    candidate_visits: float = 1.8,
) -> list[dict]:
    return [
        {
            "treatment": treatment,
            "model": f"r{model_index + 1}",
            "mean_visits_per_state": (
                candidate_visits
                if treatment == "candidate"
                else control_visits
            ),
        }
        for treatment in ("control", "candidate")
        for model_index in range(5)
    ]


def test_comparisons_pair_treatments_by_replica_and_seed() -> None:
    rows = []

    for treatment, collection_fraction, self_kills in (
        ("control", 0.25, 1),
        ("candidate", 0.50, 0),
    ):
        for model_index in range(5):
            model = f"r{model_index + 1}"

            for world_seed in (1, 2):
                rows.extend(
                    [
                        {
                            "treatment": treatment,
                            "model": model,
                            "scenario": "classic",
                            "world_seed": world_seed,
                            "collection_fraction": collection_fraction,
                            "self_kills": self_kills,
                        },
                        {
                            "treatment": treatment,
                            "model": model,
                            "scenario": "coin-heaven",
                            "world_seed": world_seed,
                            "collection_fraction": collection_fraction,
                            "self_kills": 0,
                        },
                    ]
                )

    comparisons = _comparisons(rows)

    assert comparisons[
        "classic_self_kill_candidate_minus_control"
    ]["mean_difference"] == pytest.approx(-1.0)
    assert comparisons[
        "classic_collection_candidate_minus_control"
    ]["mean_difference"] == pytest.approx(0.25)
    assert comparisons[
        "coin_heaven_retention_candidate_minus_control"
    ]["mean_difference"] == pytest.approx(0.25)


def test_all_registered_criteria_can_pass() -> None:
    summaries = [
        make_summary(
            treatment,
            f"r{model_index + 1}",
            collection_fraction=(
                0.36 if treatment == "candidate" else 0.40
            ),
            self_kill_rate=(
                0.10 if treatment == "candidate" else 0.20
            ),
        )
        for treatment in ("control", "candidate")
        for model_index in range(5)
    ]
    comparisons = {
        "classic_self_kill_candidate_minus_control": {
            "mean_difference": -0.10,
            "ci_upper": -0.01,
        },
        "classic_collection_candidate_minus_control": {
            "mean_difference": -0.04,
            "ci_lower": -0.04,
        },
        "coin_heaven_retention_candidate_minus_control": {
            "ci_lower": 0.0,
        },
    }

    criteria = _criteria(
        [],
        summaries,
        comparisons,
        make_training_diagnostics(),
        deterministic=True,
        repeat_latency_ok=True,
    )

    assert all(criteria.values())


def test_registered_safety_performance_and_reuse_failures() -> None:
    summaries = [
        make_summary(
            treatment,
            f"r{model_index + 1}",
            collection_fraction=(
                0.0 if treatment == "candidate" else 0.40
            ),
            self_kill_rate=0.20,
        )
        for treatment in ("control", "candidate")
        for model_index in range(5)
    ]
    comparisons = {
        "classic_self_kill_candidate_minus_control": {
            "mean_difference": 0.0,
            "ci_upper": 0.05,
        },
        "classic_collection_candidate_minus_control": {
            "mean_difference": -0.10,
            "ci_lower": -0.06,
        },
        "coin_heaven_retention_candidate_minus_control": {
            "ci_lower": -0.06,
        },
    }

    criteria = _criteria(
        [],
        summaries,
        comparisons,
        make_training_diagnostics(candidate_visits=1.79),
        deterministic=True,
        repeat_latency_ok=True,
    )

    assert not criteria["classic_self_kill_rate_below_control"]
    assert not criteria["at_least_four_replicas_lower_self_kill"]
    assert not criteria["classic_collection_decrease_within_0_05"]
    assert not criteria[
        "mean_visits_per_state_ratio_at_least_0_90"
    ]
    assert not criteria["classic_self_kill_ci_upper_below_zero"]
    assert not criteria["candidate_classic_collection_positive"]
    assert not criteria[
        "coin_heaven_retention_ci_lower_above_minus_0_05"
    ]


def test_determinism_and_latency_failures_are_reported() -> None:
    summaries = [
        make_summary(
            treatment,
            f"r{model_index + 1}",
            collection_fraction=0.40,
            self_kill_rate=(
                0.10 if treatment == "candidate" else 0.20
            ),
        )
        for treatment in ("control", "candidate")
        for model_index in range(5)
    ]
    summaries[0]["decision_time_p95_ms"] = 50.0

    comparisons = {
        "classic_self_kill_candidate_minus_control": {
            "mean_difference": -0.10,
            "ci_upper": -0.01,
        },
        "classic_collection_candidate_minus_control": {
            "mean_difference": 0.0,
            "ci_lower": 0.0,
        },
        "coin_heaven_retention_candidate_minus_control": {
            "ci_lower": 0.0,
        },
    }

    criteria = _criteria(
        [],
        summaries,
        comparisons,
        make_training_diagnostics(),
        deterministic=False,
        repeat_latency_ok=False,
    )

    assert not criteria["deterministic_repeats"]
    assert not criteria["primary_latency_within_limits"]
    assert not criteria["repeat_latency_within_limits"]
    assert criteria["classic_self_kill_ci_upper_below_zero"]
    assert criteria[
        "coin_heaven_retention_ci_lower_above_minus_0_05"
    ]
