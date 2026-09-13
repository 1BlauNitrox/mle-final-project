"""Tests for the Issue #142 full no-route analysis."""

from __future__ import annotations

import pytest

from training.analyze_issue142_full_no_route import _comparisons, _criteria


def _summaries(candidate_self_kill: float = 0.01) -> list[dict]:
    return [
        {
            "treatment": treatment,
            "model": f"r{model}",
            "scenario": scenario,
            "mean_collection_fraction": 0.4,
            "self_kill_rate": candidate_self_kill if treatment == "candidate" else 0.05,
            "decision_time_p95_ms": 1.0,
            "decision_time_max_ms": 2.0,
        }
        for treatment in ("control", "candidate")
        for model in range(1, 6)
        for scenario in ("classic", "coin-heaven", "loot-crate")
    ]


def _diagnostics(candidate_visits: float = 1.8) -> list[dict]:
    return [
        {
            "treatment": treatment,
            "model": f"r{model}",
            "mean_visits_per_state": candidate_visits if treatment == "candidate" else 2.0,
        }
        for treatment in ("control", "candidate")
        for model in range(1, 6)
    ]


def test_comparisons_pair_models_and_seeds() -> None:
    rows = [
        {
            "treatment": treatment,
            "model": f"r{model}",
            "scenario": "classic",
            "world_seed": seed,
            "self_kills": self_kills,
            "collection_fraction": collection,
        }
        for treatment, self_kills, collection in (
            ("control", 1, 0.25),
            ("candidate", 0, 0.5),
        )
        for model in range(1, 6)
        for seed in (1, 2)
    ]
    comparisons = _comparisons(rows)
    assert comparisons["classic_self_kill_candidate_minus_control"][
        "mean_difference"
    ] == pytest.approx(-1.0)
    assert comparisons["classic_collection_candidate_minus_control"][
        "mean_difference"
    ] == pytest.approx(0.25)


def test_all_registered_criteria_can_pass() -> None:
    criteria = _criteria(
        _summaries(),
        {
            "classic_self_kill_candidate_minus_control": {
                "mean_difference": -0.04,
                "ci_upper": -0.01,
            }
        },
        _diagnostics(),
        True,
        True,
        True,
    )
    assert len(criteria) == 9
    assert all(criteria.values())


def test_every_registered_failure_is_reported() -> None:
    summaries = _summaries(candidate_self_kill=0.06)
    for row in summaries:
        if row["treatment"] == "candidate":
            row["mean_collection_fraction"] = 0.34
        row["decision_time_p95_ms"] = 50.0
        row["decision_time_max_ms"] = 100.0
    criteria = _criteria(
        summaries,
        {
            "classic_self_kill_candidate_minus_control": {
                "mean_difference": 0.01,
                "ci_upper": 0.02,
            }
        },
        _diagnostics(candidate_visits=1.79),
        False,
        False,
        False,
    )
    assert len(criteria) == 9
    assert not any(criteria.values())
