"""Tests for the Issue #139 half-strength analysis."""

from __future__ import annotations

import pytest

from training.analyze_issue139_half_strength import (
    _classic_collection_comparison,
    _criteria,
)


def _summaries(
    *, candidate_collection: float = 0.5, candidate_self_kill: float = 0.04
) -> list[dict]:
    rows = []
    for treatment in ("control", "candidate"):
        for index in range(5):
            for scenario in ("classic", "coin-heaven", "loot-crate"):
                rows.append(
                    {
                        "treatment": treatment,
                        "model": f"r{index + 1}",
                        "scenario": scenario,
                        "mean_collection_fraction": (
                            candidate_collection if treatment == "candidate" else 0.4
                        ),
                        "self_kill_rate": (
                            candidate_self_kill if treatment == "candidate" else 0.02
                        ),
                        "decision_time_p95_ms": 1.0,
                        "decision_time_max_ms": 2.0,
                    }
                )
    return rows


def _diagnostics(candidate_visits: float = 1.8) -> list[dict]:
    return [
        {
            "treatment": treatment,
            "model": f"r{index + 1}",
            "mean_visits_per_state": candidate_visits if treatment == "candidate" else 2.0,
        }
        for treatment in ("control", "candidate")
        for index in range(5)
    ]


def test_collection_comparison_pairs_models_and_seeds() -> None:
    rows = [
        {
            "treatment": treatment,
            "model": f"r{model}",
            "scenario": "classic",
            "world_seed": seed,
            "collection_fraction": value,
        }
        for treatment, value in (("control", 0.25), ("candidate", 0.5))
        for model in range(1, 6)
        for seed in (1, 2)
    ]
    comparison = _classic_collection_comparison(rows)
    assert comparison["mean_difference"] == pytest.approx(0.25)
    assert comparison["resampler_seed"] == 139


def test_all_registered_criteria_can_pass() -> None:
    criteria = _criteria(
        _summaries(),
        {"mean_difference": 0.1, "ci_lower": 0.01},
        _diagnostics(),
        True,
        True,
        True,
    )
    assert len(criteria) == 9
    assert all(criteria.values())


def test_every_registered_failure_is_reported() -> None:
    summaries = _summaries(candidate_collection=0.3, candidate_self_kill=0.055)
    for row in summaries:
        row["decision_time_p95_ms"] = 50.0
        row["decision_time_max_ms"] = 100.0
    criteria = _criteria(
        summaries,
        {"mean_difference": 0.0, "ci_lower": 0.0},
        _diagnostics(candidate_visits=1.79),
        False,
        False,
        False,
    )
    assert len(criteria) == 9
    assert not any(criteria.values())
