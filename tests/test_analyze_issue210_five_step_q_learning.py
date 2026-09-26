"""Tests for the Issue #210 five-step analysis."""

from __future__ import annotations

import pytest

from training.analyze_issue210_five_step_q_learning import (
    evaluate_criteria,
    peaceful_elimination_comparison,
)


def _summaries(candidate_eliminations: float = 0.3) -> list[dict]:
    rows = []
    for treatment in ("control", "candidate"):
        for index in range(5):
            for scenario in (
                "peaceful",
                "coincollector",
                "classic",
                "coin-heaven",
                "loot-crate",
            ):
                rows.append(
                    {
                        "treatment": treatment,
                        "replica": f"r{index + 1}",
                        "scenario": scenario,
                        "mean_collection_fraction": (
                            0.5 if treatment == "candidate" else 0.48
                        ),
                        "mean_opponents_eliminated": (
                            candidate_eliminations
                            if treatment == "candidate"
                            else 0.2
                        ),
                        "self_kill_rate": 0.05,
                    }
                )
    return rows


def test_peaceful_comparison_pairs_replicas_and_seeds() -> None:
    rows = [
        {
            "treatment": treatment,
            "replica": f"r{replica}",
            "scenario": "peaceful",
            "world_seed": seed,
            "opponents_eliminated": value,
        }
        for treatment, value in (("control", 0.25), ("candidate", 0.5))
        for replica in range(1, 6)
        for seed in (1, 2)
    ]
    comparison = peaceful_elimination_comparison(rows)
    assert comparison["mean_difference"] == pytest.approx(0.25)
    assert comparison["resampler_seed"] == 210


def test_all_registered_criteria_can_pass() -> None:
    criteria = evaluate_criteria(
        _summaries(),
        {"mean_difference": 0.1, "ci_lower": 0.01},
        True,
        True,
    )
    assert len(criteria) == 9
    assert all(criteria.values())


def test_registered_failures_are_reported() -> None:
    summaries = _summaries(candidate_eliminations=0.1)
    for row in summaries:
        if row["treatment"] == "candidate":
            row["mean_collection_fraction"] = 0.3
            row["self_kill_rate"] = 0.09
        else:
            row["mean_collection_fraction"] = 0.5
            row["self_kill_rate"] = 0.05
    criteria = evaluate_criteria(
        summaries,
        {"mean_difference": -0.1, "ci_lower": -0.03},
        False,
        False,
    )
    assert len(criteria) == 9
    assert not any(criteria.values())
