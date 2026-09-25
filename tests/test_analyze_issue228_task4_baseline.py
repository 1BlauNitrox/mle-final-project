"""Tests for the Issue #228 Task 4 baseline decision."""

from __future__ import annotations

from training.analyze_issue228_task4_baseline import (
    competitive_first_place_interval,
    evaluate_criteria,
    rank_replicas,
)


def _summaries(*, first_place: float = 0.10) -> list[dict]:
    rows = []
    for index in range(1, 6):
        for scenario in (
            "competitive",
            "mixed",
            "peaceful",
            "classic",
            "coin-heaven",
            "loot-crate",
        ):
            rows.append(
                {
                    "replica": f"r{index}",
                    "scenario": scenario,
                    "mean_collection_fraction": {
                        "classic": 0.15,
                        "coin-heaven": 0.90,
                        "loot-crate": 0.20,
                    }.get(scenario, 0.0),
                    "mean_opponents_eliminated": 0.10,
                    "self_kill_rate": 0.20,
                    "mean_score": 1.0,
                    "survival_rate": 0.5,
                    "first_place_rate": (
                        first_place if scenario in {"competitive", "mixed", "peaceful"} else None
                    ),
                    "tied_first_rate": 0.0,
                    "mean_placement": 2.0,
                    "decision_time_p95_ms": 1.0,
                    "decision_time_max_ms": 2.0,
                }
            )
    return rows


def test_registered_boundary_values_pass() -> None:
    criteria = evaluate_criteria(_summaries(), True, True)
    assert len(criteria) == 10
    assert all(criteria.values())


def test_competitive_failure_blocks_baseline() -> None:
    criteria = evaluate_criteria(_summaries(first_place=0.0), True, True)
    assert not criteria["competitive_first_place_rate_at_least_0_10"]
    assert not criteria["four_replicas_positive_first_place_rate"]


def test_bootstrap_and_ranking_are_deterministic() -> None:
    rows = _summaries()
    for row in rows:
        if row["scenario"] == "competitive":
            row["first_place_rate"] += int(row["replica"][1:]) / 100
    interval = competitive_first_place_interval(rows)
    assert interval["seed"] == 228
    assert interval["resamples"] == 10_000
    assert rank_replicas(rows)[0] == "r5"
