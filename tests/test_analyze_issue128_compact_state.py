"""Tests for the Issue #128 compact-state analysis."""

from __future__ import annotations

import pytest

from training.analyze_issue128_compact_state import (
    _comparisons,
    _criteria,
    _evaluation_unseen_rate,
)


def make_summary(
    treatment: str,
    model: str,
    collection_fraction: float,
) -> dict:
    return {
        "treatment": treatment,
        "model": model,
        "scenario": "classic",
        "mean_collection_fraction": collection_fraction,
        "decision_time_p95_ms": 1.0,
        "decision_time_max_ms": 2.0,
    }


def make_rows(
    *,
    candidate_self_kills: int = 0,
    candidate_unseen: int = 0,
) -> list[dict]:
    rows = []

    for treatment in ("control", "candidate"):
        for model_index in range(5):
            rows.append(
                {
                    "treatment": treatment,
                    "model": f"r{model_index + 1}",
                    "scenario": "classic",
                    "self_kills": (
                        candidate_self_kills
                        if treatment == "candidate"
                        else 0
                    ),
                    "evaluation_decisions": 100,
                    "evaluation_unseen_decisions": (
                        candidate_unseen
                        if treatment == "candidate"
                        else 1
                    ),
                }
            )

    return rows


def make_training_diagnostics() -> list[dict]:
    return [
        {
            "treatment": treatment,
            "model": f"r{model_index + 1}",
            "mean_visits_per_state": (
                4.0 if treatment == "candidate" else 2.0
            ),
        }
        for treatment in ("control", "candidate")
        for model_index in range(5)
    ]


def test_comparisons_pair_treatments_by_replica_and_seed() -> None:
    rows = []

    for treatment, classic_fraction, coin_fraction in (
        ("control", 0.25, 0.75),
        ("candidate", 0.50, 0.75),
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
                            "collection_fraction": classic_fraction,
                        },
                        {
                            "treatment": treatment,
                            "model": model,
                            "scenario": "coin-heaven",
                            "world_seed": world_seed,
                            "collection_fraction": coin_fraction,
                        },
                    ]
                )

    comparisons = _comparisons(rows)

    assert comparisons[
        "classic_candidate_minus_control"
    ]["mean_difference"] == pytest.approx(0.25)
    assert comparisons[
        "coin_heaven_candidate_minus_control"
    ]["mean_difference"] == pytest.approx(0.0)


def test_all_registered_criteria_can_pass() -> None:
    summaries = [
        make_summary(
            treatment,
            f"r{model_index + 1}",
            0.4 if treatment == "candidate" else 0.2,
        )
        for treatment in ("control", "candidate")
        for model_index in range(5)
    ]
    comparisons = {
        "classic_candidate_minus_control": {
            "mean_difference": 0.2,
            "ci_lower": 0.1,
        },
        "coin_heaven_candidate_minus_control": {
            "ci_lower": 0.0,
        },
    }

    criteria = _criteria(
        make_rows(),
        summaries,
        comparisons,
        make_training_diagnostics(),
        deterministic=True,
        repeat_latency_ok=True,
    )

    assert all(criteria.values())


def test_safety_and_unseen_state_regressions_fail() -> None:
    summaries = [
        make_summary(
            treatment,
            f"r{model_index + 1}",
            0.4 if treatment == "candidate" else 0.2,
        )
        for treatment in ("control", "candidate")
        for model_index in range(5)
    ]
    comparisons = {
        "classic_candidate_minus_control": {
            "mean_difference": 0.2,
            "ci_lower": 0.1,
        },
        "coin_heaven_candidate_minus_control": {
            "ci_lower": 0.0,
        },
    }

    criteria = _criteria(
        make_rows(
            candidate_self_kills=1,
            candidate_unseen=1,
        ),
        summaries,
        comparisons,
        make_training_diagnostics(),
        deterministic=True,
        repeat_latency_ok=True,
    )

    assert not criteria["classic_self_kill_rate_within_limit"]
    assert not criteria[
        "candidate_unseen_state_rate_below_control"
    ]


def test_unseen_state_rate_uses_ratio_of_totals() -> None:
    rows = [
        {
            "treatment": "candidate",
            "evaluation_decisions": 10,
            "evaluation_unseen_decisions": 2,
        },
        {
            "treatment": "candidate",
            "evaluation_decisions": 30,
            "evaluation_unseen_decisions": 2,
        },
    ]

    assert _evaluation_unseen_rate(
        rows,
        "candidate",
    ) == pytest.approx(0.1)