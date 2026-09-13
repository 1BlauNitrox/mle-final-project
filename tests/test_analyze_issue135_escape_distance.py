"""Tests for the Issue #135 escape-distance analysis."""

from __future__ import annotations

import pytest

from training.analyze_issue135_escape_distance import _comparisons, _criteria


def make_summaries(
    *,
    candidate_self_kill: float = 0.1,
    control_self_kill: float = 0.2,
    p95: float = 1.0,
    maximum: float = 2.0,
) -> list[dict]:
    return [
        {
            "treatment": treatment,
            "model": f"r{index + 1}",
            "scenario": "classic",
            "mean_collection_fraction": 0.4,
            "self_kill_rate": (
                candidate_self_kill
                if treatment == "candidate"
                else control_self_kill
            ),
            "decision_time_p95_ms": p95,
            "decision_time_max_ms": maximum,
        }
        for treatment in ("control", "candidate")
        for index in range(5)
    ]


def make_diagnostics(candidate_visits: float = 1.8) -> list[dict]:
    return [
        {
            "treatment": treatment,
            "model": f"r{index + 1}",
            "mean_visits_per_state": (
                candidate_visits if treatment == "candidate" else 2.0
            ),
        }
        for treatment in ("control", "candidate")
        for index in range(5)
    ]


def test_comparisons_pair_treatments_by_replica_and_seed() -> None:
    rows = []
    for treatment, collection, self_kills in (
        ("control", 0.25, 1),
        ("candidate", 0.50, 0),
    ):
        for model_index in range(5):
            for world_seed in (1, 2):
                rows.append(
                    {
                        "treatment": treatment,
                        "model": f"r{model_index + 1}",
                        "scenario": "classic",
                        "world_seed": world_seed,
                        "collection_fraction": collection,
                        "self_kills": self_kills,
                    }
                )

    comparisons = _comparisons(rows)

    assert comparisons[
        "classic_self_kill_candidate_minus_control"
    ]["mean_difference"] == pytest.approx(-1.0)
    assert comparisons[
        "classic_collection_candidate_minus_control"
    ]["mean_difference"] == pytest.approx(0.25)
    assert comparisons[
        "classic_self_kill_candidate_minus_control"
    ]["resampler_seed"] == 135
    assert comparisons[
        "classic_collection_candidate_minus_control"
    ]["resampler_seed"] == 136


def test_all_registered_criteria_can_pass() -> None:
    comparisons = {
        "classic_self_kill_candidate_minus_control": {
            "mean_difference": -0.1,
            "ci_upper": -0.01,
        },
        "classic_collection_candidate_minus_control": {
            "mean_difference": -0.04,
        },
    }

    criteria = _criteria(
        [],
        make_summaries(),
        comparisons,
        make_diagnostics(),
        True,
        True,
        True,
    )

    assert len(criteria) == 8
    assert all(criteria.values())


def test_every_registered_failure_is_reported() -> None:
    comparisons = {
        "classic_self_kill_candidate_minus_control": {
            "mean_difference": 0.0,
            "ci_upper": 0.05,
        },
        "classic_collection_candidate_minus_control": {
            "mean_difference": -0.06,
        },
    }

    criteria = _criteria(
        [],
        make_summaries(
            candidate_self_kill=0.2,
            control_self_kill=0.2,
            p95=50.0,
            maximum=100.0,
        ),
        comparisons,
        make_diagnostics(candidate_visits=1.79),
        False,
        False,
        False,
    )

    assert len(criteria) == 8
    assert not any(criteria.values())


def test_control_visits_must_be_positive() -> None:
    diagnostics = make_diagnostics()
    for row in diagnostics:
        if row["treatment"] == "control":
            row["mean_visits_per_state"] = 0.0

    with pytest.raises(ValueError, match="must be positive"):
        _criteria(
            [],
            make_summaries(),
            {
                "classic_self_kill_candidate_minus_control": {
                    "mean_difference": -0.1,
                    "ci_upper": -0.01,
                },
                "classic_collection_candidate_minus_control": {
                    "mean_difference": 0.0,
                },
            },
            diagnostics,
            True,
            True,
            True,
        )
