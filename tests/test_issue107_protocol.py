"""Contract tests for the prospective Issue #107 factorial campaign."""

from __future__ import annotations

import numpy as np
import pytest

from training.analyze_issue107_task2_factorial import (
    _bootstrap_result,
    _eligibility,
    _select_cell,
    _select_representative,
)
from training.run_issue107_campaign import main, validate_protocol


def _comparison(mean: float, lower: float) -> dict[str, float]:
    return {
        "mean_difference": mean,
        "bonferroni_98_75_lower": lower,
        "ci95_lower": lower,
    }


def _guards(lower: float = 0.0) -> dict[str, dict[str, float]]:
    return {f"scenario-{index}": {"ci95_lower": lower} for index in range(6)}


def _summaries() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cell, classic, coin, self_kill in (
        ("A", 0.20, 0.50, 0.40),
        ("B", 0.30, 0.51, 0.20),
        ("C", 0.28, 0.70, 0.35),
        ("D", 0.35, 0.72, 0.18),
    ):
        for index in range(1, 6):
            rows.extend(
                [
                    {
                        "cell": cell,
                        "model": f"r{index}",
                        "scenario": "classic",
                        "mean_collection_fraction": classic + index / 1000,
                        "self_kill_rate": self_kill,
                    },
                    {
                        "cell": cell,
                        "model": f"r{index}",
                        "scenario": "coin-heaven",
                        "mean_collection_fraction": coin,
                        "self_kill_rate": 0.0,
                    },
                ]
            )
    return rows


def test_protocol_is_complete_and_uses_disjoint_registered_seeds() -> None:
    report = validate_protocol()

    assert report["plans"] == {
        "A": "issue107-cell-a-control",
        "B": "issue107-cell-b-escape",
        "C": "issue107-cell-c-replay",
        "D": "issue107-cell-d-combined",
        "untrained": "issue107-untrained",
        "frozen_task1": "issue107-frozen-task1",
    }
    assert report["training_replicas"] == 20
    assert report["training_episodes"] == 200_000
    assert report["evaluation_episodes"] == 5_120
    assert report["compute_authorized"] is False
    assert report["scientific_result"] is None


def test_campaign_refuses_execution_without_explicit_authorization(capsys) -> None:
    assert main([]) == 1
    assert "--authorize-compute" in capsys.readouterr().err


def test_bootstrap_reports_registered_and_multiplicity_adjusted_intervals() -> None:
    result = _bootstrap_result(np.full((5, 40), 0.2), resampler_seed=107)

    assert result["mean_difference"] == pytest.approx(0.2)
    assert result["ci95_lower"] == pytest.approx(0.2)
    assert result["bonferroni_98_75_lower"] == pytest.approx(0.2)
    assert result["paired_models"] == 5
    assert result["paired_seeds_per_model"] == 40
    assert result["resamples"] == 10_000


def test_treatment_eligibility_requires_efficacy_and_every_guard() -> None:
    comparisons = {
        "escape_b_minus_a": _comparison(0.15, 0.01),
        "escape_d_minus_c": _comparison(0.16, 0.01),
        "replay_c_minus_a": _comparison(0.10, 0.01),
        "replay_d_minus_b": _comparison(0.11, 0.01),
        "guards_b_minus_a": _guards(),
        "guards_c_minus_a": _guards(),
        "guards_d_minus_c": _guards(),
        "guards_d_minus_b": _guards(),
    }

    assert _eligibility(comparisons) == {"A": True, "B": True, "C": True, "D": True}

    comparisons["guards_d_minus_b"] = _guards(lower=-0.05)
    assert _eligibility(comparisons)["D"] is False


def test_no_eligible_treatment_falls_back_to_control() -> None:
    gates = {
        cell: {
            "overall_passed": False,
            "task2_passed": False,
            "task1_passed": False,
        }
        for cell in "ABCD"
    }
    eligibility = {"A": True, "B": False, "C": False, "D": False}

    assert _select_cell(eligibility, gates, _summaries()) == "A"


def test_eligible_cells_use_preregistered_lexicographic_order() -> None:
    gates = {
        "A": {"overall_passed": False, "task2_passed": False, "task1_passed": False},
        "B": {"overall_passed": False, "task2_passed": True, "task1_passed": False},
        "C": {"overall_passed": False, "task2_passed": False, "task1_passed": True},
        "D": {"overall_passed": True, "task2_passed": True, "task1_passed": True},
    }
    eligibility = {"A": True, "B": True, "C": True, "D": True}

    assert _select_cell(eligibility, gates, _summaries()) == "D"


def test_representative_is_median_classic_replica_not_best_seed() -> None:
    assert _select_representative("D", _summaries()) == "r3"
