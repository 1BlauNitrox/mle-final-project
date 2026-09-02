"""Tests for Issue #41 compact result analysis."""

from __future__ import annotations

import pytest

from training.analyze_dqn_task1_baseline import (
    _evaluate_criteria,
    _models_from_manifest,
    _summarize_rows,
)
from training.evaluate_dqn_task1_baseline import EvaluationModel


def test_summarize_rows_uses_ratio_of_totals_and_raw_latency() -> None:
    rows = [
        _row(coins=50, steps=100, invalid=1),
        _row(coins=25, steps=400, invalid=0),
    ]

    summary = _summarize_rows(
        model="run-01",
        rows=rows,
        decision_times=[1.0, 2.0, 3.0, 100.0],
        training_world_seed=12001,
        agent_seed=22001,
        training_episodes=10000,
        training_duration_seconds=1.0,
        model_sha256="a" * 64,
    )

    assert summary["mean_collection_fraction"] == pytest.approx(0.75)
    assert summary["full_clear_count"] == 1
    assert summary["zero_coin_count"] == 0
    assert summary["steps_per_coin"] == pytest.approx(500 / 75)
    assert summary["coins_per_100_steps"] == pytest.approx(15.0)
    assert summary["invalid_action_rate"] == pytest.approx(1 / 500)
    assert summary["decision_count"] == 4
    assert summary["decision_time_p95_ms"] == pytest.approx(85.45)
    assert summary["max_decision_time_ms"] == 100.0


def test_models_from_manifest_uses_evaluated_subset_and_recorded_seeds() -> None:
    manifest = {
        "artifacts": {
            "run-03": {
                "run": 3,
                "agent_seed": 23003,
                "artifact_name": "run-03-final-checkpoint.pt",
            },
            "run-01": {
                "run": 1,
                "agent_seed": 23001,
                "artifact_name": "run-01-final-checkpoint.pt",
            },
        }
    }

    assert _models_from_manifest(manifest) == (
        EvaluationModel(1, 23001, "run-01-final-checkpoint.pt"),
        EvaluationModel(3, 23003, "run-03-final-checkpoint.pt"),
    )


def test_partial_series_does_not_apply_five_model_criterion() -> None:
    summaries = [
        {
            "mean_collection_fraction": 0.8,
            "invalid_action_rate": 0.0,
        }
        for _ in range(3)
    ]
    aggregate = {
        "mean_collection_fraction": 0.8,
        "invalid_action_rate": 0.0,
        "action_bomb": 0,
        "decision_time_p95_ms": 1.0,
        "max_decision_time_ms": 2.0,
    }
    manifest = {
        "models": {
            f"run-{run:02d}": {"deterministic": True, "immutable": True}
            for run in range(1, 4)
        }
    }

    criteria = _evaluate_criteria(summaries, aggregate, manifest)

    assert criteria["individual_models"]["passed"] is None
    assert "requires exactly 5" in criteria["individual_models"]["reason"]


def _row(*, coins: int, steps: int, invalid: int) -> dict[str, object]:
    return {
        "coins_collected": coins,
        "initially_available_coins": 50,
        "episode_steps": steps,
        "attempted_actions": steps,
        "invalid_actions": invalid,
        "action_up": steps,
        "action_right": 0,
        "action_down": 0,
        "action_left": 0,
        "action_wait": 0,
        "action_bomb": 0,
    }
