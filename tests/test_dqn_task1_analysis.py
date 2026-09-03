"""Tests for Issue #41 compact result analysis."""

from __future__ import annotations

import json
import math

import pytest

from training.analyze_dqn_task1_baseline import _aggregate_summaries, _summarize_rows


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


def test_step_metrics_use_survival_steps_not_episode_steps() -> None:
    """docs/0007 defines the step basis as survival_steps."""
    rows = [
        _row(coins=10, steps=400, invalid=0, survival_steps=100),
        _row(coins=10, steps=400, invalid=0, survival_steps=100),
    ]

    summary = _summarize_rows(
        model="run-01",
        rows=rows,
        decision_times=[1.0],
        training_world_seed=12001,
        agent_seed=22001,
        training_episodes=10,
        training_duration_seconds=1.0,
        model_sha256="a" * 64,
    )

    assert summary["total_steps"] == 200
    assert summary["steps_per_coin"] == pytest.approx(200 / 20)
    assert summary["coins_per_100_steps"] == pytest.approx(10.0)


def test_zero_coin_model_reports_unavailable_steps_per_coin() -> None:
    """A group that collected nothing has no defined steps-per-coin value."""
    rows = [
        _row(coins=0, steps=400, invalid=0),
        _row(coins=0, steps=400, invalid=0),
    ]

    summary = _summarize_rows(
        model="run-03",
        rows=rows,
        decision_times=[1.0],
        training_world_seed=12003,
        agent_seed=22003,
        training_episodes=10,
        training_duration_seconds=1.0,
        model_sha256="c" * 64,
    )

    assert summary["steps_per_coin"] is None
    assert summary["coins_per_100_steps"] == 0.0
    assert summary["total_steps"] == 800
    assert summary["zero_coin_count"] == 2


def test_aggregate_with_a_zero_coin_model_stays_finite_and_json_safe() -> None:
    """A fully diverged run must not poison the aggregate with NaN.

    Reconstructing total steps as ``total_coins * steps_per_coin`` evaluated
    ``0 * inf`` for a zero-coin model, which is ``nan`` in Python and serializes
    to a bare ``NaN`` token that strict JSON parsers reject.
    """
    healthy = _summarize_rows(
        model="run-01",
        rows=[
            _row(coins=50, steps=200, invalid=0),
            _row(coins=50, steps=200, invalid=0),
        ],
        decision_times=[1.0],
        training_world_seed=12001,
        agent_seed=22001,
        training_episodes=10,
        training_duration_seconds=1.0,
        model_sha256="a" * 64,
    )
    diverged = _summarize_rows(
        model="run-03",
        rows=[
            _row(coins=0, steps=400, invalid=0),
            _row(coins=0, steps=400, invalid=0),
        ],
        decision_times=[1.0],
        training_world_seed=12003,
        agent_seed=22003,
        training_episodes=10,
        training_duration_seconds=1.0,
        model_sha256="c" * 64,
    )

    aggregate = _aggregate_summaries([healthy, diverged])

    assert aggregate["total_steps"] == 1200
    assert aggregate["total_coins"] == 100
    assert math.isfinite(aggregate["steps_per_coin"])
    assert aggregate["steps_per_coin"] == pytest.approx(1200 / 100)
    assert math.isfinite(aggregate["coins_per_100_steps"])
    assert aggregate["mean_collection_fraction"] == pytest.approx(0.5)

    # allow_nan=False is what a strict JSON reader enforces.
    json.dumps(aggregate, allow_nan=False)


def test_aggregate_of_only_zero_coin_models_reports_unavailable() -> None:
    diverged = [
        _summarize_rows(
            model=f"run-{index:02d}",
            rows=[
                _row(coins=0, steps=400, invalid=0),
                _row(coins=0, steps=400, invalid=0),
            ],
            decision_times=[1.0],
            training_world_seed=12000 + index,
            agent_seed=22000 + index,
            training_episodes=10,
            training_duration_seconds=1.0,
            model_sha256=f"{index}" * 64,
        )
        for index in (1, 2)
    ]

    aggregate = _aggregate_summaries(diverged)

    assert aggregate["total_coins"] == 0
    assert aggregate["steps_per_coin"] is None
    assert aggregate["coins_per_100_steps"] == 0.0
    json.dumps(aggregate, allow_nan=False)


def _row(
    *,
    coins: int,
    steps: int,
    invalid: int,
    survival_steps: int | None = None,
) -> dict[str, object]:
    return {
        "coins_collected": coins,
        "initially_available_coins": 50,
        "episode_steps": steps,
        "survival_steps": steps if survival_steps is None else survival_steps,
        "attempted_actions": steps,
        "invalid_actions": invalid,
        "action_up": steps,
        "action_right": 0,
        "action_down": 0,
        "action_left": 0,
        "action_wait": 0,
        "action_bomb": 0,
    }
