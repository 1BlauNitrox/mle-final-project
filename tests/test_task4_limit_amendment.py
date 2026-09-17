"""The final run's limit amendment only raises its training limits, and nothing else."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import pilot_task4_competition as pilot

ROOT = Path(__file__).resolve().parents[1]


def load(profile):
    return json.loads((ROOT / f"experiments/{pilot.profile_dir(profile)}/config.json").read_text(encoding="utf-8"))


def test_the_final_run_may_train_until_monday_morning():
    cfg = load("final-training")
    limits = pilot.stage_limits(cfg, "training")
    assert limits["wall_seconds"] == 363600 > cfg["training_limits"]["wall_seconds"]
    assert limits["cpu_seconds"] == 2400000 > cfg["training_limits"]["cpu_seconds"]
    assert limits["memory_bytes"] == cfg["training_limits"]["memory_bytes"]


def test_six_workers_cannot_exhaust_the_raised_cpu_budget_before_the_wall():
    limits = pilot.stage_limits(load("final-training"), "training")
    assert limits["cpu_seconds"] > 6 * limits["wall_seconds"]


def test_evaluation_and_latency_limits_are_untouched():
    cfg = load("final-training")
    for stage in ("evaluation", "latency"):
        assert pilot.stage_limits(cfg, stage) == cfg[f"{stage}_limits"]


@pytest.mark.parametrize("profile", [p for p in pilot.PROFILES if p != "final-training"])
def test_completed_screens_keep_the_limits_they_ran_under(profile):
    cfg = load(profile)
    for stage in ("training", "evaluation", "latency"):
        assert pilot.stage_limits(cfg, stage) == cfg[f"{stage}_limits"]


def test_an_amendment_can_never_lower_a_limit(monkeypatch):
    monkeypatch.setitem(pilot.LIMIT_AMENDMENTS, "final-training", {"training_limits": {"wall_seconds": 1}})
    with pytest.raises(ValueError):
        pilot.stage_limits(load("final-training"), "training")
