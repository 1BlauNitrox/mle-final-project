"""Prospective contracts for the final tabular freeze."""

from pathlib import Path

import yaml

from training.run_plan import load_plan

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "training/run_plans/issue230-final-tabular-freeze-confirmation.yaml"
CONFIG = (
    ROOT
    / "experiments/2026-09-21-final-tabular-freeze-"
    "DerKleineKonkurrenzvernichter/config.yaml"
)


def test_confirmation_plan_is_evaluation_only_and_complete() -> None:
    plan = load_plan(PLAN)

    assert plan.agent == "DerKleineKonkurrenzvernichter"
    assert plan.artifact_path == "model.npz"
    assert plan.state_representation == "compact_opponent"
    assert plan.tabular_initialization == "task2_prior"
    assert plan.episode_budget == 240
    assert all(job.kind == "evaluation" for job in plan.jobs)
    assert len(plan.jobs) == 240


def test_failed_treatments_are_not_eligible() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    candidates = config["candidate_pool"]

    assert candidates["installed_task3_incumbent"]["eligible"] is True
    assert all(
        candidate["eligible"] is False
        for name, candidate in candidates.items()
        if name != "installed_task3_incumbent"
    )
    assert config["expected_cost"]["training_episodes"] == 0
