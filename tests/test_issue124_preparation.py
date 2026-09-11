"""Verify the proposed matched schedule without authorizing execution."""

import json
from collections import Counter
from pathlib import Path

from training.run_plan import load_plan

ROOT = Path(__file__).resolve().parents[1]


def test_proposed_matrix_has_equal_budget_and_paired_scenario_seeds():
    plans = [load_plan(ROOT / f"training/run_plans/issue124-cell-{cell}.yaml") for cell in "abcd"]
    signatures = []
    evaluation = []
    for plan in plans:
        training = [job for job in plan.jobs if job.kind == "training"]
        assert len(training) == 100
        assert sum(job.rounds for job in training) == 50000
        assert all(job.rounds == 500 for job in training)
        assert Counter(job.scenario for job in training) == {
            "coin-heaven": 20,
            "loot-crate": 20,
            "classic": 60,
        }
        signature = [
            (job.replica, job.scenario, job.world_seed, job.agent_seed) for job in training
        ]
        assert len(set(signature)) == len(signature)
        signatures.append(sorted(signature))
        evaluation.append(
            [
                (job.replica, job.stage_or_suite, job.world_seed, job.agent_seed)
                for job in plan.jobs
                if job.kind == "evaluation"
            ]
        )
        assert all(
            replica.parent_artifact_sha256
            == "4ad409472e7ca008dfcc82aa26017017c90259b65f9e593020d1b63921430f60"
            for replica in plan.replicas
        )
        assert plan.escape_continuations == "on"
        assert plan.replay_treatment == "uniform"
    assert all(signature == signatures[0] for signature in signatures)
    assert all(signature == evaluation[0] for signature in evaluation)
    assert [plan.action_masking for plan in plans] == [
        "none",
        "none",
        "framework_legal",
        "framework_legal",
    ]


def test_new_job_seeds_do_not_overlap_prior_registered_seed_values():
    registered = set()
    for path in (ROOT / "training/run_plans").glob("issue124-*.yaml"):
        plan = load_plan(path)
        registered.update(seed for job in plan.jobs for seed in (job.world_seed, job.agent_seed))
    inventory = json.loads(
        (
            ROOT / "experiments/2026-09-10-task2-rehearsal-mask/registered-seed-inventory.json"
        ).read_text()
    )
    assert inventory["commit"] == "1ce18c8736b6a50773b60b95ec0f11dd4c901028"
    for name, record in inventory["files"].items():
        assert not registered.intersection(record["seeds"]), name
