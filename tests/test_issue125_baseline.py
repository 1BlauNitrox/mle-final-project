"""Exploratory baseline binding preserves artifacts and paired conditions."""

from pathlib import Path

import pytest

from scripts.migrate_task3_dqn_successor import sha256_file
from scripts.prepare_task3_baseline import prepare
from training.run_plan import load_plan


def test_exploratory_baseline_is_bound_without_overwriting_source(tmp_path):
    parent = Path(__file__).resolve().parents[1] / "agent_code/DagobertDuckDQNTask2/checkpoint.pt"
    digest = sha256_file(parent)
    output = tmp_path / "baseline"
    binding = prepare(parent, digest, output, "a" * 40)
    assert not binding["scientific_training_authorized"]
    assert not binding["task2_complete"]
    assert sha256_file(parent) == digest
    candidate = load_plan(output / "candidate.yaml")
    reference = load_plan(output / "reference.yaml")

    def conditions(plan, replica):
        return [
            (j.stage_or_suite, j.scenario, j.world_seed, j.agent_seed, j.opponents)
            for j in plan.jobs
            if j.kind == "evaluation" and j.replica == replica
        ]

    assert conditions(candidate, "r1") == conditions(reference, "frozen")
    assert candidate.action_masking == reference.action_masking
    assert candidate.escape_continuations == reference.escape_continuations
    assert len(load_plan(output / "smoke.yaml").jobs) == 6
    with pytest.raises(ValueError, match="already exists"):
        prepare(parent, digest, output, "a" * 40)


def test_baseline_rejects_wrong_parent_before_writing(tmp_path):
    parent = tmp_path / "parent"
    parent.write_bytes(b"untrusted")
    with pytest.raises(ValueError, match="checksum"):
        prepare(parent, "0" * 64, tmp_path / "output", "a" * 40)
    assert not (tmp_path / "output").exists()
