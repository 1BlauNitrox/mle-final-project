"""Contracts for the preregistered Issue #46 execution inputs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from agent_code.DagobertDuckDQNTask2.persistence import (
    CHECKPOINT_PATH,
    load_training_checkpoint,
)
from training.prepare_dqn_task2_experiment import (
    REPLICAS,
    SOURCE_SHA256,
    prepare_starting_artifacts,
)
from training.run_plan import REPOSITORY_ROOT, load_plan


def test_preparation_preserves_weights_and_separates_random_streams(
    tmp_path: Path,
) -> None:
    manifest_path = prepare_starting_artifacts(tmp_path / "starts")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = load_training_checkpoint(CHECKPOINT_PATH)

    assert manifest["source"]["sha256"] == SOURCE_SHA256
    assert [record["agent_seed"] for record in manifest["artifacts"]] == [
        replica.agent_seed for replica in REPLICAS
    ]

    action_draws: set[float] = set()
    replay_states: set[str] = set()
    for replica in REPLICAS:
        path = tmp_path / "starts" / replica.replica / "checkpoint.pt"
        loaded = load_training_checkpoint(path)
        assert loaded.agent_seed == replica.agent_seed
        assert loaded.completed_episodes == 0
        assert loaded.learner.update_steps == 0
        assert len(loaded.replay_buffer) == 0
        for network_name in ("online_network", "target_network"):
            actual = getattr(loaded.learner, network_name).state_dict()
            expected = getattr(source.learner, network_name).state_dict()
            assert all(torch.equal(actual[name], expected[name]) for name in expected)
        action_draws.add(float(loaded.action_rng.random()))
        replay_states.add(
            json.dumps(loaded.replay_buffer.state_dict()["rng_state"], sort_keys=True)
        )

    assert len(action_draws) == len(REPLICAS)
    assert len(replay_states) == len(REPLICAS)
    assert prepare_starting_artifacts(tmp_path / "starts") == manifest_path


def test_preparation_refuses_unknown_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "starts"
    output.mkdir()
    (output / "unknown.txt").write_text("do not replace\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to replace"):
        prepare_starting_artifacts(output)


def test_registered_plans_expand_to_exact_issue_46_budget(tmp_path: Path) -> None:
    starts = tmp_path / "starts"
    prepare_starting_artifacts(starts)
    plan_source = REPOSITORY_ROOT / "training" / "run_plans" / "issue46-dqn-task2-trained.yaml"
    raw = yaml.safe_load(plan_source.read_text(encoding="utf-8"))
    for replica in raw["replicas"]:
        replica["parent_artifact"] = str(starts / replica["id"] / "checkpoint.pt")
    test_plan = tmp_path / "plan.yaml"
    test_plan.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    plan = load_plan(test_plan)

    assert len(plan.replicas) == 5
    assert sum(job.rounds for job in plan.jobs if job.kind == "training") == 50_000
    assert sum(job.rounds for job in plan.jobs if job.kind == "evaluation") == 1_200
    assert len([job for job in plan.jobs if job.kind == "evaluation"]) == 1_200
    assert {job.stage_or_suite for job in plan.jobs if job.kind == "evaluation"} == {
        "classic-primary",
        "classic-repeat",
        "coin-heaven-primary",
        "coin-heaven-repeat",
        "loot-crate-primary",
        "loot-crate-repeat",
    }
