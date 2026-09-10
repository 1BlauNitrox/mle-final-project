"""Parent migration and paired retention contracts for Task 3 preparation."""

from dataclasses import replace

import pytest
import torch

from agent_code.DagobertDuckDQNTask2.config import DEFAULT_CONFIG
from agent_code.DagobertDuckDQNTask2.model import build_q_network
from agent_code.DagobertDuckDQNTask3.migration import migrate_online_network
from training.run_experiment import REPOSITORY_ROOT
from training.run_plan import load_plan


def test_migration_cli_rejects_wrong_hash_before_loading(tmp_path, monkeypatch):
    from scripts import migrate_task3_dqn_successor as cli

    source = tmp_path / "parent.pt"
    output = tmp_path / "new.pt"
    source.write_bytes(b"not a checkpoint")
    monkeypatch.setattr(
        "sys.argv",
        [
            "migrate",
            "--parent",
            str(source),
            "--output",
            str(output),
            "--parent-sha256",
            "0" * 64,
        ],
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert not output.exists()
    assert source.read_bytes() == b"not a checkpoint"


def test_migration_cli_refuses_overwriting_parent(tmp_path, monkeypatch):
    from scripts import migrate_task3_dqn_successor as cli

    source = tmp_path / "parent.pt"
    source.write_bytes(b"parent")
    monkeypatch.setattr(
        "sys.argv",
        [
            "migrate",
            "--parent",
            str(source),
            "--output",
            str(source),
            "--parent-sha256",
            cli.sha256_file(source),
        ],
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert source.read_bytes() == b"parent"


def test_control_parent_migration_preserves_q_values_with_neutral_suffix():
    parent = build_q_network(
        config=replace(DEFAULT_CONFIG, escape_continuation_features=False), seed=109
    )
    successor = migrate_online_network(parent)
    # Arbitrary learned weights in the discarded columns cannot affect an off
    # parent's outputs because its continuation inputs are always zero.
    generator = torch.Generator().manual_seed(109)
    prefix = torch.rand((100, 21), generator=generator)
    parent_states = torch.cat((prefix, torch.zeros(100, 5)), dim=1)
    successor_states = torch.cat((parent_states, torch.rand((100, 13), generator=generator)), dim=1)
    with torch.no_grad():
        torch.testing.assert_close(
            parent(parent_states), successor(successor_states), atol=1e-5, rtol=0
        )


def test_active_escape_parent_cannot_silently_lose_features():
    parent = build_q_network(
        config=replace(DEFAULT_CONFIG, escape_continuation_features=True), seed=109
    )
    successor = migrate_online_network(parent)
    assert successor.config.escape_continuation_features
    states = torch.rand((100, 26), generator=torch.Generator().manual_seed(125))
    with torch.no_grad():
        torch.testing.assert_close(
            parent(states),
            successor(torch.cat((states, torch.zeros(100, 13)), 1)),
            atol=1e-5,
            rtol=0,
        )


def test_frozen_predecessor_has_identical_evaluation_conditions():
    root = REPOSITORY_ROOT / "training" / "run_plans"
    candidate = load_plan(root / "issue109-task3-vs-peaceful.yaml")
    reference = load_plan(root / "issue109-task2-predecessor-vs-peaceful.yaml")

    def conditions(plan, replica):
        return [
            (j.stage_or_suite, j.scenario, j.world_seed, j.agent_seed, j.opponents, j.rounds)
            for j in plan.jobs
            if j.kind == "evaluation" and j.replica == replica
        ]

    assert conditions(candidate, "r1") == conditions(reference, "frozen")
    assert len(conditions(reference, "frozen")) == 80
