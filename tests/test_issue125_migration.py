"""Active escape and mode preservation through Task 3 migration and callbacks."""

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from agent_code.DagobertDuckDQNTask2.features import state_to_features as parent_features
from agent_code.DagobertDuckDQNTask3 import callbacks
from agent_code.DagobertDuckDQNTask3.features import state_to_features
from agent_code.DagobertDuckDQNTask3.persistence import load_training_checkpoint
from scripts import migrate_task3_dqn_successor as cli
from tests.test_DagobertDuckDQNTask3_successor import make_state


@pytest.mark.parametrize("enabled", [False, True])
def test_all_parent_features_match_on_bomb_crate_and_opponent_boards(enabled):
    rng = np.random.default_rng(125)
    for index in range(50):
        state = make_state(others=[("opponent", 0, True, (5, 5))])
        for position in [(1, 1), (1, 3), (4, 4), (2, 4)]:
            state["field"][position] = int(rng.integers(0, 2))
        state["bombs"] = [((3, 3), index % 4)] if index % 2 else [((3, 5), index % 4)]
        assert state_to_features(state, include_continuation_features=enabled)[:26] == (
            parent_features(state, include_continuation_features=enabled)
        )


def test_cli_preserves_active_online_target_and_modes_but_resets_training(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "agent_code/DagobertDuckDQNTask2/checkpoint.pt"
    payload = torch.load(source, map_location="cpu", weights_only=True)
    payload["config"]["escape_continuation_features"] = True
    payload["config"]["action_masking"] = True
    for key in ("online_network", "target_network"):
        payload["learner_state"][key]["layers.0.weight"][:, 21:] = (
            0.3 if key == "online_network" else 0.7
        )
    parent = tmp_path / "parent.pt"
    output = tmp_path / "task3.pt"
    torch.save(payload, parent)
    before = hashlib.sha256(parent.read_bytes()).hexdigest()
    monkeypatch.setattr(
        "sys.argv",
        ["migrate", "--parent", str(parent), "--output", str(output), "--parent-sha256", before],
    )
    cli.main()
    loaded = load_training_checkpoint(output)
    assert loaded.config.escape_continuation_features and loaded.config.action_masking
    assert (
        loaded.completed_episodes == loaded.learner.update_steps == len(loaded.replay_buffer) == 0
    )
    for key, network in (
        ("online_network", loaded.learner.online_network),
        ("target_network", loaded.learner.target_network),
    ):
        weights = network.state_dict()["layers.0.weight"]
        assert torch.equal(weights[:, :26], payload["learner_state"][key]["layers.0.weight"])
        assert torch.count_nonzero(weights[:, 26:]) == 0
    assert hashlib.sha256(parent.read_bytes()).hexdigest() == before
    monkeypatch.setattr(callbacks, "CHECKPOINT_PATH", output)
    monkeypatch.delenv(callbacks.ACTION_MASKING_ENV, raising=False)
    monkeypatch.delenv(callbacks.ESCAPE_CONTINUATIONS_ENV, raising=False)
    agent = SimpleNamespace(train=True, logger=Mock())
    callbacks.setup(agent)
    assert agent.config.action_masking and agent.config.escape_continuation_features
    monkeypatch.setenv(callbacks.ACTION_MASKING_ENV, "none")
    with pytest.raises(ValueError, match="does not match"):
        callbacks.setup(SimpleNamespace(train=True, logger=Mock()))


def test_evaluation_rejects_explicit_mode_mismatch(monkeypatch):
    original = callbacks.load_evaluation_checkpoint(callbacks.CHECKPOINT_PATH)
    monkeypatch.setattr(
        callbacks,
        "load_evaluation_checkpoint",
        lambda path: replace(original, config=replace(original.config, action_masking=True)),
    )
    monkeypatch.setenv(callbacks.ACTION_MASKING_ENV, "none")
    with pytest.raises(ValueError, match="does not match"):
        callbacks.setup(SimpleNamespace(train=False, logger=Mock()))
