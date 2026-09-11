"""Continuation preserves the selected policy and fails closed on prerequisites."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from agent_code.DagobertDuckDQNTask3.persistence import load_training_checkpoint, save_checkpoint
from scripts import prepare_task3_continuation as preparation
from tests.test_task3_campaign import passing_rows
from training import analyze_task3_campaign as analysis
from training.run_task3_campaign import sha256, validate_protocol, verify_prerequisite


def test_coincollector_protocol_is_paired_and_keeps_existing_capability_levels():
    config, plans, report = validate_protocol(protocol="coincollector")
    assert report["issue"] == 137 and not report["compute_authorized"]
    assert report["training_episodes"] == 50000 and report["evaluation_episodes"] == 1920
    assert config["gates"]["elimination_min"] == 0.4
    assert config["gates"]["first_place_min"] == 0.55
    assert plans["reference"].agent == "DagobertDuckDQNTask3"
    rows = passing_rows()
    for row in rows:
        row["suite"] = row["suite"].replace("classic-peaceful", "classic-coincollector")
    result = analysis.decide(rows, config)
    assert result["status"] == "exploratory_pass" and not result["task2_complete"]
    assert result["next_decision"] == "review_next_stage_no_automatic_launch"


def test_preparation_preserves_both_networks_and_resets_training(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "agent_code/DagobertDuckDQNTask3/checkpoint.pt"
    loaded = load_training_checkpoint(source)
    with torch.no_grad():
        loaded.learner.target_network.layers[0].bias.add_(0.7)
    parent = tmp_path / "selected.pt"
    save_checkpoint(
        learner=loaded.learner,
        replay_buffer=loaded.replay_buffer,
        action_rng=np.random.default_rng(137),
        epsilon=0.1,
        completed_episodes=10000,
        agent_seed=137,
        path=parent,
    )
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "result.json").write_text(
        json.dumps(
            {
                "selected_artifact_sha256": sha256(parent),
                "selected_replica": "r3",
                "authorization": {"identity": {"reviewed_commit": "a" * 40}},
            }
        )
    )
    (evidence / "observations.json.gz").write_bytes(b"synthetic prerequisite stub")
    # Compact statistical verification is separately exercised end-to-end.
    monkeypatch.setattr(preparation, "verify_compact", lambda _: {"status": "exploratory_pass"})
    monkeypatch.setattr(analysis, "verify_compact", lambda _: {"status": "exploratory_pass"})
    before = sha256(parent)
    binding = preparation.prepare(parent, evidence, tmp_path / "binding")
    successor = load_training_checkpoint(tmp_path / "binding" / binding["successor"]["path"])
    assert (
        successor.completed_episodes
        == successor.learner.update_steps
        == len(successor.replay_buffer)
        == 0
    )
    assert successor.epsilon == successor.config.initial_epsilon
    for name in ("online_network", "target_network"):
        old, new = getattr(loaded.learner, name), getattr(successor.learner, name)
        assert all(torch.equal(v, new.state_dict()[k]) for k, v in old.state_dict().items())
    assert sha256(parent) == before
    assert validate_protocol(tmp_path / "binding", "coincollector")[2]["parent_bound"]
    with pytest.raises(ValueError, match="already exists"):
        preparation.prepare(parent, evidence, tmp_path / "binding")


def test_no_passing_peaceful_result_means_no_output(tmp_path, monkeypatch):
    monkeypatch.setattr(
        preparation, "verify_compact", lambda _: {"status": "exploratory_mixed_or_negative"}
    )
    with pytest.raises(ValueError, match="passing peaceful"):
        preparation.prepare(tmp_path / "parent", tmp_path / "evidence", tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_prerequisite_cannot_name_an_unselected_parent(tmp_path, monkeypatch):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "result.json").write_text(json.dumps({"selected_artifact_sha256": "a" * 64}))
    monkeypatch.setattr(analysis, "verify_compact", lambda _: {"status": "exploratory_pass"})
    binding = {
        "parent": {"sha256": "b" * 64},
        "prerequisite": {"path": "evidence", "result_sha256": sha256(evidence / "result.json")},
    }
    with pytest.raises(ValueError, match="mechanically selected"):
        verify_prerequisite(tmp_path, binding)
