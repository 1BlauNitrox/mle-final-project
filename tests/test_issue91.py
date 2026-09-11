from types import SimpleNamespace

import numpy as np
import pytest
import torch

from agent_code.DagobertDuckDQNTask2 import callbacks
from agent_code.DagobertDuckDQNTask2.model import compute_bellman_targets
from agent_code.DagobertDuckDQNTask2.persistence import load_training_checkpoint
from training.analyze_issue91 import decision, q_probe
from training.run_issue91 import plan_spec, prepare_parent


def test_gamma_initialization_and_callback_preserve_the_single_factor(tmp_path, monkeypatch):
    paths = [tmp_path / "a.pt", tmp_path / "b.pt"]
    for path, gamma in zip(paths, (0.9, 0.97), strict=True):
        prepare_parent(path, gamma)
    a, b = [load_training_checkpoint(path) for path in paths]
    assert a.config.discount_factor == 0.9 and b.config.discount_factor == 0.97
    assert a.config.escape_continuation_features and b.config.escape_continuation_features
    for key, value in a.learner.online_network.state_dict().items():
        assert torch.equal(value, b.learner.online_network.state_dict()[key])
    for key, value in a.learner.target_network.state_dict().items():
        assert torch.equal(value, b.learner.target_network.state_dict()[key])
    assert len(a.replay_buffer) == len(b.replay_buffer) == 0
    assert a.completed_episodes == b.completed_episodes == 0
    monkeypatch.setattr(callbacks, "CHECKPOINT_PATH", paths[1])
    monkeypatch.setenv("BOMBERMAN_DQN_ESCAPE_CONTINUATIONS", "on")
    monkeypatch.setenv("BOMBERMAN_DQN_ACTION_MASKING", "none")
    monkeypatch.setenv("BOMBERMAN_DQN_REPLAY_TREATMENT", "uniform")
    agent = SimpleNamespace(logger=SimpleNamespace(info=lambda *args: None))
    callbacks._setup_training_policy(agent, 19100001)
    assert agent.config.discount_factor == agent.learner.config.discount_factor == 0.97
    assert np.isfinite(q_probe(paths[1])["q_values"]).all()
    with pytest.raises(ValueError):
        prepare_parent(paths[1], 0.97)


def test_gamma_changes_bootstrap_but_not_terminal_reward():
    args = {
        "rewards": torch.tensor([1.0, 1.0]),
        "next_q_values": torch.full((2, 6), 10.0),
        "terminals": torch.tensor([False, True]),
    }
    control = compute_bellman_targets(**args, discount_factor=0.9)
    treatment = compute_bellman_targets(**args, discount_factor=0.97)
    assert torch.allclose(control, torch.tensor([10.0, 1.0]))
    assert torch.allclose(treatment, torch.tensor([10.7, 1.0]))


def test_plans_pair_every_condition_except_checkpoint(tmp_path):
    a, b = [plan_spec(cell, tmp_path / f"{cell}.pt") for cell in ("A", "B")]
    assert a["training_stages"] == b["training_stages"]
    assert a["evaluation_suites"] == b["evaluation_suites"]
    assert [(r["world_seed"], r["agent_seed"]) for r in a["replicas"]] == [
        (r["world_seed"], r["agent_seed"]) for r in b["replicas"]
    ]
    assert sum(s["rounds"] for s in a["training_stages"]) == 10000
    assert len(a["replicas"]) == 5


def test_positive_collection_cannot_override_safety_or_invalid_evaluation():
    positive = {"mean_difference": 0.2, "ci95_lower": 0.1}
    safe = {"survival": {"ci95_lower": -0.01}}
    assert decision(positive, safe, True, True) == "adopt_gamma_0_97"
    assert decision(positive, {"survival": {"ci95_lower": -0.05}}, True, True) == (
        "retain_gamma_0_90"
    )
    assert decision(positive, safe, False, True) == "retain_gamma_0_90"
    assert decision({"mean_difference": 0.09, "ci95_lower": 0.01}, safe, True, True) == (
        "retain_gamma_0_90"
    )


def test_staged_training_and_evaluation_keep_gamma_and_parent_immutable(tmp_path):
    import yaml

    from training.run_issue91 import sha
    from training.run_plan import execute_plan, load_plan

    parent = tmp_path / "parent.pt"
    prepare_parent(parent, 0.97)
    before = sha(parent)
    spec = plan_spec("B", parent)
    spec["plan_id"] = "issue91-integration-only"
    spec["replicas"] = [spec["replicas"][0]]
    spec["replicas"][0]["world_seed"] = 9100901
    spec["replicas"][0]["agent_seed"] = 19100901
    spec["training_stages"] = [
        {"id": "coin", "scenario": "coin-heaven", "rounds": 1, "opponents": []}
    ]
    spec["evaluation_suites"] = [
        {
            "id": "smoke",
            "population": "development",
            "scenario": "classic",
            "rounds": 1,
            "opponents": [],
            "world_seeds": [29100901],
            "agent_seeds": [39100901],
        }
    ]
    path = tmp_path / "smoke.yaml"
    path.write_text(yaml.safe_dump(spec))
    result = execute_plan(load_plan(path), output_root=tmp_path / "outputs")
    loaded = load_training_checkpoint(result / "artifacts/r1/coin/checkpoint.pt")
    assert loaded.config.discount_factor == 0.97
    assert loaded.completed_episodes == 1
    assert sha(parent) == before
