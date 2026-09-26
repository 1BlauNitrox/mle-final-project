from pathlib import Path

import numpy as np
import pytest
import torch

from agent_code.DagobertDuckDQNTask3.config import DQNConfig
from agent_code.DagobertDuckDQNTask3.model import DQNLearner
from agent_code.DagobertDuckDQNTask3.persistence import load_training_checkpoint, save_checkpoint
from agent_code.DagobertDuckDQNTask3.replay import ReplayBuffer
from scripts.analyze_stability_test import interval
from scripts.run_stability_test import epsilon, initialize, jobs, read, save_generation, sha


def test_preserve_state_bundle_and_paired_weights(tmp_path):
    config = DQNConfig()
    learner = DQNLearner(config=config, seed=13)
    # Populate actual Adam moments without requiring a scientific game.
    learner.optimizer.zero_grad()
    sum(p.sum() for p in learner.online_network.parameters()).backward()
    learner.optimizer.step()
    learner.update_steps = 852
    replay = ReplayBuffer(capacity=config.replay_capacity, seed=13)
    # Use the real transition API and retain one nonempty state.
    replay.add(
        state=np.zeros(39, dtype=np.float32),
        action_index=0,
        reward=1.0,
        next_state=np.ones(39, dtype=np.float32),
        terminal=False,
    )
    parent = tmp_path / "parent.pt"
    save_checkpoint(
        learner=learner,
        replay_buffer=replay,
        action_rng=np.random.default_rng(13),
        epsilon=0.2,
        completed_episodes=8000,
        agent_seed=13,
        path=parent,
    )
    before = sha(parent)
    for arm in ["reset", "preserve"]:
        initialize(parent, tmp_path / f"{arm}.pt", arm, 99, 0.0001)
    reset, keep = [load_training_checkpoint(tmp_path / f"{a}.pt") for a in ["reset", "preserve"]]
    assert len(reset.replay_buffer) == 0 and len(keep.replay_buffer) == 1
    assert not reset.learner.optimizer.state and keep.learner.optimizer.state
    assert reset.learner.update_steps == keep.learner.update_steps == 852
    assert reset.completed_episodes == keep.completed_episodes == 8000
    for agent in [reset, keep]:
        assert agent.config.learning_rate == 0.0001
        assert agent.learner.optimizer.param_groups[0]["lr"] == 0.0001
        for network in ["online_network", "target_network"]:
            for k, v in getattr(learner, network).state_dict().items():
                assert torch.equal(v, getattr(agent.learner, network).state_dict()[k])
    assert reset.action_rng.bit_generator.state == keep.action_rng.bit_generator.state
    assert (
        reset.replay_buffer.state_dict()["rng_state"]
        == keep.replay_buffer.state_dict()["rng_state"]
    )
    for i, state in learner.optimizer.state_dict()["state"].items():
        for key, tensor in state.items():
            assert torch.equal(tensor, keep.learner.optimizer.state_dict()["state"][i][key])
    assert sha(parent) == before


def test_registered_exploration_and_jobs():
    assert sum(epsilon(99, i) for i in range(100)) == 20
    assert all(
        sum(epsilon(99, i) for i in range(start, start + 5)) == 1 for start in range(0, 100, 5)
    )
    cfg = read(
        Path(__file__).parents[1] / "experiments/2026-09-19-continuation-stability/config.json"
    )
    planned = jobs(cfg)
    assert sum(j[0] == "_train" for j in planned) == 6
    assert sum(j[0] == "_evaluate" for j in planned) == 35


def test_generation_is_matched_and_old_generation_survives(tmp_path):
    checkpoint = tmp_path / "working.pt"
    checkpoint.write_bytes(b"first")
    save_generation(tmp_path, checkpoint, [{"world_seed": 1}])
    first = read(tmp_path / "resume.json")
    checkpoint.write_bytes(b"second")
    save_generation(tmp_path, checkpoint, [{"world_seed": 1}, {"world_seed": 2}])
    second = read(tmp_path / "resume.json")
    assert first["generation"] != second["generation"]
    for state in [first, second]:
        directory = tmp_path / state["generation"]
        assert sha(directory / "checkpoint.pt") == state["checkpoint_sha256"]
        assert sha(directory / "rows.json") == state["rows_sha256"]
        assert len(read(directory / "rows.json")) == state["episodes"]


def test_paired_interval_requires_three_replicas():
    assert interval([[2, 2], [2, 2], [2, 2]], seed=1, samples=50)["ci95"] == [2, 2]
    with pytest.raises(ValueError, match="Three paired"):
        interval([[1, 2]], seed=1, samples=10)


def test_exported_analysis_and_corruption_rejection(tmp_path):
    import gzip
    import json

    from scripts.analyze_stability_test import analyze
    from scripts.run_stability_test import write

    cfg = read(
        Path(__file__).parents[1] / "experiments/2026-09-19-continuation-stability/config.json"
    )
    cfg["episodes"] = 1
    cfg["bootstrap_resamples"] = 10
    cfg["train_ranges"] = [[v[0], v[0]] for v in cfg["train_ranges"]]
    for suite in cfg["evaluation"].values():
        suite["seeds"][1] = suite["seeds"][0]
    write(tmp_path / "config.json", cfg)
    write(tmp_path / "binding.json", {"config_sha256": sha(tmp_path / "config.json")})
    artifacts = [f"{arm}-r{r}" for arm in cfg["arms"] for r in range(1, 4)]
    for artifact in artifacts:
        d = tmp_path / "training" / artifact
        d.mkdir(parents=True)
        (d / "final.pt").write_bytes(artifact.encode())
        r = int(artifact[-1]) - 1
        rows = [
            {
                "world_seed": cfg["train_ranges"][r][0],
                "opponents": cfg["training_opponents"],
                "slot": 0,
                "epsilon": epsilon(cfg["learner_seeds"][r], 0),
                "optimizer_updates_this_episode": 1,
            }
        ]
        with gzip.open(d / "episodes.json.gz", "wt") as stream:
            json.dump(rows, stream)
        write(
            d / "result.json",
            {
                "complete": True,
                "episodes": 1,
                "checkpoint_sha256": sha(d / "final.pt"),
                "rows_sha256": sha(d / "episodes.json.gz"),
            },
        )
    for artifact in ["reference", *artifacts]:
        checkpoint = (
            cfg["reference_sha256"]
            if artifact == "reference"
            else sha(tmp_path / "training" / artifact / "final.pt")
        )
        for suite, setting in cfg["evaluation"].items():
            d = tmp_path / "evaluation" / artifact / suite
            d.mkdir(parents=True)
            native = dict(
                score=1,
                kills=0,
                self_kills=0,
                survived=1,
                coins=1,
                invalid=0,
                initially_available_coins=9,
                decision_times_ms=[1.0, 2.0],
            )
            rows = [
                {
                    "world_seed": setting["seeds"][0],
                    "opponents": setting["opponents"],
                    "slot": 0,
                    "epsilon": 0,
                    "checkpoint_sha256": checkpoint,
                    "native": native,
                }
            ]
            with gzip.open(d / "episodes.json.gz", "wt") as stream:
                json.dump(rows, stream)
            write(
                d / "result.json",
                {
                    "complete": True,
                    "checkpoint_sha256": checkpoint,
                    "rows_sha256": sha(d / "episodes.json.gz"),
                },
            )
    result = analyze(tmp_path)
    assert result["complete"] and not result["submission_promotion"]
    assert not result["eligible_for_further_stability_testing"]
    assert result["gates"]["score_reference"] and not result["gates"]["score_reset"]
    (tmp_path / "evaluation/reference/classic/episodes.json.gz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Corrupt evaluation"):
        analyze(tmp_path)
