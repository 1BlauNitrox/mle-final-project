"""Observation, migration, resume and decision contracts for issue 211."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from agent_code.DagobertDuckDQNObservation.config import DQNConfig
from agent_code.DagobertDuckDQNObservation.features import normalize_features, state_to_features
from agent_code.DagobertDuckDQNObservation.observations import ObservationHistory, hunting_geometry
from agent_code.DagobertDuckDQNObservation.persistence import load_training_checkpoint
from scripts.analyze_observation_screen import classify, interval, loops
from scripts.observation_migration import migrate
from scripts.run_observation_screen import read, resume, save_generation


def state(position=(5, 5), step=1, round_id=1):
    field = np.zeros((17, 17), dtype=int)
    field[[0, -1], :] = -1
    field[:, [0, -1]] = -1
    field[2:16:2, 2:16:2] = -1
    return {
        "round": round_id,
        "step": step,
        "field": field,
        "self": ("me", 0, True, position),
        "others": [("opponent", 0, True, (13, 13))],
        "bombs": [],
        "coins": [],
        "explosion_map": np.zeros_like(field),
    }


def test_geometry_distinguishes_the_aliased_example_and_preserves_prefix():
    first, second = state(), state((7, 5), step=2)
    assert state_to_features(first) == state_to_features(second)
    assert hunting_geometry(first)[0] > hunting_geometry(second)[0]
    history = ObservationHistory()
    value = history.encode(first, DQNConfig(observation_mode="geometry"))
    np.testing.assert_array_equal(
        value[:39],
        normalize_features(state_to_features(first, include_continuation_features=False)),
    )
    assert value.shape == (56,)
    assert np.all(value[51:] == 0)


def test_route_geometry_respects_obstacles_and_no_opponents():
    blocked = state()
    blocked["field"][6, :] = -1
    values = hunting_geometry(blocked)
    assert values[0] == 1 and values[5] == 0
    blocked["others"] = []
    np.testing.assert_array_equal(hunting_geometry(blocked), np.zeros(12))


def test_geometry_describes_safe_escape_counts_without_action_override():
    value = state()
    value["others"] = [("other", 0, True, (7, 5))]
    observation = hunting_geometry(value)
    assert np.all(np.isfinite(observation))
    assert 0 <= observation[-1] <= observation[-2] <= 1


def test_memory_is_causal_idempotent_bounded_and_resets():
    history = ObservationHistory()
    config = DQNConfig(observation_mode="memory")
    first = history.encode(state(), config)
    assert first[51] == 0
    np.testing.assert_array_equal(first, history.encode(state(), config))
    for step in range(2, 22):
        result = history.encode(state(step=step), config)
    assert result[51] == 1.0
    assert len(history.positions) == 16
    assert history.encode(state(round_id=2), config)[51] == 0


def test_training_replay_uses_same_observation_as_act():
    from agent_code.DagobertDuckDQNObservation import callbacks, train

    agent = SimpleNamespace(
        config=DQNConfig(observation_mode="memory"),
        train=True,
        epsilon=0.0,
        action_rng=np.random.default_rng(2),
        policy_network=SimpleNamespace(),
        pending_transition=None,
        episode_event_counts=__import__("collections").Counter(),
    )
    captured = []
    from unittest.mock import patch

    with patch.object(
        callbacks,
        "select_action",
        side_effect=lambda **kw: captured.append(kw["state"].copy()) or "RIGHT",
    ):
        old, new = state(), state((6, 5), step=2)
        callbacks.act(agent, old)
        train.game_events_occurred(agent, old, "RIGHT", new, ["MOVED_RIGHT"])
        pending = agent.pending_transition
        callbacks.act(agent, new)
    np.testing.assert_array_equal(pending.state, captured[0])
    np.testing.assert_array_equal(pending.next_state, captured[1])
    assert pending.next_state[55] == 1 / 16  # LEFT neighbour was the previous tile
    assert len(agent.observation_history.positions) == 2


def test_equal_migration_resets_training_but_preserves_both_networks(tmp_path):
    from agent_code.DagobertDuckDQNTask3.config import DQNConfig as ParentConfig
    from agent_code.DagobertDuckDQNTask3.model import DQNLearner
    from agent_code.DagobertDuckDQNTask3.persistence import save_checkpoint
    from agent_code.DagobertDuckDQNTask3.replay import ReplayBuffer

    parent = tmp_path / "parent.pt"
    config = replace(ParentConfig(), learning_rate=0.0002)
    learner = DQNLearner(config=config, seed=3)
    with torch.no_grad():
        next(learner.target_network.parameters()).add_(0.1)
    save_checkpoint(
        learner=learner,
        replay_buffer=ReplayBuffer(capacity=10000, seed=3),
        action_rng=np.random.default_rng(3),
        epsilon=0.1,
        completed_episodes=8000,
        agent_seed=3,
        path=parent,
    )
    for mode in ("control", "geometry", "memory"):
        destination = tmp_path / f"{mode}.pt"
        migrate(parent, destination, mode, 7)
        loaded = load_training_checkpoint(destination)
        assert loaded.completed_episodes == 0 and len(loaded.replay_buffer) == 0
        assert loaded.learner.update_steps == 0 and not loaded.learner.optimizer.state
        for name in ("online_network", "target_network"):
            network = getattr(loaded.learner, name)
            before = getattr(learner, name)
            torch.testing.assert_close(network.layers[0].weight[:, :39], before.layers[0].weight)
            assert torch.count_nonzero(network.layers[0].weight[:, 39:]) == 0


def test_resume_keeps_matched_generation_when_incomplete_generation_exists(tmp_path):
    directory = tmp_path / "training"
    directory.mkdir()
    checkpoint = directory / "working.pt"
    checkpoint.write_bytes(b"model25")
    save_generation(directory, checkpoint, [{"i": i} for i in range(25)])
    old = read(directory / "resume.json")
    checkpoint.write_bytes(b"partial26")
    (directory / "generation-incomplete").mkdir()
    restored, rows = resume(directory, tmp_path / "unused")
    assert restored.read_bytes() == b"model25" and len(rows) == 25
    assert read(directory / "resume.json") == old


def test_loop_denominator_does_not_reward_missing_late_game_exposure():
    assert loops([{"late_steps": []}])["rate"] is None
    steps = [
        {
            "position": [i % 2, 0],
            "crates_left": 0,
            "coins_visible": 0,
            "hazards": False,
            "progress": False,
            "board_coins_sha256": "same",
        }
        for i in range(30)
    ]
    assert loops([{"late_steps": steps}])["rate"] == 1
    steps[15]["hazards"] = True
    assert loops([{"late_steps": steps}])["eligible_windows"] == 0


def test_uncertainty_and_decisions_do_not_conflate_negative_with_invalid():
    assert interval(np.zeros((3, 8)), resamples=20, seed=1)["ci95"] == [0, 0]
    with pytest.raises(ValueError, match="three"):
        interval(np.zeros((2, 8)), resamples=20, seed=1)
    assert classify({"score_ci": False}, True, False) == "promising_but_unresolved"
    assert classify({"score_ci": False}, False, False) == "inconclusive"
    assert classify({"safety": False}, True, True) == "reject_for_this_deadline"
    assert classify({"safety": True}, True, False) == "eligible_for_confirmation"
