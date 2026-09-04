"""Tests for the DQN Task 2 successor's training callbacks.

Coin-movement shaping, transition bookkeeping, and the optimizer/replay
plumbing are inherited unchanged from the parent and already covered by
tests/test_DagobertDuckDQN_train.py; this file focuses on what issue #44
actually adds: training being enabled again, the BOMB action flowing
through a six-way action space, the bomb-usefulness shaping event, and the
per-episode event-count diagnostics.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import agent_code.DagobertDuckDQNTask2.callbacks as callbacks
import agent_code.DagobertDuckDQNTask2.train as training
from agent_code.DagobertDuckDQNTask2.config import DEFAULT_CONFIG
from agent_code.DagobertDuckDQNTask2.model import DQNLearner
from agent_code.DagobertDuckDQNTask2.persistence import save_checkpoint
from agent_code.DagobertDuckDQNTask2.replay import ReplayBuffer


def make_config():
    """Create a small configuration that trains after one transition."""
    return replace(
        DEFAULT_CONFIG,
        batch_size=1,
        replay_warmup=1,
        replay_capacity=8,
        target_update_interval=10,
    )


def make_agent() -> SimpleNamespace:
    """Create a minimal initialized training callback object."""
    config = make_config()
    learner = DQNLearner(config=config, seed=123)

    return SimpleNamespace(
        train=True,
        logger=Mock(),
        config=config,
        learner=learner,
        replay_buffer=ReplayBuffer(capacity=config.replay_capacity, seed=456),
        action_rng=np.random.default_rng(789),
        epsilon=config.initial_epsilon,
        completed_episodes=0,
        agent_seed=123,
        policy_network=learner.online_network,
    )


def make_game_state(
    *,
    step: int,
    position: tuple[int, int] = (3, 3),
    bomb_possible: bool = True,
    field: np.ndarray | None = None,
) -> dict:
    """Create one framework-compatible Task 2 state."""
    if field is None:
        field = np.zeros((7, 7), dtype=int)
        field[0, :] = -1
        field[-1, :] = -1
        field[:, 0] = -1
        field[:, -1] = -1

    return {
        "round": 1,
        "step": step,
        "field": field,
        "self": ("DagobertDuckDQNTask2", 0, bomb_possible, position),
        "coins": [(5, 3)],
        "bombs": [],
        "others": [],
        "explosion_map": np.zeros_like(field),
    }


def test_setup_training_initializes_episode_state() -> None:
    agent = make_agent()

    training.setup_training(agent)

    assert agent.episode_reward == 0.0
    assert agent.absolute_td_errors == []
    assert agent.losses == []
    assert agent.episode_target_synchronizations == 0
    assert dict(agent.episode_event_counts) == {}
    assert agent.pending_transition is None


def test_callback_setup_restores_all_training_objects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = make_agent()
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        learner=source.learner,
        replay_buffer=source.replay_buffer,
        action_rng=source.action_rng,
        epsilon=source.epsilon,
        completed_episodes=source.completed_episodes,
        agent_seed=source.agent_seed,
        path=checkpoint_path,
    )
    monkeypatch.setattr(callbacks, "CHECKPOINT_PATH", checkpoint_path)
    restored = SimpleNamespace(logger=Mock())

    callbacks._setup_training_policy(restored, source.agent_seed)

    assert restored.completed_episodes == 0
    assert restored.epsilon == source.epsilon
    assert restored.replay_buffer.capacity == source.replay_buffer.capacity
    assert restored.policy_network is restored.learner.online_network


def test_bomb_placement_on_a_crate_is_useful() -> None:
    field = np.zeros((7, 7), dtype=int)
    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    field[:, -1] = -1
    field[4, 3] = 1  # a crate one tile right of the agent

    old_state = make_game_state(step=1, position=(3, 3), field=field)

    assert training._bomb_usefulness_event(old_state, ["BOMB_DROPPED"]) == "USEFUL_BOMB_PLACED"


def test_bomb_placement_with_no_crate_target_is_wasteful() -> None:
    old_state = make_game_state(step=1, position=(3, 3))

    assert training._bomb_usefulness_event(old_state, ["BOMB_DROPPED"]) == "WASTEFUL_BOMB_PLACED"


def test_bomb_usefulness_event_requires_bomb_dropped() -> None:
    old_state = make_game_state(step=1, position=(3, 3))

    assert training._bomb_usefulness_event(old_state, []) is None
    assert training._bomb_usefulness_event(old_state, ["INVALID_ACTION"]) is None


def test_bomb_transition_carries_the_usefulness_reward() -> None:
    agent = make_agent()
    training.setup_training(agent)

    field = np.zeros((7, 7), dtype=int)
    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    field[:, -1] = -1
    field[4, 3] = 1

    old_state = make_game_state(step=1, position=(3, 3), field=field)
    new_state = make_game_state(step=2, position=(3, 3), field=field, bomb_possible=False)

    training.game_events_occurred(agent, old_state, "BOMB", new_state, ["BOMB_DROPPED"])

    assert agent.pending_transition is not None
    assert agent.pending_transition.action == "BOMB"
    assert agent.pending_transition.reward == pytest.approx(0.5)
    assert agent.episode_event_counts["USEFUL_BOMB_PLACED"] == 1


def test_invalid_bomb_attempt_gets_no_usefulness_shaping() -> None:
    agent = make_agent()
    training.setup_training(agent)

    old_state = make_game_state(step=1, position=(3, 3), bomb_possible=False)
    new_state = make_game_state(step=2, position=(3, 3), bomb_possible=False)

    training.game_events_occurred(agent, old_state, "BOMB", new_state, ["INVALID_ACTION"])

    assert agent.pending_transition is not None
    assert agent.pending_transition.reward == pytest.approx(-0.5)  # INVALID_ACTION only
    assert "USEFUL_BOMB_PLACED" not in agent.episode_event_counts
    assert "WASTEFUL_BOMB_PLACED" not in agent.episode_event_counts


def test_event_counts_are_tallied_across_the_episode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = make_agent()
    training.setup_training(agent)
    monkeypatch.setattr(training, "CHECKPOINT_PATH", tmp_path / "checkpoint.pt")
    last_state = make_game_state(step=1)

    training.game_events_occurred(
        agent,
        last_state,
        "RIGHT",
        make_game_state(step=2, position=(4, 3)),
        ["COIN_COLLECTED"],
    )
    metrics = training.end_of_round(
        agent,
        last_state,
        "RIGHT",
        ["COIN_COLLECTED", "SURVIVED_ROUND"],
    )

    assert metrics["event_count_coin_collected"] == 2.0  # one per callback above
    assert metrics["event_count_survived_round"] == 1.0
    assert metrics["event_count_killed_self"] == 0.0


def test_event_counts_reset_after_round(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = make_agent()
    training.setup_training(agent)
    monkeypatch.setattr(training, "CHECKPOINT_PATH", tmp_path / "checkpoint.pt")

    training.end_of_round(agent, None, None, ["COIN_COLLECTED"])

    assert dict(agent.episode_event_counts) == {}


def test_six_action_transition_updates_the_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A full ordinary-then-terminal cycle through a BOMB action trains the network."""
    agent = make_agent()
    training.setup_training(agent)
    monkeypatch.setattr(training, "CHECKPOINT_PATH", tmp_path / "checkpoint.pt")
    last_state = make_game_state(step=1)

    training.game_events_occurred(
        agent,
        last_state,
        "BOMB",
        make_game_state(step=2, position=(3, 3), bomb_possible=False),
        ["BOMB_DROPPED"],
    )
    metrics = training.end_of_round(
        agent,
        last_state,
        "BOMB",
        ["KILLED_SELF"],
    )

    assert metrics["update_count"] == 1
    assert agent.learner.update_steps == 1
    assert metrics["event_count_killed_self"] == 1.0
