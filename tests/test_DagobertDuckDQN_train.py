"""Tests for DagobertDuckDQN training callbacks."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import agent_code.DagobertDuckDQN.train as training
from agent_code.DagobertDuckDQN.config import DEFAULT_CONFIG
from agent_code.DagobertDuckDQN.model import DQNLearner
from agent_code.DagobertDuckDQN.replay import ReplayBuffer


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
        replay_buffer=ReplayBuffer(
            capacity=config.replay_capacity,
            seed=456,
        ),
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
) -> dict:
    """Create one framework-compatible Task 1 state."""
    field = np.zeros((7, 7), dtype=int)
    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    field[:, -1] = -1

    return {
        "round": 1,
        "step": step,
        "field": field,
        "self": ("DagobertDuckDQN", 0, False, position),
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
    assert agent.pending_transition is None


def test_first_surviving_transition_is_kept_pending() -> None:
    agent = make_agent()
    training.setup_training(agent)

    training.game_events_occurred(
        agent,
        make_game_state(step=1),
        "RIGHT",
        make_game_state(step=2, position=(4, 3)),
        [],
    )

    assert len(agent.replay_buffer) == 0
    assert agent.pending_transition is not None


def test_next_callback_finalizes_previous_transition() -> None:
    agent = make_agent()
    training.setup_training(agent)

    training.game_events_occurred(
        agent,
        make_game_state(step=1),
        "RIGHT",
        make_game_state(step=2, position=(4, 3)),
        [],
    )
    training.game_events_occurred(
        agent,
        make_game_state(step=2, position=(4, 3)),
        "RIGHT",
        make_game_state(step=3, position=(5, 3)),
        ["COIN_COLLECTED"],
    )

    replay_state = agent.replay_buffer.state_dict()

    assert len(agent.replay_buffer) == 1
    assert replay_state["terminals"].tolist() == [False]
    assert agent.pending_transition is not None


def test_surviving_final_transition_is_added_once_as_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = make_agent()
    training.setup_training(agent)
    monkeypatch.setattr(
        training,
        "CHECKPOINT_PATH",
        tmp_path / "checkpoint.pt",
    )
    last_state = make_game_state(step=1)

    training.game_events_occurred(
        agent,
        last_state,
        "RIGHT",
        make_game_state(step=2, position=(4, 3)),
        ["COIN_COLLECTED"],
    )
    training.end_of_round(
        agent,
        last_state,
        "RIGHT",
        ["COIN_COLLECTED", "SURVIVED_ROUND"],
    )

    replay_state = agent.replay_buffer.state_dict()

    assert len(agent.replay_buffer) == 1
    assert replay_state["terminals"].tolist() == [True]


def test_death_finalizes_previous_and_terminal_transitions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = make_agent()
    training.setup_training(agent)
    monkeypatch.setattr(
        training,
        "CHECKPOINT_PATH",
        tmp_path / "checkpoint.pt",
    )

    training.game_events_occurred(
        agent,
        make_game_state(step=1),
        "RIGHT",
        make_game_state(step=2, position=(4, 3)),
        [],
    )
    training.end_of_round(
        agent,
        make_game_state(step=2, position=(4, 3)),
        "RIGHT",
        ["GOT_KILLED"],
    )

    replay_state = agent.replay_buffer.state_dict()

    assert len(agent.replay_buffer) == 2
    assert replay_state["terminals"].tolist() == [False, True]


def test_end_of_round_updates_model_and_returns_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = make_agent()
    training.setup_training(agent)
    checkpoint_path = tmp_path / "checkpoint.pt"
    monkeypatch.setattr(
        training,
        "CHECKPOINT_PATH",
        checkpoint_path,
    )
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

    assert metrics["shaped_reward"] == pytest.approx(10.0)
    assert metrics["epsilon"] == pytest.approx(DEFAULT_CONFIG.initial_epsilon)
    assert metrics["mean_abs_td_error"] is not None
    assert agent.learner.update_steps == 1
    assert agent.completed_episodes == 1
    assert agent.epsilon == pytest.approx(
        DEFAULT_CONFIG.initial_epsilon * DEFAULT_CONFIG.epsilon_decay
    )
    assert checkpoint_path.is_file()


def test_episode_state_is_reset_after_round(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = make_agent()
    training.setup_training(agent)
    monkeypatch.setattr(
        training,
        "CHECKPOINT_PATH",
        tmp_path / "checkpoint.pt",
    )

    training.end_of_round(
        agent,
        None,
        None,
        [],
    )

    assert agent.episode_reward == 0.0
    assert agent.absolute_td_errors == []
    assert agent.pending_transition is None


def test_round_without_optimizer_update_reports_unavailable_td_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = make_agent()
    agent.config = replace(
        agent.config,
        batch_size=2,
        replay_warmup=2,
    )
    agent.learner = DQNLearner(
        config=agent.config,
        seed=123,
    )
    agent.replay_buffer = ReplayBuffer(
        capacity=agent.config.replay_capacity,
        seed=456,
    )
    training.setup_training(agent)
    monkeypatch.setattr(
        training,
        "CHECKPOINT_PATH",
        tmp_path / "checkpoint.pt",
    )

    metrics = training.end_of_round(
        agent,
        None,
        None,
        [],
    )

    assert metrics["mean_abs_td_error"] is None
