"""Tests for DagobertDuckDQN training callbacks."""

from dataclasses import replace
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


def test_moving_closer_to_a_coin_emits_the_towards_event() -> None:
    # The only coin sits at (5, 3); moving from x=3 to x=4 closes the distance.
    event = training._coin_movement_event(
        make_game_state(step=1, position=(3, 3)),
        make_game_state(step=2, position=(4, 3)),
        "RIGHT",
    )

    assert event == "MOVED_TOWARDS_COIN"


def test_moving_away_from_a_coin_emits_the_away_event() -> None:
    event = training._coin_movement_event(
        make_game_state(step=1, position=(3, 3)),
        make_game_state(step=2, position=(2, 3)),
        "LEFT",
    )

    assert event == "MOVED_AWAY_FROM_COIN"


def test_unchanged_distance_emits_no_movement_event() -> None:
    # With a single coin the distance always changes by one, so an unchanged
    # minimum needs two coins that swap which one is nearest.
    old_state = make_game_state(step=1, position=(3, 3))
    new_state = make_game_state(step=2, position=(3, 4))
    old_state["coins"] = [(5, 3), (5, 4)]
    new_state["coins"] = [(5, 3), (5, 4)]

    assert training._coin_movement_event(old_state, new_state, "DOWN") is None


@pytest.mark.parametrize("action", ["WAIT", "BOMB"])
def test_non_movement_actions_emit_no_movement_event(action: str) -> None:
    event = training._coin_movement_event(
        make_game_state(step=1, position=(3, 3)),
        make_game_state(step=2, position=(4, 3)),
        action,
    )

    assert event is None


def test_no_visible_coin_emits_no_movement_event() -> None:
    old_state = make_game_state(step=1, position=(3, 3))
    new_state = make_game_state(step=2, position=(4, 3))
    old_state["coins"] = []
    new_state["coins"] = []

    assert training._coin_movement_event(old_state, new_state, "RIGHT") is None


def test_shaping_matches_the_tabular_agent_exactly() -> None:
    """#53 compares the two families, so their shaping must be identical.

    Activates once PR #37 lands the tabular shaping; until then the tabular
    agent has no shaping to compare against.
    """
    from agent_code.DerKleineVermoegensumverteiler import train as tabular

    if not hasattr(tabular, "_coin_movement_event"):
        pytest.skip("tabular shaping arrives with PR #37")

    old_state = make_game_state(step=1, position=(3, 3))

    for action, new_position in (
        ("RIGHT", (4, 3)),
        ("LEFT", (2, 3)),
        ("DOWN", (3, 4)),
        ("WAIT", (3, 3)),
    ):
        new_state = make_game_state(step=2, position=new_position)
        assert training._coin_movement_event(
            old_state, new_state, action
        ) == tabular._coin_movement_event(old_state, new_state, action)


def test_shaping_reward_is_added_to_the_pending_transition() -> None:
    agent = make_agent()
    training._initialize_training_state(agent)

    training.game_events_occurred(
        agent,
        make_game_state(step=1, position=(3, 3)),
        "RIGHT",
        make_game_state(step=2, position=(4, 3)),
        [],
    )

    # No framework events, so the reward is the shaping term alone.
    assert agent.pending_transition is not None
    assert agent.pending_transition.reward == pytest.approx(0.1)


def test_initialize_training_state_resets_episode_state() -> None:
    agent = make_agent()

    training._initialize_training_state(agent)

    assert agent.episode_reward == 0.0
    assert agent.absolute_td_errors == []
    assert agent.losses == []
    assert agent.episode_target_synchronizations == 0
    assert agent.pending_transition is None


def test_setup_training_rejects_the_frozen_baseline() -> None:
    agent = make_agent()

    with pytest.raises(RuntimeError, match="frozen Task 1 baseline"):
        training.setup_training(agent)


def test_first_surviving_transition_is_kept_pending() -> None:
    agent = make_agent()
    training._initialize_training_state(agent)

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
    training._initialize_training_state(agent)

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


def test_surviving_final_transition_is_added_once_as_terminal() -> None:
    agent = make_agent()
    training._initialize_training_state(agent)
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


def test_death_finalizes_previous_and_terminal_transitions() -> None:
    agent = make_agent()
    training._initialize_training_state(agent)

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


def test_end_of_round_updates_model_and_returns_metrics() -> None:
    agent = make_agent()
    training._initialize_training_state(agent)
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
    assert metrics["mean_loss"] is not None
    assert metrics["replay_size"] == 1
    assert metrics["update_count"] == 1
    assert metrics["target_synchronizations"] == 0
    assert metrics["episode_target_synchronizations"] == 0
    assert agent.learner.update_steps == 1
    assert agent.completed_episodes == 1
    assert agent.epsilon == pytest.approx(
        DEFAULT_CONFIG.initial_epsilon * DEFAULT_CONFIG.epsilon_decay
    )


def test_episode_state_is_reset_after_round() -> None:
    agent = make_agent()
    training._initialize_training_state(agent)

    training.end_of_round(
        agent,
        None,
        None,
        [],
    )

    assert agent.episode_reward == 0.0
    assert agent.absolute_td_errors == []
    assert agent.losses == []
    assert agent.episode_target_synchronizations == 0
    assert agent.pending_transition is None


def test_round_without_optimizer_update_reports_unavailable_td_error() -> None:
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
    training._initialize_training_state(agent)

    metrics = training.end_of_round(
        agent,
        None,
        None,
        [],
    )

    assert metrics["mean_abs_td_error"] is None
    assert metrics["mean_loss"] is None
