"""Integration tests for the Task 1 framework callbacks."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from agent_code.DerKleineVermoegensumverteiler import callbacks, train
from agent_code.DerKleineVermoegensumverteiler.config import (
    ACTIONS,
    EPSILON_DECAY,
    INITIAL_EPSILON,
    MINIMUM_EPSILON,
)
from agent_code.DerKleineVermoegensumverteiler.features import state_to_features
from agent_code.DerKleineVermoegensumverteiler.rewards import reward_from_events


def make_game_state(
    *,
    position: tuple[int, int] = (3, 3),
    coins: list[tuple[int, int]] | None = None,
) -> dict:
    field = np.zeros((7, 7), dtype=int)
    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    field[:, -1] = -1

    return {
        "round": 1,
        "step": 1,
        "field": field,
        "self": (
            "DerKleineVermoegensumverteiler",
            0,
            True,
            position,
        ),
        "coins": [] if coins is None else coins,
        "bombs": [],
        "others": [],
        "explosion_map": np.zeros_like(field),
    }


def make_agent(*, training: bool) -> SimpleNamespace:
    agent = SimpleNamespace(
        train=training,
        logger=Mock(),
    )

    callbacks.setup(agent)

    if training:
        train.setup_training(agent)

    return agent


def test_evaluation_initializes_with_zero_epsilon() -> None:
    agent = make_agent(training=False)

    assert agent.epsilon == 0.0


def test_training_initializes_with_configured_epsilon() -> None:
    agent = make_agent(training=True)

    assert agent.epsilon == INITIAL_EPSILON


def test_act_returns_only_task1_actions() -> None:
    agent = make_agent(training=True)
    game_state = make_game_state(coins=[(5, 3)])

    selected_actions = {
        callbacks.act(agent, game_state)
        for _ in range(200)
    }

    assert selected_actions <= set(ACTIONS)
    assert "BOMB" not in selected_actions


def test_evaluation_action_sequence_is_seeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BOMBERMAN_AGENT_SEED", "42")

    first_agent = make_agent(training=False)
    second_agent = make_agent(training=False)
    game_state = make_game_state(coins=[(5, 3)])

    first_actions = [
        callbacks.act(first_agent, game_state)
        for _ in range(20)
    ]
    second_actions = [
        callbacks.act(second_agent, game_state)
        for _ in range(20)
    ]

    assert first_actions == second_actions


def test_invalid_agent_seed_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "BOMBERMAN_AGENT_SEED",
        "not-an-integer",
    )

    agent = SimpleNamespace(
        train=False,
        logger=Mock(),
    )

    with pytest.raises(
        ValueError,
        match="must be an integer",
    ):
        callbacks.setup(agent)


def test_ordinary_transition_updates_q_table() -> None:
    agent = make_agent(training=True)

    old_game_state = make_game_state(
        position=(3, 3),
        coins=[(5, 3)],
    )
    new_game_state = make_game_state(
        position=(4, 3),
        coins=[(5, 3)],
    )

    old_state = state_to_features(old_game_state)
    assert old_state is not None

    train.game_events_occurred(
        agent,
        old_game_state,
        "RIGHT",
        new_game_state,
        ["COIN_COLLECTED"],
    )

    assert agent.q_table.q_values(old_state)[1] > 0.0
    assert len(agent.absolute_td_errors) == 1
    assert agent.episode_reward > 0.0


def test_initial_lifecycle_transition_is_ignored() -> None:
    agent = make_agent(training=True)
    new_game_state = make_game_state(coins=[(5, 3)])

    train.game_events_occurred(
        agent,
        None,
        None,
        new_game_state,
        [],
    )

    assert len(agent.q_table) == 0
    assert agent.episode_reward == 0.0


def test_terminal_transition_does_not_bootstrap() -> None:
    agent = make_agent(training=True)
    last_game_state = make_game_state(coins=[(5, 3)])
    last_state = state_to_features(last_game_state)

    assert last_state is not None

    metrics = train.end_of_round(
        agent,
        last_game_state,
        "WAIT",
        ["WAITED"],
    )

    expected_reward = reward_from_events(["WAITED"])
    expected_value = (
        agent.q_table.learning_rate * expected_reward
    )

    wait_index = ACTIONS.index("WAIT")

    assert (
        agent.q_table.q_values(last_state)[wait_index]
        == pytest.approx(expected_value)
    )
    assert metrics["q_table_size"] == 1.0


def test_end_of_round_reports_metrics_and_decays_epsilon() -> None:
    agent = make_agent(training=True)
    last_game_state = make_game_state(coins=[(5, 3)])

    metrics = train.end_of_round(
        agent,
        last_game_state,
        "WAIT",
        ["WAITED"],
    )

    assert metrics["epsilon"] == INITIAL_EPSILON
    assert "shaped_reward" in metrics
    assert "q_table_size" in metrics
    assert "mean_abs_td_error" in metrics

    assert agent.epsilon == pytest.approx(
        max(
            MINIMUM_EPSILON,
            INITIAL_EPSILON * EPSILON_DECAY,
        )
    )


def test_episode_metrics_are_reset_after_round() -> None:
    agent = make_agent(training=True)
    last_game_state = make_game_state(coins=[(5, 3)])

    train.end_of_round(
        agent,
        last_game_state,
        "WAIT",
        ["WAITED"],
    )

    assert agent.episode_reward == 0.0
    assert agent.absolute_td_errors == []