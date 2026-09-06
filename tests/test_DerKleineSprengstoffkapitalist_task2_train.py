"""Tests for the tabular Task 2 training callbacks."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from agent_code.DerKleineSprengstoffkapitalist import callbacks
from agent_code.DerKleineSprengstoffkapitalist import train as training
from agent_code.DerKleineSprengstoffkapitalist.config import (
    ACTIONS,
    EPSILON_DECAY,
    INITIAL_EPSILON,
    MINIMUM_EPSILON,
)
from agent_code.DerKleineSprengstoffkapitalist.features import (
    state_to_features,
)
from agent_code.DerKleineSprengstoffkapitalist.migration import (
    load_parent_prior,
)
from agent_code.DerKleineSprengstoffkapitalist.model import QTable
from agent_code.DerKleineSprengstoffkapitalist.persistence import (
    load_model,
    save_model,
)
from agent_code.DerKleineSprengstoffkapitalist.rewards import (
    reward_from_events,
)


@pytest.fixture
def model_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Redirect all Task 2 model writes into a temporary directory."""

    path = tmp_path / "model.npz"

    monkeypatch.setattr(callbacks, "MODEL_PATH", path)
    monkeypatch.setattr(training, "MODEL_PATH", path)

    return path


def make_field(size: int = 9) -> np.ndarray:
    """Create an open arena surrounded by stone walls."""

    field = np.zeros((size, size), dtype=int)

    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    field[:, -1] = -1

    return field


def make_game_state(
    *,
    position: tuple[int, int] = (4, 4),
    coins: list[tuple[int, int]] | None = None,
    bombs: list[tuple[tuple[int, int], int]] | None = None,
    others: list[tuple] | None = None,
    bomb_available: bool = True,
    step: int = 1,
) -> dict:
    """Create a synthetic framework-compatible game state."""

    field = make_field()

    return {
        "round": 1,
        "step": step,
        "field": field,
        "self": (
            "DerKleineSprengstoffkapitalist",
            0,
            bomb_available,
            position,
        ),
        "coins": [] if coins is None else coins,
        "bombs": [] if bombs is None else bombs,
        "others": [] if others is None else others,
        "explosion_map": np.zeros_like(field),
    }


def make_agent() -> SimpleNamespace:
    """Create a minimal trainable agent without loading a saved model."""

    agent = SimpleNamespace(
        train=True,
        logger=Mock(),
        q_table=QTable(parent_values={}),
        epsilon=INITIAL_EPSILON,
        completed_episodes=0,
    )

    training.setup_training(agent)

    return agent


def test_setup_training_initializes_episode_state() -> None:
    """Training setup must initialize all per-episode attributes."""

    agent = SimpleNamespace(
        logger=Mock(),
        epsilon=INITIAL_EPSILON,
    )

    training.setup_training(agent)

    assert agent.episode_reward == pytest.approx(0.0)
    assert agent.absolute_td_errors == []
    assert agent.pending_transition is None

    agent.logger.info.assert_called_once()


def test_fresh_training_agent_uses_parent_prior(
    model_path: Path,
) -> None:
    """A fresh Task 2 agent must start from the frozen Task 1 prior."""

    assert not model_path.exists()

    agent = SimpleNamespace(
        train=True,
        logger=Mock(),
    )

    callbacks.setup(agent)

    parent_prior = load_parent_prior()

    assert agent.q_table.parent_values.keys() == parent_prior.values.keys()
    assert agent.epsilon == pytest.approx(INITIAL_EPSILON)
    assert agent.completed_episodes == 0

    parent_state = next(iter(parent_prior.values))

    np.testing.assert_array_equal(
        agent.q_table.parent_values[parent_state],
        parent_prior.values[parent_state],
    )


def test_all_six_actions_are_valid_training_actions() -> None:
    """Training must accept all Task 2 actions, including BOMB."""

    for action in ACTIONS:
        agent = make_agent()

        old_game_state = make_game_state(step=1)
        new_game_state = make_game_state(
            bomb_available=action != "BOMB",
            bombs=[((4, 4), 3)] if action == "BOMB" else [],
            step=2,
        )

        training.game_events_occurred(
            agent,
            old_game_state,
            action,
            new_game_state,
            [],
        )

        assert agent.pending_transition is not None
        assert agent.pending_transition.action == action
        agent.logger.warning.assert_not_called()


def test_invalid_training_action_is_ignored() -> None:
    """Unknown actions must not create a pending transition."""

    agent = make_agent()

    training.game_events_occurred(
        agent,
        make_game_state(step=1),
        "UNKNOWN",
        make_game_state(step=2),
        [],
    )

    assert agent.pending_transition is None
    assert len(agent.q_table) == 0
    agent.logger.warning.assert_called_once()


def test_regular_transition_is_updated_on_next_callback() -> None:
    """A surviving transition is finalized when the next callback arrives."""

    agent = make_agent()

    first_old_game_state = make_game_state(
        position=(3, 3),
        coins=[(5, 3)],
        step=1,
    )
    first_new_game_state = make_game_state(
        position=(4, 3),
        coins=[(5, 3)],
        step=2,
    )

    first_state = state_to_features(first_old_game_state)

    assert first_state is not None

    right_index = ACTIONS.index("RIGHT")
    old_value = agent.q_table.q_values(first_state)[right_index]

    training.game_events_occurred(
        agent,
        first_old_game_state,
        "RIGHT",
        first_new_game_state,
        [],
    )

    # The transition remains pending until another callback proves that it
    # was not the terminal transition.
    assert agent.pending_transition is not None
    assert len(agent.absolute_td_errors) == 0
    assert agent.q_table.q_values(first_state)[right_index] == pytest.approx(old_value)

    second_new_game_state = make_game_state(
        position=(4, 3),
        coins=[(5, 3)],
        step=3,
    )

    training.game_events_occurred(
        agent,
        first_new_game_state,
        "WAIT",
        second_new_game_state,
        ["WAITED"],
    )

    expected_reward = reward_from_events(["MOVED_TOWARDS_COIN"])

    assert first_state in agent.q_table.values
    assert len(agent.absolute_td_errors) == 1
    assert agent.episode_reward == pytest.approx(expected_reward)

    assert agent.q_table.q_values(first_state)[right_index] > old_value

    # The second transition is now waiting to be finalized.
    assert agent.pending_transition is not None
    assert agent.pending_transition.action == "WAIT"


def test_bomb_transition_updates_bomb_q_value() -> None:
    """The new BOMB action must participate in ordinary Q-learning updates."""

    agent = make_agent()

    old_game_state = make_game_state(
        position=(4, 4),
        bomb_available=True,
        step=1,
    )
    new_game_state = make_game_state(
        position=(4, 4),
        bomb_available=False,
        bombs=[((4, 4), 3)],
        step=2,
    )

    old_state = state_to_features(old_game_state)

    assert old_state is not None

    bomb_index = ACTIONS.index("BOMB")
    old_bomb_value = agent.q_table.q_values(old_state)[bomb_index]

    training.game_events_occurred(
        agent,
        old_game_state,
        "BOMB",
        new_game_state,
        ["BOMB_DROPPED"],
    )

    assert agent.pending_transition is not None
    assert agent.pending_transition.action == "BOMB"
    assert agent.pending_transition.reward == pytest.approx(0.0)

    following_game_state = make_game_state(
        position=(4, 4),
        bomb_available=False,
        bombs=[((4, 4), 2)],
        step=3,
    )

    training.game_events_occurred(
        agent,
        new_game_state,
        "WAIT",
        following_game_state,
        ["WAITED"],
    )

    new_bomb_value = agent.q_table.q_values(old_state)[bomb_index]

    assert old_state in agent.q_table.values
    assert new_bomb_value != pytest.approx(old_bomb_value)
    assert len(agent.absolute_td_errors) == 1


def test_terminal_transition_does_not_bootstrap(
    model_path: Path,
) -> None:
    """A terminal update must use only the terminal reward."""

    agent = make_agent()
    agent.q_table = QTable(
        learning_rate=1.0,
        discount_factor=0.9,
        parent_values={},
    )

    last_game_state = make_game_state(step=1)
    last_state = state_to_features(last_game_state)

    assert last_state is not None

    # A large value for another action would affect the update if terminal
    # transitions incorrectly used a bootstrap value.
    current_values = np.zeros(len(ACTIONS), dtype=float)
    current_values[ACTIONS.index("RIGHT")] = 100.0
    agent.q_table.values[last_state] = current_values

    metrics = training.end_of_round(
        agent,
        last_game_state,
        "WAIT",
        ["GOT_KILLED"],
    )

    expected_reward = reward_from_events(["GOT_KILLED"])
    learned_wait_value = agent.q_table.q_values(last_state)[ACTIONS.index("WAIT")]

    assert learned_wait_value == pytest.approx(expected_reward)
    assert metrics["shaped_reward"] == pytest.approx(expected_reward)
    assert model_path.is_file()


def test_final_pending_transition_is_updated_exactly_once(
    model_path: Path,
) -> None:
    """The final game callback must not cause a duplicate Q update."""

    agent = make_agent()

    old_game_state = make_game_state(
        position=(3, 3),
        coins=[(4, 3)],
        step=1,
    )
    new_game_state = make_game_state(
        position=(4, 3),
        coins=[],
        step=2,
    )

    old_state = state_to_features(old_game_state)

    assert old_state is not None

    training.game_events_occurred(
        agent,
        old_game_state,
        "RIGHT",
        new_game_state,
        ["COIN_COLLECTED"],
    )

    assert agent.pending_transition is not None
    assert len(agent.absolute_td_errors) == 0

    metrics = training.end_of_round(
        agent,
        old_game_state,
        "RIGHT",
        [
            "COIN_COLLECTED",
            "SURVIVED_ROUND",
        ],
    )

    expected_reward = reward_from_events(
        [
            "COIN_COLLECTED",
            "SURVIVED_ROUND",
        ]
    )
    expected_value = agent.q_table.learning_rate * expected_reward

    learned_value = agent.q_table.q_values(old_state)[ACTIONS.index("RIGHT")]

    assert learned_value == pytest.approx(expected_value)
    assert metrics["shaped_reward"] == pytest.approx(expected_reward)
    assert metrics["mean_abs_td_error"] == pytest.approx(abs(expected_reward))
    assert agent.pending_transition is None
    assert model_path.is_file()


def test_end_of_round_saves_model_and_decays_epsilon(
    model_path: Path,
) -> None:
    """Completing a round must save all resumable training state."""

    agent = make_agent()
    last_game_state = make_game_state(step=1)

    metrics = training.end_of_round(
        agent,
        last_game_state,
        "WAIT",
        [
            "WAITED",
            "SURVIVED_ROUND",
        ],
    )

    expected_reward = reward_from_events(
        [
            "WAITED",
            "SURVIVED_ROUND",
        ]
    )
    expected_epsilon = max(
        MINIMUM_EPSILON,
        INITIAL_EPSILON * EPSILON_DECAY,
    )

    assert metrics["shaped_reward"] == pytest.approx(expected_reward)
    assert metrics["epsilon"] == pytest.approx(INITIAL_EPSILON)
    assert isinstance(metrics["q_table_size"], int)
    assert metrics["mean_abs_td_error"] >= 0.0

    assert agent.completed_episodes == 1
    assert agent.epsilon == pytest.approx(expected_epsilon)
    assert agent.episode_reward == pytest.approx(0.0)
    assert agent.absolute_td_errors == []
    assert agent.pending_transition is None
    assert model_path.is_file()

    loaded = load_model(model_path)

    assert loaded.completed_episodes == 1
    assert loaded.epsilon == pytest.approx(expected_epsilon)


def test_callbacks_resume_saved_training_state(
    model_path: Path,
) -> None:
    """Training setup must resume epsilon, episode count and Q-values."""

    parent_prior = load_parent_prior()
    source_table = QTable(parent_values=parent_prior.values)

    game_state = make_game_state(step=1)
    state = state_to_features(game_state)

    assert state is not None

    source_table.update(
        state=state,
        action="BOMB",
        reward=3.0,
        next_state=None,
        terminal=True,
    )

    save_model(
        source_table,
        epsilon=0.42,
        completed_episodes=17,
        path=model_path,
    )

    restored_agent = SimpleNamespace(
        train=True,
        logger=Mock(),
    )

    callbacks.setup(restored_agent)

    assert restored_agent.completed_episodes == 17
    assert restored_agent.epsilon == pytest.approx(0.42)

    np.testing.assert_array_equal(
        restored_agent.q_table.q_values(state),
        source_table.q_values(state),
    )


def test_end_of_round_returns_complete_episode_metrics(
    model_path: Path,
) -> None:
    agent = make_agent()

    old_game_state = make_game_state(
        position=(4, 4),
        bomb_available=True,
        step=1,
    )

    # Eine Kiste direkt über dem Agenten macht die Bombe nützlich.
    old_game_state["field"][4, 3] = 1

    new_game_state = make_game_state(
        position=(4, 4),
        bomb_available=False,
        bombs=[((4, 4), 3)],
        step=2,
    )
    new_game_state["field"][4, 3] = 1

    training.game_events_occurred(
        agent,
        old_game_state,
        "BOMB",
        new_game_state,
        [
            "BOMB_DROPPED",
            "COIN_FOUND",
        ],
    )

    metrics = training.end_of_round(
        agent,
        new_game_state,
        "WAIT",
        [
            "CRATE_DESTROYED",
            "COIN_COLLECTED",
            "SURVIVED_ROUND",
        ],
    )

    assert metrics["coins_collected"] == pytest.approx(1.0)
    assert metrics["coins_found"] == pytest.approx(1.0)
    assert metrics["crates_destroyed"] == pytest.approx(1.0)
    assert metrics["bombs_dropped"] == pytest.approx(1.0)
    assert metrics["useful_bombs"] == pytest.approx(1.0)
    assert metrics["self_kills"] == pytest.approx(0.0)
    assert metrics["survived_round"] == pytest.approx(1.0)
    assert metrics["invalid_actions"] == pytest.approx(0.0)

    assert "shaped_reward" in metrics
    assert "q_table_size" in metrics
    assert "mean_abs_td_error" in metrics
    assert "epsilon" in metrics

    assert agent.episode_event_counts == Counter()
    assert model_path.is_file()


def test_invalid_actions_are_counted(
    model_path: Path,
) -> None:
    agent = make_agent()

    training.game_events_occurred(
        agent,
        make_game_state(step=1),
        "BOMB",
        make_game_state(step=2),
        ["INVALID_ACTION"],
    )

    metrics = training.end_of_round(
        agent,
        make_game_state(step=2),
        "WAIT",
        ["GOT_KILLED"],
    )

    assert metrics["invalid_actions"] == pytest.approx(1.0)
    assert metrics["bombs_dropped"] == pytest.approx(0.0)
    assert metrics["useful_bombs"] == pytest.approx(0.0)

def test_matching_survivor_callback_counts_final_step_events_once(
    model_path: Path,
) -> None:
    agent = make_agent()

    old_game_state = make_game_state(step=1)
    new_game_state = make_game_state(step=2)

    final_step_events = [
        "COIN_COLLECTED",
        "COIN_FOUND",
        "CRATE_DESTROYED",
        "BOMB_DROPPED",
        "INVALID_ACTION",
    ]

    training.game_events_occurred(
        agent,
        old_game_state,
        "WAIT",
        new_game_state,
        final_step_events,
    )

    metrics = training.end_of_round(
        agent,
        old_game_state,
        "WAIT",
        [*final_step_events, "SURVIVED_ROUND"],
    )

    assert metrics["coins_collected"] == pytest.approx(1.0)
    assert metrics["coins_found"] == pytest.approx(1.0)
    assert metrics["crates_destroyed"] == pytest.approx(1.0)
    assert metrics["bombs_dropped"] == pytest.approx(1.0)
    assert metrics["invalid_actions"] == pytest.approx(1.0)
    assert metrics["survived_round"] == pytest.approx(1.0)