"""Integration tests for the Task 1 framework callbacks."""

import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from agent_code.DerKleineVermoegensumverteiler import (
    callbacks,
    train,
)
from agent_code.DerKleineVermoegensumverteiler.config import (
    ACTIONS,
    EPSILON_DECAY,
    INITIAL_EPSILON,
    MINIMUM_EPSILON,
)
from agent_code.DerKleineVermoegensumverteiler.features import (
    state_to_features,
)
from agent_code.DerKleineVermoegensumverteiler.model import (
    QTable,
)
from agent_code.DerKleineVermoegensumverteiler.persistence import (
    load_model,
    save_model,
)
from agent_code.DerKleineVermoegensumverteiler.rewards import (
    reward_from_events,
)


@pytest.fixture
def model_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Use an isolated model path for every callback test."""
    path = tmp_path / "model.npz"

    monkeypatch.setattr(
        callbacks,
        "MODEL_PATH",
        path,
    )
    monkeypatch.setattr(
        train,
        "MODEL_PATH",
        path,
    )

    return path


def make_game_state(
    *,
    position: tuple[int, int] = (3, 3),
    coins: list[tuple[int, int]] | None = None,
    step: int = 1,
) -> dict:
    """Create a small framework-compatible game state."""
    field = np.zeros((7, 7), dtype=int)

    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    field[:, -1] = -1

    return {
        "round": 1,
        "step": step,
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
    """Create and initialize a minimal callback agent object."""
    agent = SimpleNamespace(
        train=training,
        logger=Mock(),
    )

    callbacks.setup(agent)

    if training:
        train._initialize_training_state(agent)

    return agent


def test_training_without_model_creates_new_q_table(
    model_path: Path,
) -> None:
    agent = make_agent(training=True)

    assert not model_path.exists()
    assert len(agent.q_table) == 0
    assert agent.epsilon == INITIAL_EPSILON
    assert agent.completed_episodes == 0


def test_evaluation_without_model_is_rejected(
    model_path: Path,
) -> None:
    assert not model_path.exists()

    agent = SimpleNamespace(
        train=False,
        logger=Mock(),
    )

    with pytest.raises(
        FileNotFoundError,
        match="Evaluation model does not exist",
    ):
        callbacks.setup(agent)


def test_evaluation_loads_model_with_zero_epsilon(
    model_path: Path,
) -> None:
    save_model(
        QTable(),
        epsilon=0.8,
        completed_episodes=5,
        path=model_path,
    )

    agent = make_agent(training=False)

    assert agent.epsilon == 0.0
    assert agent.completed_episodes == 5
    assert len(agent.q_table) == 0


def test_training_resumes_saved_state(
    model_path: Path,
) -> None:
    saved_model = QTable()

    state = (
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        2,
    )

    saved_model.update(
        state=state,
        action="RIGHT",
        reward=5.0,
        next_state=None,
        terminal=True,
    )

    save_model(
        saved_model,
        epsilon=0.4,
        completed_episodes=12,
        path=model_path,
    )

    agent = make_agent(training=True)

    assert agent.epsilon == pytest.approx(0.4)
    assert agent.completed_episodes == 12

    np.testing.assert_array_equal(
        agent.q_table.q_values(state),
        saved_model.q_values(state),
    )


def test_invalid_agent_seed_is_rejected(
    model_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert not model_path.exists()

    monkeypatch.setenv(
        "BOMBERMAN_AGENT_SEED",
        "not-an-integer",
    )

    agent = SimpleNamespace(
        train=True,
        logger=Mock(),
    )

    with pytest.raises(
        ValueError,
        match="must be an integer",
    ):
        callbacks.setup(agent)


def test_act_returns_only_task1_actions(
    model_path: Path,
) -> None:
    assert not model_path.exists()

    agent = make_agent(training=True)
    game_state = make_game_state(coins=[(5, 3)])

    selected_actions = {callbacks.act(agent, game_state) for _ in range(200)}

    assert selected_actions <= set(ACTIONS)
    assert "BOMB" not in selected_actions


def test_evaluation_action_sequence_is_seeded(
    model_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save_model(
        QTable(),
        epsilon=0.8,
        completed_episodes=5,
        path=model_path,
    )

    monkeypatch.setenv(
        "BOMBERMAN_AGENT_SEED",
        "42",
    )

    first_agent = make_agent(training=False)
    second_agent = make_agent(training=False)

    game_state = make_game_state(coins=[(5, 3)])

    first_actions = [callbacks.act(first_agent, game_state) for _ in range(20)]
    second_actions = [callbacks.act(second_agent, game_state) for _ in range(20)]

    assert first_actions == second_actions


def test_evaluation_does_not_modify_model(
    model_path: Path,
) -> None:
    saved_model = QTable()

    state = (
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        2,
    )

    saved_model.update(
        state=state,
        action="RIGHT",
        reward=5.0,
        next_state=None,
        terminal=True,
    )

    save_model(
        saved_model,
        epsilon=0.7,
        completed_episodes=3,
        path=model_path,
    )

    bytes_before_evaluation = model_path.read_bytes()

    agent = make_agent(training=False)
    game_state = make_game_state(coins=[(5, 3)])

    for _ in range(20):
        action = callbacks.act(agent, game_state)
        assert action in ACTIONS

    bytes_after_evaluation = model_path.read_bytes()

    assert bytes_after_evaluation == bytes_before_evaluation


def test_initial_lifecycle_transition_is_ignored(
    model_path: Path,
) -> None:
    assert not model_path.exists()

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
    assert agent.absolute_td_errors == []


def test_ordinary_transition_updates_q_table(
    model_path: Path,
) -> None:
    agent = make_agent(training=True)

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

    first_old_state = state_to_features(first_old_game_state)
    assert first_old_state is not None

    train.game_events_occurred(
        agent,
        first_old_game_state,
        "RIGHT",
        first_new_game_state,
        ["COIN_COLLECTED"],
    )

    right_index = ACTIONS.index("RIGHT")

    # The first transition is initially kept pending because the framework
    # might end the round and report the same transition as terminal.
    assert agent.q_table.q_values(first_old_state)[right_index] == pytest.approx(0.0)
    assert agent.pending_transition is not None
    assert agent.absolute_td_errors == []
    assert agent.episode_reward == pytest.approx(0.0)

    second_new_game_state = make_game_state(
        position=(4, 3),
        coins=[(5, 3)],
        step=3,
    )

    train.game_events_occurred(
        agent,
        first_new_game_state,
        "WAIT",
        second_new_game_state,
        ["WAITED"],
    )

    # Receiving the next callback proves that the first transition was not
    # terminal. It must now be finalized as an ordinary update.
    assert agent.q_table.q_values(first_old_state)[right_index] > 0.0
    assert len(agent.absolute_td_errors) == 1
    assert agent.episode_reward == pytest.approx(
        reward_from_events(
            [
                "COIN_COLLECTED",
                "MOVED_TOWARDS_COIN",
            ]
        )
    )

    # The second transition is now pending.
    assert agent.pending_transition is not None
    assert agent.pending_transition.action == "WAIT"


def test_unsupported_training_action_is_ignored(
    model_path: Path,
) -> None:
    assert not model_path.exists()

    agent = make_agent(training=True)

    old_game_state = make_game_state(
        position=(3, 3),
        coins=[(5, 3)],
    )
    new_game_state = make_game_state(
        position=(3, 3),
        coins=[(5, 3)],
    )

    train.game_events_occurred(
        agent,
        old_game_state,
        "BOMB",
        new_game_state,
        [],
    )

    assert len(agent.q_table) == 0
    assert agent.episode_reward == 0.0
    agent.logger.warning.assert_called_once()


def test_terminal_transition_does_not_bootstrap(
    model_path: Path,
) -> None:
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
    expected_value = agent.q_table.learning_rate * expected_reward

    wait_index = ACTIONS.index("WAIT")

    assert agent.q_table.q_values(last_state)[wait_index] == pytest.approx(expected_value)
    assert metrics["q_table_size"] == 1
    assert isinstance(metrics["q_table_size"], int)
    assert model_path.is_file()


def test_end_of_round_reports_metrics_and_decays_epsilon(
    model_path: Path,
) -> None:
    agent = make_agent(training=True)

    last_game_state = make_game_state(coins=[(5, 3)])

    metrics = train.end_of_round(
        agent,
        last_game_state,
        "WAIT",
        ["WAITED"],
    )

    assert metrics["epsilon"] == pytest.approx(INITIAL_EPSILON)
    assert "shaped_reward" in metrics
    assert "q_table_size" in metrics
    assert "mean_abs_td_error" in metrics

    expected_next_epsilon = max(
        MINIMUM_EPSILON,
        INITIAL_EPSILON * EPSILON_DECAY,
    )

    assert agent.epsilon == pytest.approx(expected_next_epsilon)
    assert agent.completed_episodes == 1
    assert model_path.is_file()

    loaded = load_model(model_path)

    assert loaded.epsilon == pytest.approx(expected_next_epsilon)
    assert loaded.completed_episodes == 1


def test_episode_metrics_are_reset_after_round(
    model_path: Path,
) -> None:
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
    assert model_path.is_file()


def test_surviving_final_transition_is_updated_once_as_terminal(
    model_path: Path,
) -> None:
    agent = make_agent(training=True)

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

    train.game_events_occurred(
        agent,
        old_game_state,
        "RIGHT",
        new_game_state,
        ["COIN_COLLECTED"],
    )

    # No ordinary update yet because the transition may be the final one.
    assert agent.q_table.q_values(old_state)[ACTIONS.index("RIGHT")] == pytest.approx(0.0)
    assert agent.pending_transition is not None

    metrics = train.end_of_round(
        agent,
        old_game_state,
        "RIGHT",
        ["COIN_COLLECTED", "SURVIVED_ROUND"],
    )

    expected_reward = reward_from_events(["COIN_COLLECTED", "SURVIVED_ROUND"])
    expected_value = agent.q_table.learning_rate * expected_reward

    assert agent.q_table.q_values(old_state)[ACTIONS.index("RIGHT")] == pytest.approx(
        expected_value
    )
    assert len(agent.absolute_td_errors) == 0
    assert agent.pending_transition is None
    assert metrics["shaped_reward"] == pytest.approx(expected_reward)


def test_death_finalizes_pending_and_death_transitions(
    model_path: Path,
) -> None:
    agent = make_agent(training=True)

    first_old_state = make_game_state(
        position=(3, 3),
        coins=[(5, 3)],
        step=1,
    )
    first_new_state = make_game_state(
        position=(4, 3),
        coins=[(5, 3)],
        step=2,
    )

    death_state = make_game_state(
        position=(4, 3),
        coins=[(5, 3)],
        step=2,
    )

    train.game_events_occurred(
        agent,
        first_old_state,
        "RIGHT",
        first_new_state,
        [],
    )

    assert agent.pending_transition is not None
    assert len(agent.q_table) == 0

    metrics = train.end_of_round(
        agent,
        death_state,
        "WAIT",
        ["WAITED", "GOT_KILLED"],
    )

    first_features = state_to_features(first_old_state)
    death_features = state_to_features(death_state)

    assert first_features is not None
    assert death_features is not None

    # The previous transition was finalized ordinarily.
    assert first_features in agent.q_table.values

    # The separate death transition was finalized terminally.
    expected_death_reward = reward_from_events(["WAITED", "GOT_KILLED"])
    expected_death_value = agent.q_table.learning_rate * expected_death_reward

    assert agent.q_table.q_values(death_features)[ACTIONS.index("WAIT")] == pytest.approx(
        expected_death_value
    )

    assert agent.pending_transition is None

    expected_movement_reward = reward_from_events(["MOVED_TOWARDS_COIN"])

    assert metrics["shaped_reward"] == pytest.approx(
        expected_movement_reward + expected_death_reward
    )


def test_frozen_baseline_rejects_training_without_modifying_model() -> None:
    model_path = train.MODEL_PATH
    hash_before = hashlib.sha256(model_path.read_bytes()).hexdigest()

    agent = SimpleNamespace(
        train=True,
        logger=Mock(),
    )

    with pytest.raises(
        RuntimeError,
        match="frozen Task 1 baseline",
    ):
        train.setup_training(agent)

    hash_after = hashlib.sha256(model_path.read_bytes()).hexdigest()

    assert hash_after == hash_before
    assert hash_after == ("4e1da63a819ef8f51b112ffaf422ab251b853915375fe486538be8595b988307")
