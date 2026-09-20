"""Tests for Issue #224 opponent-kill reward redistribution."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent_code.DerKleineKonkurrenzvernichter import callbacks, train
from agent_code.DerKleineKonkurrenzvernichter.kill_reward import (
    CAUSAL_BOMB_KILL_REWARD,
    redistributed_kill_reward,
)
from agent_code.DerKleineKonkurrenzvernichter.model import QTable
from agent_code.DerKleineKonkurrenzvernichter.persistence import (
    load_model,
    save_model,
)

STATE = (0, 0, 0, 0, 0, 0, 0, 0)
NEXT_STATE = (0, 0, 0, 0, 0, 0, 0, 1)


def _pending_bomb(*, reward: float = 0.0) -> train.PendingTransition:
    return train.PendingTransition(
        identity=(1, 10),
        state=STATE,
        action="BOMB",
        next_state=NEXT_STATE,
        next_action_mask=None,
        reward=reward,
        diagnostic_events=("BOMB_DROPPED",),
        bomb_position=(3, 3),
    )


def _agent() -> SimpleNamespace:
    return SimpleNamespace(
        q_table=QTable(
            learning_rate=1.0,
            discount_factor=0.9,
            feature_count=8,
            initialization="zeros",
        ),
        potential_shaping="none",
        episode_reward=0.0,
        absolute_td_errors=[],
        deferred_bomb_transition=train.DeferredBombTransition(
            transition=_pending_bomb(),
            position=(3, 3),
            origin_step=10,
        ),
    )


def _state(*, step: int, bombs: list[tuple[tuple[int, int], int]]) -> dict:
    return {"round": 1, "step": step, "bombs": bombs}


def test_exact_discounted_reward() -> None:
    assert redistributed_kill_reward(
        kill_count=1,
        delay=4,
        discount_factor=0.9,
        native_reward=5.0,
    ) == pytest.approx(3.2805)


def test_active_bomb_is_not_rewarded_before_outcome() -> None:
    agent = _agent()

    events = train._resolve_deferred_bomb(
        agent,
        old_game_state=_state(step=11, bombs=[((3, 3), 3)]),
        new_game_state=_state(step=12, bombs=[((3, 3), 2)]),
        events=[],
    )

    assert events == []
    assert agent.deferred_bomb_transition is not None
    assert len(agent.q_table) == 0


def test_successful_kill_is_attributed_without_double_counting() -> None:
    agent = _agent()

    reward_events = train._resolve_deferred_bomb(
        agent,
        old_game_state=_state(step=14, bombs=[((3, 3), 0)]),
        new_game_state=_state(step=15, bombs=[]),
        events=["KILLED_OPPONENT", "CRATE_DESTROYED"],
    )

    assert reward_events == ["CRATE_DESTROYED"]
    assert agent.deferred_bomb_transition is None
    assert agent.q_table.q_values(STATE)[5] == pytest.approx(5.0 * 0.9**4)


@pytest.mark.parametrize(
    "events",
    [[], ["CRATE_DESTROYED"], ["KILLED_SELF"]],
)
def test_bomb_without_opponent_kill_receives_no_kill_credit(events: list[str]) -> None:
    agent = _agent()

    reward_events = train._resolve_deferred_bomb(
        agent,
        old_game_state=_state(step=14, bombs=[((3, 3), 0)]),
        new_game_state=_state(step=15, bombs=[]),
        events=events,
    )

    assert reward_events == events
    assert agent.q_table.q_values(STATE)[5] == 0.0


def test_terminal_kill_flushes_bomb_and_removes_native_reward() -> None:
    agent = _agent()

    reward_events = train._resolve_deferred_bomb_at_terminal(
        agent,
        last_game_state=_state(step=14, bombs=[((3, 3), 0)]),
        events=["KILLED_OPPONENT", "SURVIVED_ROUND"],
    )

    assert reward_events == ["SURVIVED_ROUND"]
    assert agent.deferred_bomb_transition is None
    assert agent.q_table.q_values(STATE)[5] == pytest.approx(5.0 * 0.9**4)


def test_kill_reward_mode_round_trip_and_callback_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    path = tmp_path / "model.npz"
    q_table = QTable(feature_count=8, initialization="zeros")
    save_model(
        q_table,
        epsilon=0.5,
        completed_episodes=1,
        state_representation="compact_opponent",
        initialization="zeros",
        kill_reward_mode=CAUSAL_BOMB_KILL_REWARD,
        path=path,
    )

    loaded = load_model(path)
    assert loaded.kill_reward_mode == CAUSAL_BOMB_KILL_REWARD

    monkeypatch.setattr(callbacks, "MODEL_PATH", path)
    monkeypatch.setenv(callbacks.INITIALIZATION_ENV, "zeros")
    monkeypatch.setenv(callbacks.KILL_REWARD_MODE_ENV, "native")
    agent = SimpleNamespace(train=True, logger=Mock())
    with pytest.raises(ValueError, match=callbacks.KILL_REWARD_MODE_ENV):
        callbacks.setup(agent)


def test_invalid_kill_reward_mode_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(callbacks.KILL_REWARD_MODE_ENV, "unknown")
    with pytest.raises(ValueError, match=callbacks.KILL_REWARD_MODE_ENV):
        callbacks._read_kill_reward_mode()
