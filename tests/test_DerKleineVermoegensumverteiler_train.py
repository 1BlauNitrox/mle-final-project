"""Tests for the DerKleineVermoegensumverteiler shaping events."""

from __future__ import annotations

import numpy as np
import pytest

import agent_code.DerKleineVermoegensumverteiler.train as training


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
        "self": ("DerKleineVermoegensumverteiler", 0, False, position),
        "coins": [(5, 3)] if coins is None else coins,
        "bombs": [],
        "others": [],
        "explosion_map": np.zeros_like(field),
    }


def test_moving_closer_to_a_coin_emits_the_towards_event() -> None:
    # The only coin sits at (5, 3); moving from x=3 to x=4 closes the distance.
    event = training._coin_movement_event(
        make_game_state(position=(3, 3)),
        make_game_state(position=(4, 3), step=2),
        "RIGHT",
    )

    assert event == "MOVED_TOWARDS_COIN"


def test_moving_away_from_a_coin_emits_the_away_event() -> None:
    event = training._coin_movement_event(
        make_game_state(position=(3, 3)),
        make_game_state(position=(2, 3), step=2),
        "LEFT",
    )

    assert event == "MOVED_AWAY_FROM_COIN"


def test_unchanged_distance_emits_no_movement_event() -> None:
    # With one coin the distance always changes by one, so an unchanged minimum
    # needs two coins that swap which is nearest.
    coins = [(5, 3), (5, 4)]
    event = training._coin_movement_event(
        make_game_state(position=(3, 3), coins=coins),
        make_game_state(position=(3, 4), coins=coins, step=2),
        "DOWN",
    )

    assert event is None


@pytest.mark.parametrize("action", ["WAIT", "BOMB"])
def test_non_movement_actions_emit_no_movement_event(action: str) -> None:
    event = training._coin_movement_event(
        make_game_state(position=(3, 3)),
        make_game_state(position=(4, 3), step=2),
        action,
    )

    assert event is None


def test_no_visible_coin_emits_no_movement_event() -> None:
    event = training._coin_movement_event(
        make_game_state(position=(3, 3), coins=[]),
        make_game_state(position=(4, 3), coins=[], step=2),
        "RIGHT",
    )

    assert event is None


def test_collected_coin_still_scores_the_step_that_reached_it() -> None:
    """Both distances use the old coin list, so the final step is not lost."""
    event = training._coin_movement_event(
        make_game_state(position=(4, 3), coins=[(5, 3)]),
        make_game_state(position=(5, 3), coins=[], step=2),
        "RIGHT",
    )

    assert event == "MOVED_TOWARDS_COIN"


def test_shaping_reward_reaches_the_pending_transition() -> None:
    from types import SimpleNamespace

    from agent_code.DerKleineVermoegensumverteiler.config import REWARDS
    from agent_code.DerKleineVermoegensumverteiler.model import QTable

    agent = SimpleNamespace(
        q_table=QTable(),
        epsilon=0.0,
        logger=__import__("logging").getLogger("shaping-test"),
    )
    training.setup_training(agent)

    training.game_events_occurred(
        agent,
        make_game_state(position=(3, 3)),
        "RIGHT",
        make_game_state(position=(4, 3), step=2),
        [],
    )

    # No framework events, so the reward is the shaping term alone.
    assert agent.pending_transition is not None
    assert agent.pending_transition.reward == pytest.approx(
        REWARDS["MOVED_TOWARDS_COIN"]
    )