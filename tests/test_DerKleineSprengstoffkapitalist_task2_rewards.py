"""Tests for the tabular Task 2 reward configuration."""

from __future__ import annotations

import pytest

from agent_code.DerKleineSprengstoffkapitalist.config import (
    REWARDS,
)
from agent_code.DerKleineSprengstoffkapitalist.rewards import (
    reward_from_events,
)


def test_task1_rewards_remain_unchanged() -> None:
    assert REWARDS["COIN_COLLECTED"] == pytest.approx(10.0)
    assert REWARDS["INVALID_ACTION"] == pytest.approx(-0.5)
    assert REWARDS["WAITED"] == pytest.approx(-0.1)

    assert REWARDS["MOVED_TOWARDS_COIN"] == pytest.approx(0.1)
    assert REWARDS["MOVED_AWAY_FROM_COIN"] == pytest.approx(-0.1)


def test_required_task2_rewards_are_centralized() -> None:
    assert REWARDS["CRATE_DESTROYED"] == pytest.approx(1.0)
    assert REWARDS["COIN_FOUND"] == pytest.approx(2.0)
    assert REWARDS["KILLED_SELF"] == pytest.approx(-10.0)
    assert REWARDS["GOT_KILLED"] == pytest.approx(-10.0)
    assert REWARDS["SURVIVED_ROUND"] == pytest.approx(5.0)


def test_complete_reward_mapping() -> None:
    assert REWARDS == {
        "COIN_COLLECTED": 10.0,
        "INVALID_ACTION": -0.5,
        "WAITED": -0.1,
        "MOVED_TOWARDS_COIN": 0.1,
        "MOVED_AWAY_FROM_COIN": -0.1,
        "CRATE_DESTROYED": 1.0,
        "COIN_FOUND": 2.0,
        "KILLED_SELF": -10.0,
        "GOT_KILLED": -10.0,
        "SURVIVED_ROUND": 5.0,
    }


def test_task2_native_events_are_summed() -> None:
    reward = reward_from_events(
        [
            "CRATE_DESTROYED",
            "COIN_FOUND",
            "SURVIVED_ROUND",
        ]
    )

    assert reward == pytest.approx(8.0)


def test_death_penalty_is_combined_with_other_events() -> None:
    reward = reward_from_events(
        [
            "CRATE_DESTROYED",
            "KILLED_SELF",
        ]
    )

    assert reward == pytest.approx(-9.0)


def test_existing_coin_shaping_still_works() -> None:
    toward_reward = reward_from_events(["MOVED_TOWARDS_COIN"])
    away_reward = reward_from_events(["MOVED_AWAY_FROM_COIN"])

    assert toward_reward == pytest.approx(0.1)
    assert away_reward == pytest.approx(-0.1)


def test_bomb_placement_has_no_direct_reward() -> None:
    assert reward_from_events(["BOMB_DROPPED"]) == 0.0
    assert reward_from_events(["BOMB_EXPLODED"]) == 0.0


def test_unknown_events_have_zero_reward() -> None:
    assert reward_from_events(["UNKNOWN_EVENT"]) == 0.0


def test_empty_event_sequence_has_zero_reward() -> None:
    assert reward_from_events([]) == 0.0


def test_reward_function_accepts_generators() -> None:
    events = (
        event
        for event in [
            "CRATE_DESTROYED",
            "COIN_FOUND",
        ]
    )

    assert reward_from_events(events) == pytest.approx(3.0)
