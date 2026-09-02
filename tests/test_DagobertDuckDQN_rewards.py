"""Tests for the Task 1 reward calculation."""

import pytest

from agent_code.DagobertDuckDQN.config import (
    REWARDS,
)
from agent_code.DagobertDuckDQN.rewards import (
    reward_from_events,
)


def test_coin_collection_has_positive_reward() -> None:
    reward = reward_from_events(["COIN_COLLECTED"])

    assert reward == pytest.approx(REWARDS["COIN_COLLECTED"])
    assert reward > 0.0


def test_invalid_action_has_negative_reward() -> None:
    reward = reward_from_events(["INVALID_ACTION"])

    assert reward == pytest.approx(REWARDS["INVALID_ACTION"])
    assert reward < 0.0


def test_waiting_has_small_negative_reward() -> None:
    reward = reward_from_events(["WAITED"])

    assert reward == pytest.approx(REWARDS["WAITED"])
    assert reward < 0.0

    assert abs(reward) < abs(REWARDS["INVALID_ACTION"])


def test_multiple_event_rewards_are_added() -> None:
    reward = reward_from_events(
        [
            "COIN_COLLECTED",
            "WAITED",
        ]
    )

    expected = REWARDS["COIN_COLLECTED"] + REWARDS["WAITED"]

    assert reward == pytest.approx(expected)


def test_repeated_events_are_counted_repeatedly() -> None:
    reward = reward_from_events(
        [
            "WAITED",
            "WAITED",
        ]
    )

    assert reward == pytest.approx(2 * REWARDS["WAITED"])


def test_unknown_events_are_neutral() -> None:
    reward = reward_from_events(
        [
            "SOME_UNKNOWN_EVENT",
            "ANOTHER_UNKNOWN_EVENT",
        ]
    )

    assert reward == 0.0


def test_unknown_events_do_not_hide_known_rewards() -> None:
    reward = reward_from_events(
        [
            "SOME_UNKNOWN_EVENT",
            "COIN_COLLECTED",
        ]
    )

    assert reward == pytest.approx(REWARDS["COIN_COLLECTED"])


def test_empty_event_list_has_zero_reward() -> None:
    assert reward_from_events([]) == 0.0


def test_reward_function_accepts_generators() -> None:
    events = (
        event
        for event in [
            "COIN_COLLECTED",
            "WAITED",
        ]
    )

    reward = reward_from_events(events)

    expected = REWARDS["COIN_COLLECTED"] + REWARDS["WAITED"]

    assert reward == pytest.approx(expected)


def test_mapping_contains_exactly_the_registered_events() -> None:
    assert set(REWARDS) == {
        "COIN_COLLECTED",
        "INVALID_ACTION",
        "WAITED",
        "MOVED_TOWARDS_COIN",
        "MOVED_AWAY_FROM_COIN",
    }


def test_movement_shaping_is_symmetric_and_small() -> None:
    """Symmetry removes the incentive to oscillate for repeated approach reward."""
    assert REWARDS["MOVED_TOWARDS_COIN"] == -REWARDS["MOVED_AWAY_FROM_COIN"]
    assert REWARDS["MOVED_TOWARDS_COIN"] > 0.0
    assert REWARDS["MOVED_TOWARDS_COIN"] < abs(REWARDS["WAITED"]) * 2
    assert REWARDS["MOVED_TOWARDS_COIN"] < REWARDS["COIN_COLLECTED"] / 10
