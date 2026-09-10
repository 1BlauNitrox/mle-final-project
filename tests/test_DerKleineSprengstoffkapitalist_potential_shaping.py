"""Tests for compact-state potential-based safety shaping."""

from __future__ import annotations

import pytest

from agent_code.DerKleineSprengstoffkapitalist.potential_shaping import (
    potential_safety_reward,
    safety_potential,
)

SAFE_STATE = (0, 15, 1, 2, 1)
DANGER_WITH_ESCAPE = (2, 1, 0, 2, 2)
DANGER_WITHOUT_ESCAPE = (2, 0, 0, 2, 2)


def test_registered_safety_potential() -> None:
    assert safety_potential(SAFE_STATE) == pytest.approx(0.0)
    assert safety_potential(
        DANGER_WITH_ESCAPE
    ) == pytest.approx(-1.0)
    assert safety_potential(
        DANGER_WITHOUT_ESCAPE
    ) == pytest.approx(-2.0)


def test_entering_danger_has_negative_shaping_reward() -> None:
    reward = potential_safety_reward(
        SAFE_STATE,
        DANGER_WITH_ESCAPE,
        terminal=False,
        discount_factor=0.9,
    )

    assert reward == pytest.approx(-0.9)


def test_leaving_danger_has_positive_shaping_reward() -> None:
    reward = potential_safety_reward(
        DANGER_WITH_ESCAPE,
        SAFE_STATE,
        terminal=False,
        discount_factor=0.9,
    )

    assert reward == pytest.approx(1.0)


def test_terminal_state_has_zero_potential() -> None:
    reward = potential_safety_reward(
        DANGER_WITHOUT_ESCAPE,
        None,
        terminal=True,
        discount_factor=0.9,
    )

    assert reward == pytest.approx(2.0)


def test_non_terminal_transition_requires_next_state() -> None:
    with pytest.raises(
        ValueError,
        match="require a next state",
    ):
        potential_safety_reward(
            SAFE_STATE,
            None,
            terminal=False,
        )


def test_discounted_shaping_rewards_telescope() -> None:
    discount_factor = 0.9

    first_reward = potential_safety_reward(
        DANGER_WITH_ESCAPE,
        DANGER_WITHOUT_ESCAPE,
        terminal=False,
        discount_factor=discount_factor,
    )
    terminal_reward = potential_safety_reward(
        DANGER_WITHOUT_ESCAPE,
        None,
        terminal=True,
        discount_factor=discount_factor,
    )

    discounted_total = (
        first_reward
        + discount_factor * terminal_reward
    )

    assert discounted_total == pytest.approx(
        -safety_potential(DANGER_WITH_ESCAPE)
    )