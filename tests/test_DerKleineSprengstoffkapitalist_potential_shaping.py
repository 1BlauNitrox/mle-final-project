"""Tests for compact-state potential-based safety shaping."""

from __future__ import annotations

import numpy as np
import pytest

from agent_code.DerKleineSprengstoffkapitalist.features.bombs_and_crates import (
    build_danger_map,
    safe_escape_exists,
)
from agent_code.DerKleineSprengstoffkapitalist.potential_shaping import (
    COMPACT_SAFETY_POTENTIAL_SHAPING,
    ESCAPE_DISTANCE_POTENTIAL_SHAPING,
    NO_POTENTIAL_SHAPING,
    apply_potential_shaping,
    escape_distance_potential,
    escape_distance_potential_from_distance,
    potential_reward_from_values,
    potential_safety_reward,
    safety_potential,
)

SAFE_STATE = (0, 15, 1, 2, 1)
DANGER_WITH_ESCAPE = (2, 1, 0, 2, 2)
DANGER_WITHOUT_ESCAPE = (2, 0, 0, 2, 2)


def make_game_state(
    *,
    field: np.ndarray | None = None,
    position: tuple[int, int] = (4, 4),
    bombs: list[tuple[tuple[int, int], int]] | None = None,
) -> dict:
    if field is None:
        field = np.zeros((9, 9), dtype=int)
        field[0, :] = -1
        field[-1, :] = -1
        field[:, 0] = -1
        field[:, -1] = -1

    return {
        "field": field,
        "self": ("agent", 0, True, position),
        "coins": [],
        "bombs": [] if bombs is None else bombs,
        "others": [],
        "explosion_map": np.zeros_like(field),
    }


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


def test_disabled_shaping_preserves_original_reward() -> None:
    reward = apply_potential_shaping(
        3.0,
        SAFE_STATE,
        DANGER_WITH_ESCAPE,
        terminal=False,
        mode=NO_POTENTIAL_SHAPING,
    )

    assert reward == pytest.approx(3.0)


def test_compact_safety_mode_adds_potential_reward() -> None:
    reward = apply_potential_shaping(
        3.0,
        SAFE_STATE,
        DANGER_WITH_ESCAPE,
        terminal=False,
        mode=COMPACT_SAFETY_POTENTIAL_SHAPING,
        discount_factor=0.9,
    )

    assert reward == pytest.approx(2.1)


def test_unknown_shaping_mode_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="Potential shaping mode",
    ):
        apply_potential_shaping(
            0.0,
            SAFE_STATE,
            DANGER_WITH_ESCAPE,
            terminal=False,
            mode="unknown",
        )


@pytest.mark.parametrize(
    ("distance", "expected"),
    [
        (0, 0.0),
        (1, -1.0),
        (2, -2.0),
        (3, -3.0),
        (7, -3.0),
        (None, -5.0),
    ],
)
def test_registered_escape_distance_potential_buckets(
    distance: int | None,
    expected: float,
) -> None:
    assert escape_distance_potential_from_distance(distance) == pytest.approx(
        expected
    )


def test_escape_distance_potential_uses_time_aware_route() -> None:
    state = make_game_state(
        position=(4, 5),
        bombs=[((4, 4), 3)],
    )

    assert escape_distance_potential(state) == pytest.approx(-1.0)
    assert escape_distance_potential(state) == pytest.approx(-1.0)


def test_existing_escape_check_still_validates_required_first_direction() -> None:
    state = make_game_state()
    field = state["field"]
    field[4, 3] = -1
    danger_map = build_danger_map(field, [], state["explosion_map"])

    assert not safe_escape_exists(
        field,
        danger_map,
        set(),
        [],
        (4, 4),
        required_first_direction=(0, -1),
    )


def test_escape_distance_potential_reports_no_route() -> None:
    field = np.full((5, 5), -1, dtype=int)
    field[2, 2] = 0
    state = make_game_state(
        field=field,
        position=(2, 2),
        bombs=[((2, 2), 3)],
    )

    assert escape_distance_potential(state) == pytest.approx(-5.0)


def test_escape_progress_and_regress_use_registered_potentials() -> None:
    progress = potential_reward_from_values(
        -2.0,
        -1.0,
        terminal=False,
        discount_factor=0.9,
    )
    regress = potential_reward_from_values(
        -1.0,
        -2.0,
        terminal=False,
        discount_factor=0.9,
    )
    entering_danger = potential_reward_from_values(
        0.0,
        -1.0,
        terminal=False,
        discount_factor=0.9,
    )

    assert progress == pytest.approx(1.1)
    assert regress == pytest.approx(-0.8)
    assert entering_danger == pytest.approx(-0.9)


def test_escape_distance_terminal_potential_is_zero() -> None:
    assert potential_reward_from_values(
        -5.0,
        None,
        terminal=True,
        discount_factor=0.9,
    ) == pytest.approx(5.0)


def test_escape_distance_mode_adds_external_potential_reward() -> None:
    reward = apply_potential_shaping(
        3.0,
        SAFE_STATE,
        DANGER_WITH_ESCAPE,
        terminal=False,
        mode=ESCAPE_DISTANCE_POTENTIAL_SHAPING,
        discount_factor=0.9,
        current_external_potential=-2.0,
        next_external_potential=-1.0,
    )

    assert reward == pytest.approx(4.1)
