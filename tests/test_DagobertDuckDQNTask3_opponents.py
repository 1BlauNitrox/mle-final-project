"""Tests for the Task 3 public-opponent feature contract."""

from __future__ import annotations

import numpy as np

from agent_code.DagobertDuckDQNTask3.features import state_to_features


def make_field(size: int = 7) -> np.ndarray:
    field = np.zeros((size, size), dtype=int)
    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    field[:, -1] = -1
    return field


def make_state(
    *,
    position: tuple[int, int] = (3, 3),
    others: list[tuple] | None = None,
    field: np.ndarray | None = None,
) -> dict:
    if field is None:
        field = make_field()
    return {
        "round": 1,
        "step": 1,
        "field": field,
        "self": ("task3", 0, True, position),
        "coins": [],
        "bombs": [],
        "others": [] if others is None else others,
        "explosion_map": np.zeros_like(field),
    }


def test_no_opponent_has_a_neutral_suffix() -> None:
    features = state_to_features(make_state())

    assert features is not None
    assert len(features) == 29
    assert features[21:] == (0,) * 8


def test_nearest_opponent_direction_distance_and_tie_break_are_public_only() -> None:
    opponents = [
        ("far", 0, True, (5, 3)),
        ("tie_later_in_order", 0, True, (3, 1)),
        ("tie_earlier_coordinate", 0, True, (1, 3)),
    ]
    features = state_to_features(make_state(others=opponents))

    assert features is not None
    assert features[21:25] == (1, -1, 0, 2)


def test_opponent_occupancy_blocks_movement_and_is_encoded() -> None:
    opponents = [
        ("up", 0, True, (3, 2)),
        ("right", 0, True, (4, 3)),
        ("down", 0, True, (3, 4)),
        ("left", 0, True, (2, 3)),
    ]
    features = state_to_features(make_state(others=opponents))

    assert features is not None
    assert features[:4] == (0, 0, 0, 0)
    assert features[27:] == (3, 3)


def test_attack_opportunity_counts_opponents_in_the_blast() -> None:
    opponents = [
        ("target_a", 0, True, (5, 3)),
        ("target_b", 0, True, (3, 5)),
    ]
    features = state_to_features(make_state(others=opponents))

    assert features is not None
    assert features[25] == 1
    assert features[26] == 1
    assert features[27] == 2


def test_wall_blocks_attack_opportunity() -> None:
    field = make_field()
    field[4, 3] = -1
    features = state_to_features(
        make_state(
            field=field,
            others=[("hidden_behind_wall", 0, True, (5, 3))],
        )
    )

    assert features is not None
    assert features[25:28] == (0, 0, 0)


def test_opponent_blocks_the_only_escape_after_an_attack() -> None:
    field = np.full((5, 5), -1, dtype=int)
    field[2, 2] = 0
    field[3, 2] = 0
    features = state_to_features(
        make_state(
            position=(2, 2),
            field=field,
            others=[("blocker", 0, True, (3, 2))],
        )
    )

    assert features is not None
    assert features[25] == 1
    assert features[26] == 0
