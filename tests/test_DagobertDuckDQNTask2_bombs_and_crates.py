"""Unit tests for blast-danger, escape, and crate-targeting geometry."""

from __future__ import annotations

import numpy as np
import pytest

from agent_code.DagobertDuckDQNTask2.features.bombs_and_crates import (
    BOMB_POWER,
    blast_footprint,
    build_danger_map,
    crates_destroyed_by_bomb_at,
    danger_countdown_bin,
    is_safe_at_arrival,
    nearest_crate_features,
    safe_direction,
    safe_escape_exists,
)


def make_field(size: int = 9) -> np.ndarray:
    """An open square arena bordered by walls, no crates."""
    field = np.zeros((size, size), dtype=int)
    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    field[:, -1] = -1
    return field


def test_blast_footprint_stops_at_walls() -> None:
    field = make_field()
    footprint = blast_footprint((4, 4), field)

    # Power 3 in each direction from (4, 4), bounded by the walls at 0 and 8.
    assert (1, 4) in footprint
    assert (7, 4) in footprint
    assert (0, 4) not in footprint
    assert (8, 4) not in footprint


def test_blast_footprint_passes_through_crates() -> None:
    """Matches items.Bomb.get_blast_coords: only walls (-1) stop propagation."""
    field = make_field()
    field[5, 4] = 1  # a crate one tile right of the bomb
    footprint = blast_footprint((4, 4), field)

    assert (5, 4) in footprint
    assert (6, 4) in footprint  # blast continues past the crate


def test_build_danger_map_uses_the_minimum_timer() -> None:
    field = make_field()
    danger = build_danger_map(field, [((4, 4), 3), ((4, 4), 1)], np.zeros_like(field))

    assert danger[(4, 4)] == 1


def test_build_danger_map_includes_active_explosions() -> None:
    field = make_field()
    explosion_map = np.zeros_like(field)
    explosion_map[4, 4] = 1

    danger = build_danger_map(field, [], explosion_map)

    assert danger[(4, 4)] == 0


def test_is_safe_at_arrival_covers_the_explosion_and_its_linger() -> None:
    danger_map = {(4, 4): 2}

    assert is_safe_at_arrival(danger_map, (4, 4), 1) is True
    assert is_safe_at_arrival(danger_map, (4, 4), 2) is False  # detonates
    assert is_safe_at_arrival(danger_map, (4, 4), 4) is False  # still lingering
    assert is_safe_at_arrival(danger_map, (4, 4), 5) is True  # linger has cleared


def test_is_safe_at_arrival_defaults_to_safe_when_untracked() -> None:
    assert is_safe_at_arrival({}, (0, 0), 0) is True


@pytest.mark.parametrize(
    ("countdown", "expected_bin"),
    [(None, 0), (0, 1), (1, 2), (2, 2), (3, 3), (10, 3)],
)
def test_danger_countdown_bin(countdown: int | None, expected_bin: int) -> None:
    danger_map = {} if countdown is None else {(4, 4): countdown}
    assert danger_countdown_bin(danger_map, (4, 4)) == expected_bin


def test_safe_direction_rejects_a_wall() -> None:
    field = make_field()
    assert safe_direction(field, {}, set(), (1, 4), (-1, 0)) is False


def test_safe_direction_rejects_a_lethal_neighbor() -> None:
    field = make_field()
    danger_map = {(5, 4): 0}
    assert safe_direction(field, danger_map, set(), (4, 4), (1, 0)) is False


def test_safe_direction_rejects_a_blocked_neighbor() -> None:
    field = make_field()
    assert safe_direction(field, {}, {(5, 4)}, (4, 4), (1, 0)) is False


def test_safe_direction_accepts_an_open_safe_neighbor() -> None:
    field = make_field()
    assert safe_direction(field, {}, set(), (4, 4), (1, 0)) is True


def test_safe_escape_exists_when_far_from_any_bomb() -> None:
    field = make_field()
    danger_map = build_danger_map(field, [((1, 1), 4)], np.zeros_like(field))

    assert safe_escape_exists(field, danger_map, set(), (7, 7)) is True


def test_safe_escape_exists_is_false_in_a_sealed_deadly_box() -> None:
    """A 1x1 pocket with a bomb right outside every exit has no escape."""
    field = np.full((5, 5), -1, dtype=int)
    field[2, 2] = 0  # the agent's only tile

    danger_map = {(2, 2): 0}

    assert safe_escape_exists(field, danger_map, set(), (2, 2)) is False


def test_safe_escape_exists_with_time_to_walk_away() -> None:
    field = make_field()
    # A freshly placed bomb (timer 3) leaves time to walk clear of the
    # footprint before it detonates, even starting on the bomb's own tile.
    danger_map = build_danger_map(field, [((4, 4), 3)], np.zeros_like(field))

    assert safe_escape_exists(field, danger_map, set(), (4, 4)) is True


def test_safe_escape_exists_is_false_with_no_time_and_no_side_exit() -> None:
    """Timer 0: every cardinal neighbor of the bomb's own tile is also in the
    blast line and equally lethal, so there is no single step to safety."""
    field = make_field()
    danger_map = build_danger_map(field, [((4, 4), 0)], np.zeros_like(field))

    assert safe_escape_exists(field, danger_map, set(), (4, 4)) is False


def test_crates_destroyed_by_bomb_at_counts_only_crates_in_the_footprint() -> None:
    field = make_field()
    field[5, 4] = 1
    field[6, 4] = 1
    field[4, 5] = 1

    assert crates_destroyed_by_bomb_at((4, 4), field) == 3


def test_crates_destroyed_by_bomb_at_is_zero_with_no_crates() -> None:
    field = make_field()
    assert crates_destroyed_by_bomb_at((4, 4), field) == 0


def test_nearest_crate_features_absent() -> None:
    field = make_field()
    assert nearest_crate_features(position=(4, 4), field=field) == (0, 0, 0, 0)


def test_nearest_crate_features_direction_and_distance() -> None:
    field = make_field()
    field[6, 4] = 1
    field[2, 2] = 1  # farther away, should not be selected

    visible, dx, dy, distance_bin = nearest_crate_features(position=(4, 4), field=field)

    assert visible == 1
    assert (dx, dy) == (1, 0)
    assert distance_bin == 2  # Manhattan distance 2


def test_blast_footprint_bounded_by_bomb_power() -> None:
    field = np.zeros((15, 15), dtype=int)
    footprint = blast_footprint((7, 7), field)

    assert (7, 7 + BOMB_POWER) in footprint
    assert (7, 7 + BOMB_POWER + 1) not in footprint
