"""Unit tests for blast-danger, escape, and crate-targeting geometry."""

from __future__ import annotations

import numpy as np
import pytest

from agent_code.DerKleineSprengstoffkapitalist.features.bombs_and_crates import (
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


def test_blast_footprint_uses_framework_power() -> None:
    field = make_field(size=11)

    footprint = blast_footprint((5, 5), field)

    assert BOMB_POWER == 3
    assert len(footprint) == 13

    assert (8, 5) in footprint
    assert (2, 5) in footprint
    assert (5, 8) in footprint
    assert (5, 2) in footprint

    assert (9, 5) not in footprint
    assert (1, 5) not in footprint
    assert (5, 9) not in footprint
    assert (5, 1) not in footprint


def test_blast_footprint_does_not_include_diagonals() -> None:
    field = make_field()
    footprint = blast_footprint((4, 4), field)

    assert (3, 3) not in footprint
    assert (5, 3) not in footprint
    assert (3, 5) not in footprint
    assert (5, 5) not in footprint


def test_crates_behind_a_wall_are_not_destroyed() -> None:
    field = make_field()

    field[5, 4] = -1
    field[6, 4] = 1

    footprint = blast_footprint((4, 4), field)

    assert (5, 4) not in footprint
    assert (6, 4) not in footprint
    assert crates_destroyed_by_bomb_at((4, 4), field) == 0


def test_blast_footprint_passes_through_crates() -> None:
    """Matches items.Bomb.get_blast_coords: only walls (-1) stop propagation."""
    field = make_field()
    field[5, 4] = 1  # a crate one tile right of the bomb
    footprint = blast_footprint((4, 4), field)

    assert (5, 4) in footprint
    assert (6, 4) in footprint  # blast continues past the crate


def test_build_danger_map_preserves_separate_overlapping_bomb_windows() -> None:
    field = make_field()
    danger = build_danger_map(
        field,
        [((4, 2), 0), ((4, 6), 4)],
        np.zeros_like(field),
    )

    assert danger[(4, 4)] == ((1, 2), (5, 6))
    assert is_safe_at_arrival(danger, (4, 4), 4) is True
    assert is_safe_at_arrival(danger, (4, 4), 5) is False


def test_build_danger_map_includes_active_explosions() -> None:
    field = make_field()
    explosion_map = np.zeros_like(field)
    explosion_map[4, 4] = 1

    danger = build_danger_map(field, [], explosion_map)

    assert danger[(4, 4)] == ((0, 1),)


def test_active_explosion_does_not_hide_a_later_bomb_window() -> None:
    field = make_field()
    explosion_map = np.zeros_like(field)
    explosion_map[4, 4] = 1

    danger = build_danger_map(field, [((4, 6), 4)], explosion_map)

    assert danger[(4, 4)] == ((0, 1), (5, 6))
    assert is_safe_at_arrival(danger, (4, 4), 5) is False


def test_is_safe_at_arrival_covers_the_explosion_and_its_linger() -> None:
    field = make_field()
    danger_map = build_danger_map(
        field,
        [((4, 4), 3)],
        np.zeros_like(field),
    )

    assert is_safe_at_arrival(danger_map, (4, 4), 1) is True
    assert is_safe_at_arrival(danger_map, (4, 4), 3) is True
    assert is_safe_at_arrival(danger_map, (4, 4), 4) is False
    assert is_safe_at_arrival(danger_map, (4, 4), 5) is False
    assert is_safe_at_arrival(danger_map, (4, 4), 6) is True


def test_is_safe_at_arrival_defaults_to_safe_when_untracked() -> None:
    assert is_safe_at_arrival({}, (0, 0), 0) is True


@pytest.mark.parametrize(
    ("start", "expected_bin"),
    [(None, 0), (0, 1), (1, 2), (2, 2), (3, 3), (10, 3)],
)
def test_danger_countdown_bin(start: int | None, expected_bin: int) -> None:
    danger_map = {} if start is None else {(4, 4): ((start, start + 1),)}
    assert danger_countdown_bin(danger_map, (4, 4)) == expected_bin


def test_safe_direction_rejects_a_wall() -> None:
    field = make_field()
    assert safe_direction(field, {}, set(), (1, 4), (-1, 0)) is False


def test_safe_direction_rejects_a_lethal_neighbor() -> None:
    field = make_field()
    danger_map = {(5, 4): ((1, 2),)}
    assert safe_direction(field, danger_map, set(), (4, 4), (1, 0)) is False


def test_safe_direction_rejects_a_blocked_neighbor() -> None:
    field = make_field()
    assert safe_direction(field, {}, {(5, 4)}, (4, 4), (1, 0)) is False


def test_safe_direction_accepts_an_open_safe_neighbor() -> None:
    field = make_field()
    assert safe_direction(field, {}, set(), (4, 4), (1, 0)) is True


def test_safe_escape_exists_when_far_from_any_bomb() -> None:
    field = make_field()
    bombs = [((1, 1), 4)]
    danger_map = build_danger_map(field, bombs, np.zeros_like(field))

    assert safe_escape_exists(field, danger_map, set(), bombs, (7, 7)) is True


def test_safe_escape_exists_is_false_in_a_sealed_deadly_box() -> None:
    """A 1x1 pocket with a bomb right outside every exit has no escape."""
    field = np.full((5, 5), -1, dtype=int)
    field[2, 2] = 0  # the agent's only tile

    danger_map = {(2, 2): ((0, 10),)}

    assert safe_escape_exists(field, danger_map, set(), [], (2, 2)) is False


def test_safe_escape_exists_with_time_to_walk_away() -> None:
    field = make_field()
    # A freshly placed bomb (timer 3) leaves time to walk clear of the
    # footprint before it detonates, even starting on the bomb's own tile.
    bombs = [((4, 4), 3)]
    danger_map = build_danger_map(field, bombs, np.zeros_like(field))

    assert safe_escape_exists(field, danger_map, set(), bombs, (4, 4)) is True


def test_safe_escape_exists_is_false_with_no_time_and_no_side_exit() -> None:
    """Timer 0: every cardinal neighbor of the bomb's own tile is also in the
    blast line and equally lethal, so there is no single step to safety."""
    field = make_field()
    bombs = [((4, 4), 0)]
    danger_map = build_danger_map(field, bombs, np.zeros_like(field))

    assert safe_escape_exists(field, danger_map, set(), bombs, (4, 4)) is False


def test_safe_escape_exists_rejects_crossing_a_still_live_bomb_after_waiting() -> None:
    """Regression (#74 re-review): occupancy must be enforced at every BFS
    step, not just the first. `blocked_positions` is a step-0-only snapshot,
    so a path that waited once before stepping onto a bomb's tile was wrongly
    accepted as soon as its *blast* window (which opens later than the tile
    is physically re-enterable) allowed it -- even though the framework
    never removes a bomb before it detonates.

    Layout (diagonal offsets stay outside the blast's cross entirely):
    (1,1) agent -- (2,1) bomb, timer 4 -- (2,2) also in the blast (down arm)
    -- (3,2) genuinely safe (diagonal from the bomb).
    The only route to (3,2) crosses the bomb's own tile.
    """
    field = np.full((6, 5), -1, dtype=int)
    field[1, 1] = 0
    field[2, 1] = 0
    field[2, 2] = 0
    field[3, 2] = 0

    bombs = [((2, 1), 4)]
    danger_map = build_danger_map(field, bombs, np.zeros_like(field))

    assert safe_escape_exists(field, danger_map, set(), bombs, (1, 1)) is False


def test_safe_escape_exists_permits_waiting_on_its_own_bomb_first() -> None:
    """Regression: occupancy must exempt waiting in place, or an agent could
    never even wait one step on the tile where it just placed its own bomb
    (as `escape_after_bomb` always does) before walking away.

    The only exit is a single corridor tile with an unrelated,
    already-resolving explosion that blocks it for one step; escape requires
    waiting once on the bomb's own tile before that exit clears, then
    reaching a genuinely safe, diagonally-offset tile. Every other direction
    is walled off, so this isolates the wait exemption specifically: without
    it, the very first wait would itself be rejected as occupying the
    agent's own live bomb, and no escape would be found.
    """
    field = np.full((7, 7), -1, dtype=int)
    field[4, 4] = 0  # agent + bomb; every neighbor but (5, 4) is a wall
    field[5, 4] = 0  # the only exit, temporarily blocked by an explosion
    field[5, 3] = 0  # diagonal from the bomb: genuinely safe once reached

    explosion_map = np.zeros_like(field)
    explosion_map[5, 4] = 1  # blocks (5, 4) for arrival times 0 and 1 only

    bombs = [((4, 4), 5)]  # detonation_time = 6: (4, 4) stays occupied a while
    danger_map = build_danger_map(field, bombs, explosion_map)

    assert safe_escape_exists(field, danger_map, set(), bombs, (4, 4)) is True


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
    field = make_field(size=11)
    field[8, 4] = 1

    visible, dx, dy, distance_bin = nearest_crate_features(position=(4, 4), field=field)

    assert visible == 1
    assert (dx, dy) == (1, 0)
    assert distance_bin == 1  # one step to the nearest useful bombing tile


def test_nearest_crate_features_is_neutral_when_no_target_is_reachable() -> None:
    field = np.full((7, 7), -1, dtype=int)
    field[1, 1] = 0
    field[3, 1] = 1

    assert nearest_crate_features(position=(1, 1), field=field) == (0, 0, 0, 0)


def test_nearest_crate_features_breaks_equal_path_ties_by_direction_order() -> None:
    field = make_field(size=11)
    field[5, 1] = 1  # useful after moving UP to (5, 4)
    field[9, 5] = 1  # useful after moving RIGHT to (6, 5)

    # DIRECTIONS is ordered UP, RIGHT, DOWN, LEFT, so equal-length targets
    # have one deterministic representation.
    assert nearest_crate_features(position=(5, 5), field=field) == (1, 0, -1, 1)


def test_blast_footprint_bounded_by_bomb_power() -> None:
    field = np.zeros((15, 15), dtype=int)
    footprint = blast_footprint((7, 7), field)

    assert (7, 7 + BOMB_POWER) in footprint
    assert (7, 7 + BOMB_POWER + 1) not in footprint


def test_escape_after_bomb_feature_simulates_a_hypothetical_bomb() -> None:
    """Regression test: `escape_after_bomb` must add a bomb at the agent's own
    position before checking escape, not just reuse the current danger map.

    A dead-end corridor exactly BOMB_POWER tiles long is entirely safe with no
    bomb on the board (so a buggy "reuse the current danger map" computation
    would report an escape exists), but placing a bomb at the closed end
    covers the whole corridor and traps the agent.
    """
    from agent_code.DerKleineSprengstoffkapitalist.features import state_to_features

    field = np.full((6, 3), -1, dtype=int)
    field[1:5, 1] = 0  # a 1-wide corridor from x=1 (dead end) to x=4

    game_state = {
        "round": 1,
        "step": 1,
        "field": field,
        "self": ("test-agent", 0, True, (1, 1)),
        "coins": [],
        "bombs": [],
        "others": [],
        "explosion_map": np.zeros_like(field),
    }

    features = state_to_features(game_state)
    assert features is not None

    escape_after_bomb = features[14]
    assert escape_after_bomb == 0
