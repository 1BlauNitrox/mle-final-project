"""Tests for the compact post-bomb escape state representation."""

from __future__ import annotations

import numpy as np
import pytest

from agent_code.DerKleineSprengstoffkapitalist.features import (
    COMPACT_ESCAPE_STATE_REPRESENTATION,
    encode_state,
    get_state_representation,
)
from agent_code.DerKleineSprengstoffkapitalist.features.compact_escape import (
    COMPACT_ESCAPE_FEATURE_COUNT,
    COMPACT_ESCAPE_FEATURE_NAMES,
    COMPACT_ESCAPE_STATE_SPACE_UPPER_BOUND,
    COMPLETE_ROUTE,
    NO_ROUTE,
    NOT_APPLICABLE,
    TEMPORARY_SAFETY_ONLY,
    compact_escape_state_to_features,
    post_bomb_escape_status,
    validate_compact_escape_features,
)


def _field(size: int = 9) -> np.ndarray:
    field = np.zeros((size, size), dtype=int)
    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    field[:, -1] = -1
    return field


def _state(
    field: np.ndarray,
    *,
    bomb_available: bool = True,
    bombs: list[tuple[tuple[int, int], int]] | None = None,
) -> dict:
    return {
        "round": 1,
        "step": 1,
        "field": field,
        "self": ("test", 0, bomb_available, (3, 3)),
        "coins": [],
        "bombs": [] if bombs is None else bombs,
        "others": [],
        "explosion_map": np.zeros_like(field),
    }


def test_schema_appends_one_categorical_feature() -> None:
    representation = get_state_representation(COMPACT_ESCAPE_STATE_REPRESENTATION)
    assert COMPACT_ESCAPE_FEATURE_COUNT == 6
    assert COMPACT_ESCAPE_FEATURE_NAMES[-1] == "escape_after_bomb_status"
    assert COMPACT_ESCAPE_STATE_SPACE_UPPER_BOUND == 19_200
    assert representation.feature_count == 6


@pytest.mark.parametrize(
    ("bomb_available", "place_crate"),
    [(False, True), (True, False)],
)
def test_not_applicable_without_available_useful_bomb(
    bomb_available: bool, place_crate: bool
) -> None:
    field = _field()
    if place_crate:
        field[5, 3] = 1
    assert post_bomb_escape_status(
        _state(field, bomb_available=bomb_available)
    ) == NOT_APPLICABLE


def test_complete_route_in_open_arena() -> None:
    field = _field()
    field[5, 3] = 1
    assert post_bomb_escape_status(_state(field)) == COMPLETE_ROUTE


def test_temporary_safety_without_complete_route() -> None:
    field = np.full((7, 7), -1, dtype=int)
    field[3, 3] = 0
    field[3, 2] = 0
    field[3, 1] = 0
    field[4, 3] = 1
    assert post_bomb_escape_status(_state(field)) == TEMPORARY_SAFETY_ONLY


def test_no_route_in_sealed_crate_position() -> None:
    field = np.full((7, 7), -1, dtype=int)
    field[3, 3] = 0
    field[4, 3] = 1
    assert post_bomb_escape_status(_state(field)) == NO_ROUTE


def test_active_bomb_blocks_the_only_temporary_move() -> None:
    field = np.full((7, 7), -1, dtype=int)
    field[3, 3] = 0
    field[3, 2] = 0
    field[3, 1] = 0
    field[4, 3] = 1
    game_state = _state(field, bombs=[((3, 2), 3)])
    assert post_bomb_escape_status(game_state) == NO_ROUTE


def test_encoding_is_deterministic_and_preserves_compact_prefix() -> None:
    field = _field()
    field[5, 3] = 1
    game_state = _state(field)
    first = compact_escape_state_to_features(game_state)
    second = encode_state(game_state, COMPACT_ESCAPE_STATE_REPRESENTATION)
    assert first == second
    assert first is not None
    assert first[-1] == COMPLETE_ROUTE


def test_validation_rejects_invalid_status() -> None:
    with pytest.raises(ValueError, match="escape_after_bomb_status"):
        validate_compact_escape_features((0, 15, 0, 0, 1, 4))
