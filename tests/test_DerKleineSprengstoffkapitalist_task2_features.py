"""Tests for the tabular Task 2 feature assembly."""

from __future__ import annotations

from math import prod

import numpy as np
import pytest

from agent_code.DerKleineSprengstoffkapitalist.features import (
    FEATURE_COUNT,
    FEATURE_DOMAINS,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    THEORETICAL_STATE_SPACE_UPPER_BOUND,
    state_to_features,
    validate_features,
)
from agent_code.DerKleineVermoegensumverteiler.features import (
    state_to_features as parent_state_to_features,
)


def make_field(size: int = 9) -> np.ndarray:
    """Create an open square arena surrounded by stone walls."""

    field = np.zeros((size, size), dtype=int)
    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    field[:, -1] = -1

    return field


def make_game_state(
    *,
    field: np.ndarray | None = None,
    position: tuple[int, int] = (4, 4),
    bomb_available: bool = True,
    coins: list[tuple[int, int]] | None = None,
    bombs: list[tuple[tuple[int, int], int]] | None = None,
    others: list[tuple] | None = None,
    explosion_map: np.ndarray | None = None,
) -> dict:
    """Create a synthetic framework-compatible game state."""

    if field is None:
        field = make_field()

    if explosion_map is None:
        explosion_map = np.zeros_like(field)

    return {
        "round": 1,
        "step": 1,
        "field": field,
        "self": (
            "DerKleineSprengstoffkapitalist",
            0,
            bomb_available,
            position,
        ),
        "coins": [] if coins is None else coins,
        "bombs": [] if bombs is None else bombs,
        "others": [] if others is None else others,
        "explosion_map": explosion_map,
    }


def test_none_state_produces_no_features() -> None:
    assert state_to_features(None) is None


def test_feature_schema_contract() -> None:
    assert FEATURE_SCHEMA_VERSION == 2
    assert FEATURE_COUNT == 17
    assert len(FEATURE_NAMES) == FEATURE_COUNT
    assert len(FEATURE_DOMAINS) == FEATURE_COUNT
    assert len(set(FEATURE_NAMES)) == FEATURE_COUNT


def test_feature_names_have_stable_order() -> None:
    assert FEATURE_NAMES == (
        "free_up",
        "free_right",
        "free_down",
        "free_left",
        "coin_visible",
        "coin_dx",
        "coin_dy",
        "coin_distance_bin",
        "bomb_available",
        "current_danger_bin",
        "safe_direction_mask",
        "escape_after_bomb",
        "crate_visible",
        "crate_dx",
        "crate_dy",
        "crate_distance_bin",
        "crates_in_current_blast_bin",
    )


def test_theoretical_state_space_upper_bound() -> None:
    calculated_bound = prod(len(domain) for domain in FEATURE_DOMAINS)

    assert calculated_bound == 84_934_656
    assert calculated_bound == THEORETICAL_STATE_SPACE_UPPER_BOUND


def test_generated_features_have_valid_count_and_domains() -> None:
    game_state = make_game_state(coins=[(6, 4)])
    features = state_to_features(game_state)

    assert features is not None
    assert len(features) == FEATURE_COUNT

    for value, domain in zip(
        features,
        FEATURE_DOMAINS,
        strict=True,
    ):
        assert type(value) is int
        assert value in domain

    validate_features(features)


@pytest.mark.parametrize(
    "game_state",
    [
        make_game_state(coins=[]),
        make_game_state(coins=[(6, 4)]),
        make_game_state(
            coins=[(2, 4)],
            bombs=[((6, 4), 2)],
        ),
        make_game_state(
            coins=[(4, 6)],
            others=[("opponent", 0, True, (5, 4))],
        ),
    ],
)
def test_task1_projection_matches_frozen_parent(
    game_state: dict,
) -> None:
    parent_features = parent_state_to_features(game_state)
    successor_features = state_to_features(game_state)

    assert parent_features is not None
    assert successor_features is not None
    assert successor_features[:8] == parent_features


def test_open_safe_directions_use_all_mask_bits() -> None:
    features = state_to_features(make_game_state())

    assert features is not None

    assert features[:4] == (1, 1, 1, 1)

    # UP | RIGHT | DOWN | LEFT = 1 | 2 | 4 | 8 = 15
    assert features[10] == 15


def test_blocked_direction_is_not_marked_safe() -> None:
    field = make_field()
    field[4, 3] = 1  # crate directly above the agent

    features = state_to_features(
        make_game_state(
            field=field,
        )
    )

    assert features is not None

    # UP is physically blocked.
    assert features[0] == 0

    # Bit 0 represents UP and must not be set.
    assert features[10] & 1 == 0

    # RIGHT, DOWN and LEFT remain safe.
    assert features[10] == 14


def test_dangerous_neighbor_is_free_but_not_safe() -> None:
    field = make_field()
    explosion_map = np.zeros_like(field)

    # RIGHT is traversable but dangerous at arrival time 1.
    explosion_map[5, 4] = 1

    features = state_to_features(
        make_game_state(
            field=field,
            explosion_map=explosion_map,
        )
    )

    assert features is not None

    # RIGHT remains physically traversable.
    assert features[1] == 1

    # Bit 1 represents RIGHT and must not be set.
    assert features[10] & (1 << 1) == 0

    # UP, DOWN and LEFT remain safe: 1 + 4 + 8 = 13.
    assert features[10] == 13


def test_current_explosion_is_encoded_as_immediate_danger() -> None:
    field = make_field()
    explosion_map = np.zeros_like(field)
    explosion_map[4, 4] = 1

    features = state_to_features(
        make_game_state(
            field=field,
            explosion_map=explosion_map,
        )
    )

    assert features is not None
    assert features[9] == 1


def test_future_bomb_danger_is_encoded() -> None:
    features = state_to_features(
        make_game_state(
            bombs=[((6, 4), 0)],
        )
    )

    assert features is not None

    # Timer zero explodes after the next action.
    assert features[9] == 2


def test_bomb_availability_is_encoded() -> None:
    available = state_to_features(make_game_state(bomb_available=True))
    unavailable = state_to_features(make_game_state(bomb_available=False))

    assert available is not None
    assert unavailable is not None

    assert available[8] == 1
    assert unavailable[8] == 0


def test_escape_feature_is_neutral_when_bomb_is_unavailable() -> None:
    features = state_to_features(make_game_state(bomb_available=False))

    assert features is not None
    assert features[11] == 0


def test_escape_after_bomb_exists_in_open_arena() -> None:
    features = state_to_features(
        make_game_state(
            bomb_available=True,
        )
    )

    assert features is not None
    assert features[11] == 1


def test_escape_after_bomb_is_false_in_sealed_position() -> None:
    field = np.full((7, 7), -1, dtype=int)
    field[3, 3] = 0

    features = state_to_features(
        make_game_state(
            field=field,
            position=(3, 3),
            bomb_available=True,
        )
    )

    assert features is not None
    assert features[11] == 0


def test_missing_crates_use_neutral_features() -> None:
    features = state_to_features(make_game_state())

    assert features is not None
    assert features[12:16] == (0, 0, 0, 0)
    assert features[16] == 0


def test_reachable_crate_target_is_encoded() -> None:
    field = make_field(size=11)
    field[8, 4] = 1

    features = state_to_features(
        make_game_state(
            field=field,
        )
    )

    assert features is not None

    assert features[12] == 1
    assert features[13:15] == (1, 0)
    assert features[15] == 1


def test_crate_in_current_blast_is_encoded() -> None:
    field = make_field()
    field[6, 4] = 1

    features = state_to_features(
        make_game_state(
            field=field,
        )
    )

    assert features is not None
    assert features[16] == 1

    bomb_has_useful_target = features[16] > 0
    assert bomb_has_useful_target is True


def test_crate_count_is_capped_at_three() -> None:
    field = make_field()

    field[5, 4] = 1
    field[6, 4] = 1
    field[4, 5] = 1
    field[4, 6] = 1

    features = state_to_features(
        make_game_state(
            field=field,
        )
    )

    assert features is not None
    assert features[16] == 3


def test_validate_features_rejects_wrong_length() -> None:
    with pytest.raises(
        ValueError,
        match="Expected 17 features",
    ):
        validate_features((0,) * 16)


def test_validate_features_rejects_non_integer() -> None:
    invalid_features = list(state_to_features(make_game_state()) or ())
    invalid_features[8] = True

    with pytest.raises(
        ValueError,
        match="must be an integer",
    ):
        validate_features(tuple(invalid_features))


def test_validate_features_rejects_value_outside_domain() -> None:
    invalid_features = list(state_to_features(make_game_state()) or ())
    invalid_features[10] = 16

    with pytest.raises(
        ValueError,
        match="invalid value",
    ):
        validate_features(tuple(invalid_features))


def test_validate_features_rejects_safe_blocked_direction() -> None:
    invalid_features = list(state_to_features(make_game_state()) or ())

    invalid_features[0] = 0
    invalid_features[10] |= 1

    with pytest.raises(
        ValueError,
        match="cannot be marked as safe",
    ):
        validate_features(tuple(invalid_features))


def test_validate_features_rejects_non_neutral_missing_coin() -> None:
    invalid_features = list(state_to_features(make_game_state()) or ())

    invalid_features[4] = 0
    invalid_features[5] = 1

    with pytest.raises(
        ValueError,
        match="Missing coins",
    ):
        validate_features(tuple(invalid_features))


def test_validate_features_rejects_non_neutral_missing_crate() -> None:
    invalid_features = list(state_to_features(make_game_state()) or ())

    invalid_features[12] = 0
    invalid_features[13] = 1

    with pytest.raises(
        ValueError,
        match="Missing crate targets",
    ):
        validate_features(tuple(invalid_features))
