"""Tests for the Task 1 baseline feature encoder."""

import numpy as np

from agent_code.DagobertDuckDQN.features import (
    normalize_features,
    state_to_features,
)


def make_game_state(
    *,
    position: tuple[int, int] = (3, 3),
    coins: list[tuple[int, int]] | None = None,
    bombs: list[tuple[tuple[int, int], int]] | None = None,
    others: list[tuple] | None = None,
) -> dict:
    """Create a small synthetic framework-compatible game state."""
    field = np.zeros((7, 7), dtype=int)

    # Add the indestructible outer wall.
    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    field[:, -1] = -1

    return {
        "round": 1,
        "step": 1,
        "field": field,
        "self": ("DagobertDuckDQN", 0, True, position),
        "coins": [] if coins is None else coins,
        "bombs": [] if bombs is None else bombs,
        "others": [] if others is None else others,
        "explosion_map": np.zeros_like(field),
    }


def test_none_state_produces_no_features() -> None:
    assert state_to_features(None) is None


def test_feature_vector_is_hashable() -> None:
    features = state_to_features(make_game_state(coins=[(5, 3)]))

    assert isinstance(features, tuple)
    hash(features)


def test_all_adjacent_tiles_are_free_in_open_space() -> None:
    features = state_to_features(make_game_state(coins=[(5, 3)]))

    assert features is not None
    assert features[:4] == (1, 1, 1, 1)


def test_direction_deltas_match_action_order() -> None:
    from agent_code.DagobertDuckDQN.features import (
        DIRECTIONS,
    )

    assert DIRECTIONS == (
        (0, -1),
        (1, 0),
        (0, 1),
        (-1, 0),
    )


def test_walls_are_encoded_as_blocked() -> None:
    game_state = make_game_state(coins=[(5, 3)])
    x, y = game_state["self"][3]

    game_state["field"][x, y - 1] = -1
    game_state["field"][x + 1, y] = 1

    features = state_to_features(game_state)

    assert features is not None
    assert features[:4] == (0, 0, 1, 1)


def test_bombs_and_opponents_block_adjacent_tiles() -> None:
    game_state = make_game_state(
        coins=[(5, 3)],
        bombs=[((3, 2), 3)],
        others=[("opponent", 0, True, (4, 3))],
    )

    features = state_to_features(game_state)

    assert features is not None
    assert features[:4] == (0, 0, 1, 1)


def test_nearest_coin_direction_and_distance_are_encoded() -> None:
    features = state_to_features(
        make_game_state(
            position=(3, 3),
            coins=[(5, 4), (1, 1)],
        )
    )

    assert features is not None

    # The nearest coin is (5, 4):
    # dx positive, dy positive, Manhattan distance 3.
    assert features[4:] == (1, 1, 1, 2)


def test_aligned_coin_has_zero_for_unchanged_axis() -> None:
    features = state_to_features(
        make_game_state(
            position=(3, 3),
            coins=[(3, 5)],
        )
    )

    assert features is not None
    assert features[4:] == (1, 0, 1, 2)


def test_missing_coins_use_explicit_no_coin_encoding() -> None:
    features = state_to_features(make_game_state(coins=[]))

    assert features is not None
    assert features[4:] == (0, 0, 0, 0)


def test_distant_coin_uses_largest_distance_bin() -> None:
    features = state_to_features(
        make_game_state(
            position=(1, 1),
            coins=[(5, 5)],
        )
    )

    assert features is not None
    assert features[4:] == (1, 1, 1, 3)


def test_coin_selection_is_deterministic_for_equal_distances() -> None:
    first = state_to_features(
        make_game_state(
            coins=[(5, 3), (1, 3)],
        )
    )
    second = state_to_features(
        make_game_state(
            coins=[(1, 3), (5, 3)],
        )
    )

    assert first == second

def test_normalization_preserves_order_and_scales_distance() -> None:
    raw_features = (1, 0, 1, 0, 1, -1, 1, 3)

    normalized = normalize_features(raw_features)

    assert normalized.shape == (8,)
    assert normalized.dtype == np.float32
    np.testing.assert_allclose(
        normalized,
        np.array(
            [1.0, 0.0, 1.0, 0.0, 1.0, -1.0, 1.0, 1.0],
            dtype=np.float32,
        ),
    )

def test_normalization_keeps_no_coin_encoding_at_zero() -> None:
    normalized = normalize_features((1, 1, 1, 1, 0, 0, 0, 0))

    np.testing.assert_array_equal(
        normalized,
        np.array(
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        ),
    )

def test_normalization_rejects_wrong_feature_count() -> None:
    with np.testing.assert_raises(ValueError):
        normalize_features((1, 0, 1))