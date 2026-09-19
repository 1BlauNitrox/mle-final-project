"""Contracts for the tabular Task 3 successor."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from agent_code.DerKleineKonkurrenzvernichter import callbacks, train
from agent_code.DerKleineKonkurrenzvernichter.features.opponents import (
    opponent_state_to_features,
)
from agent_code.DerKleineKonkurrenzvernichter.migration import load_parent_prior
from agent_code.DerKleineKonkurrenzvernichter.persistence import load_model


def game_state(*, others: list[tuple] | None = None) -> dict:
    field = np.zeros((7, 7), dtype=int)
    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    field[:, -1] = -1
    return {
        "round": 1,
        "step": 1,
        "field": field,
        "self": ("agent", 0, True, (3, 3)),
        "coins": [],
        "bombs": [],
        "others": [] if others is None else others,
        "explosion_map": np.zeros_like(field),
    }


def test_opponent_features_are_neutral_without_opponents() -> None:
    features = opponent_state_to_features(game_state())
    assert features is not None
    assert len(features) == 8
    assert features[5:] == (0, 0, 0)


def test_opponent_features_encode_route_distance_and_attack() -> None:
    features = opponent_state_to_features(
        game_state(others=[("peaceful", 0, True, (5, 3))])
    )
    assert features is not None
    assert features[5:] == (2, 2, 2)


def test_task2_prior_is_preserved_for_every_opponent_suffix() -> None:
    parent = load_parent_prior()
    loaded = load_model()
    parent_state = next(iter(parent.values))
    expected = parent.values[parent_state]

    for suffix in ((0, 0, 0), (1, 3, 1), (4, 2, 2)):
        np.testing.assert_array_equal(
            loaded.q_table.q_values((*parent_state, *suffix)),
            expected,
        )


def test_training_is_enabled_only_in_successor() -> None:
    agent = SimpleNamespace(train=True, logger=Mock())
    callbacks.setup(agent)
    train.setup_training(agent)
    assert agent.completed_episodes == 0
    assert agent.pending_transition is None
