"""Contracts for the tabular Task 3 successor."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from agent_code.DerKleineKonkurrenzvernichter import callbacks, train
from agent_code.DerKleineKonkurrenzvernichter.features.opponents import (
    opponent_state_to_features,
)
from agent_code.DerKleineKonkurrenzvernichter.features.shared_targets import (
    SHARED_TARGET_STATE_REPRESENTATION,
    TARGET_CRATE,
    TARGET_OPPONENT,
    shared_target_state_to_features,
)
from agent_code.DerKleineKonkurrenzvernichter.migration import load_parent_prior
from agent_code.DerKleineKonkurrenzvernichter.model import QTable
from agent_code.DerKleineKonkurrenzvernichter.persistence import (
    load_model,
    save_model,
)


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


def test_task2_prior_is_preserved_for_unseen_opponent_state() -> None:
    parent = load_parent_prior()
    loaded = load_model()
    unseen_state = None
    for parent_state, expected in parent.values.items():
        for suffix in ((0, 0, 0), (1, 3, 1), (4, 2, 2)):
            candidate = (*parent_state, *suffix)
            if not loaded.q_table.contains_state(candidate):
                unseen_state = candidate
                np.testing.assert_array_equal(
                    loaded.q_table.q_values(candidate),
                    expected,
                )
                break
        if unseen_state is not None:
            break

    assert unseen_state is not None


def test_shared_target_prefers_reachable_opponent_over_crate() -> None:
    state = game_state(others=[("peaceful", 0, True, (5, 3))])
    state["field"][3, 4] = 1

    features = shared_target_state_to_features(state)

    assert features is not None
    assert features[3] == 2
    assert features[4] == TARGET_OPPONENT


def test_shared_target_uses_crate_without_opponent() -> None:
    state = game_state()
    state["field"][3, 4] = 1

    features = shared_target_state_to_features(state)

    assert features is not None
    assert features[3] == 0
    assert features[4] == TARGET_CRATE


def test_shared_target_marks_safe_useful_opponent_bomb() -> None:
    features = shared_target_state_to_features(
        game_state(others=[("peaceful", 0, True, (5, 3))])
    )

    assert features is not None
    assert features[5] == 3


def test_shared_target_prior_projects_target_kind_away() -> None:
    parent = load_parent_prior()
    parent_state = next(iter(parent.values))
    q_table = QTable(
        parent_values=parent.values,
        feature_count=6,
        initialization="task2_prior",
    )
    expected = parent.values[parent_state]
    shared_state = (
        parent_state[0],
        parent_state[1],
        parent_state[2],
        parent_state[3],
        TARGET_OPPONENT,
        parent_state[4],
    )

    np.testing.assert_array_equal(q_table.q_values(shared_state), expected)


def test_fresh_training_receives_task2_prior(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(callbacks, "MODEL_PATH", tmp_path / "missing.npz")
    monkeypatch.setenv(callbacks.STATE_REPRESENTATION_ENV, "compact_opponent")
    monkeypatch.setenv(callbacks.INITIALIZATION_ENV, "task2_prior")
    agent = SimpleNamespace(train=True, logger=Mock())

    callbacks.setup(agent)

    parent = load_parent_prior()
    parent_state = next(iter(parent.values))
    np.testing.assert_array_equal(
        agent.q_table.q_values((*parent_state, 0, 0, 0)),
        parent.values[parent_state],
    )


def test_shared_target_model_round_trip(tmp_path) -> None:
    parent = load_parent_prior()
    q_table = QTable(
        parent_values=parent.values,
        feature_count=6,
        initialization="task2_prior",
    )
    state = (0, 15, 2, 1, TARGET_OPPONENT, 3)
    q_table.update(
        state=state,
        action="BOMB",
        reward=5.0,
        next_state=None,
        terminal=True,
    )
    path = tmp_path / "shared-target.npz"

    save_model(
        q_table,
        epsilon=0.5,
        completed_episodes=1,
        state_representation=SHARED_TARGET_STATE_REPRESENTATION,
        initialization="task2_prior",
        path=path,
    )
    loaded = load_model(path)

    assert loaded.state_representation == SHARED_TARGET_STATE_REPRESENTATION
    assert loaded.initialization == "task2_prior"
    np.testing.assert_array_equal(
        loaded.q_table.q_values(state),
        q_table.q_values(state),
    )


def test_final_deadline_freeze_rejects_training() -> None:
    agent = SimpleNamespace(train=True, logger=Mock())
    callbacks.setup(agent)
    with pytest.raises(RuntimeError, match="frozen final tabular agent"):
        train.setup_training(agent)
    assert agent.completed_episodes == 10000
