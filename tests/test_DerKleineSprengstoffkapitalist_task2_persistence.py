"""Tests for tabular Task 2 model persistence."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from agent_code.DerKleineSprengstoffkapitalist.config import (
    ACTIONS,
)
from agent_code.DerKleineSprengstoffkapitalist.features import (
    BASELINE_STATE_REPRESENTATION,
    COMPACT_STATE_REPRESENTATION,
)
from agent_code.DerKleineSprengstoffkapitalist.migration import (
    EXPECTED_PARENT_SHA256,
    PARENT_MODEL_PATH,
    load_parent_prior,
)
from agent_code.DerKleineSprengstoffkapitalist.model import (
    BOMB_PRIOR_MARGIN,
    PARENT_PRIOR_INITIALIZATION,
    ZERO_INITIALIZATION,
    QTable,
)
from agent_code.DerKleineSprengstoffkapitalist.persistence import (
    MODEL_PATH,
    MODEL_SCHEMA_VERSION,
    load_model,
    save_model,
)
from agent_code.DerKleineSprengstoffkapitalist.potential_shaping import (
    COMPACT_SAFETY_POTENTIAL_SHAPING,
    ESCAPE_DISTANCE_POTENTIAL_SHAPING,
    NO_POTENTIAL_SHAPING,
)

TEST_STATE = (
    1,
    1,
    1,
    1,
    0,
    0,
    0,
    0,
    1,
    0,
    15,
    1,
    0,
    0,
    0,
    0,
    0,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_model_schema_version_is_five() -> None:
    assert MODEL_SCHEMA_VERSION == 5


def test_parent_artifact_has_expected_checksum() -> None:
    assert sha256_file(PARENT_MODEL_PATH) == EXPECTED_PARENT_SHA256


def test_sparse_model_round_trip(tmp_path: Path) -> None:
    parent = load_parent_prior()
    q_table = QTable(parent_values=parent.values)

    q_table.update(
        state=TEST_STATE,
        action="BOMB",
        reward=2.0,
        next_state=None,
        terminal=True,
    )

    model_path = tmp_path / "model.npz"

    save_model(
        q_table,
        epsilon=0.8,
        completed_episodes=3,
        path=model_path,
        action_masking="framework_legal",
    )

    loaded = load_model(model_path)

    assert loaded.epsilon == pytest.approx(0.8)
    assert loaded.completed_episodes == 3
    assert loaded.parent_model_sha256 == EXPECTED_PARENT_SHA256
    assert loaded.useful_bomb_reward == pytest.approx(0.0)
    assert len(loaded.q_table) == 1
    assert loaded.action_masking == "framework_legal"
    assert loaded.state_representation == BASELINE_STATE_REPRESENTATION
    assert loaded.initialization == PARENT_PRIOR_INITIALIZATION
    assert loaded.q_table.feature_count == 17
    assert loaded.q_table.initialization == PARENT_PRIOR_INITIALIZATION
    assert loaded.q_table.total_state_visits == 1
    assert loaded.potential_shaping == NO_POTENTIAL_SHAPING

    np.testing.assert_array_equal(
        loaded.q_table.q_values(TEST_STATE),
        q_table.q_values(TEST_STATE),
    )


def test_compact_zero_initialized_model_round_trip(
    tmp_path: Path,
) -> None:
    compact_state = (0, 15, 2, 0, 1)
    q_table = QTable(
        feature_count=5,
        initialization=ZERO_INITIALIZATION,
    )

    q_table.update(
        state=compact_state,
        action="RIGHT",
        reward=2.0,
        next_state=None,
        terminal=True,
    )

    model_path = tmp_path / "compact-model.npz"

    save_model(
        q_table,
        epsilon=0.8,
        completed_episodes=3,
        state_representation=COMPACT_STATE_REPRESENTATION,
        initialization=ZERO_INITIALIZATION,
        path=model_path,
    )

    loaded = load_model(model_path)

    assert loaded.state_representation == COMPACT_STATE_REPRESENTATION
    assert loaded.initialization == ZERO_INITIALIZATION
    assert loaded.q_table.feature_count == 5
    assert loaded.q_table.initialization == ZERO_INITIALIZATION
    assert len(loaded.q_table) == 1
    assert loaded.q_table.total_state_visits == 1
    assert loaded.q_table.singleton_state_fraction == pytest.approx(1.0)

    np.testing.assert_array_equal(
        loaded.q_table.q_values(compact_state),
        q_table.q_values(compact_state),
    )


def test_useful_bomb_reward_round_trip(tmp_path: Path) -> None:
    parent = load_parent_prior()
    model_path = tmp_path / "model.npz"

    save_model(
        QTable(parent_values=parent.values),
        epsilon=1.0,
        completed_episodes=0,
        useful_bomb_reward=1.0,
        path=model_path,
    )

    assert load_model(model_path).useful_bomb_reward == pytest.approx(1.0)


def test_empty_model_remains_sparse_after_loading(
    tmp_path: Path,
) -> None:
    parent = load_parent_prior()
    q_table = QTable(parent_values=parent.values)
    model_path = tmp_path / "model.npz"

    save_model(
        q_table,
        epsilon=1.0,
        completed_episodes=0,
        path=model_path,
    )

    loaded = load_model(model_path)

    assert len(loaded.q_table) == 0
    assert loaded.q_table.total_state_visits == 0
    assert loaded.potential_shaping == NO_POTENTIAL_SHAPING


def test_read_only_lookup_does_not_create_state(
    tmp_path: Path,
) -> None:
    parent = load_parent_prior()
    q_table = QTable(parent_values=parent.values)
    model_path = tmp_path / "model.npz"

    save_model(
        q_table,
        epsilon=0.0,
        completed_episodes=0,
        path=model_path,
    )

    hash_before = sha256_file(model_path)
    loaded = load_model(model_path)

    values = loaded.q_table.q_values(TEST_STATE)

    assert values.shape == (len(ACTIONS),)
    assert len(loaded.q_table) == 0
    assert sha256_file(model_path) == hash_before


def test_known_parent_state_uses_parent_prior(
    tmp_path: Path,
) -> None:
    parent = load_parent_prior()
    parent_state = next(iter(parent.values))
    parent_values = parent.values[parent_state]

    task2_state = (
        *parent_state,
        1,
        0,
        15,
        1,
        0,
        0,
        0,
        0,
        0,
    )

    q_table = QTable(parent_values=parent.values)
    model_path = tmp_path / "model.npz"

    save_model(
        q_table,
        epsilon=1.0,
        completed_episodes=0,
        path=model_path,
    )

    loaded = load_model(model_path)
    migrated_values = loaded.q_table.q_values(task2_state)

    np.testing.assert_array_equal(
        migrated_values[:5],
        parent_values,
    )

    assert migrated_values[5] == pytest.approx(np.min(parent_values) - BOMB_PRIOR_MARGIN)


def test_unknown_parent_state_uses_neutral_prior(
    tmp_path: Path,
) -> None:
    parent = load_parent_prior()
    q_table = QTable(parent_values=parent.values)
    model_path = tmp_path / "model.npz"

    save_model(
        q_table,
        epsilon=1.0,
        completed_episodes=0,
        path=model_path,
    )

    loaded = load_model(model_path)

    unknown_state = (
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )

    np.testing.assert_array_equal(
        loaded.q_table.q_values(unknown_state),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),
    )


def test_task1_archive_is_rejected_as_task2_model() -> None:
    with pytest.raises(
        ValueError,
        match="schema version mismatch",
    ):
        load_model(PARENT_MODEL_PATH)


def test_corrupt_archive_is_rejected(
    tmp_path: Path,
) -> None:
    corrupt_path = tmp_path / "corrupt.npz"
    corrupt_path.write_bytes(b"not a numpy archive")

    with pytest.raises(
        ValueError,
        match="Could not load model archive",
    ):
        load_model(corrupt_path)


def test_checked_in_schema_three_model_loads_as_legacy_baseline() -> None:
    loaded = load_model(MODEL_PATH)

    assert loaded.state_representation == BASELINE_STATE_REPRESENTATION
    assert loaded.initialization == PARENT_PRIOR_INITIALIZATION
    assert loaded.q_table.feature_count == 17


def test_compact_representation_rejects_baseline_q_table(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="feature count does not match",
    ):
        save_model(
            QTable(),
            epsilon=1.0,
            completed_episodes=0,
            state_representation=COMPACT_STATE_REPRESENTATION,
            path=tmp_path / "invalid.npz",
        )


@pytest.mark.parametrize(
    "potential_shaping",
    (
        COMPACT_SAFETY_POTENTIAL_SHAPING,
        ESCAPE_DISTANCE_POTENTIAL_SHAPING,
    ),
)
def test_potential_shaping_mode_round_trip(
    tmp_path: Path,
    potential_shaping: str,
) -> None:
    compact_state = (0, 15, 2, 0, 1)
    q_table = QTable(
        feature_count=5,
        initialization=ZERO_INITIALIZATION,
    )
    q_table.update(
        state=compact_state,
        action="WAIT",
        reward=0.0,
        next_state=None,
        terminal=True,
    )
    model_path = tmp_path / "potential-model.npz"

    save_model(
        q_table,
        epsilon=1.0,
        completed_episodes=1,
        state_representation=COMPACT_STATE_REPRESENTATION,
        initialization=ZERO_INITIALIZATION,
        potential_shaping=potential_shaping,
        path=model_path,
    )

    loaded = load_model(model_path)

    assert loaded.potential_shaping == potential_shaping
