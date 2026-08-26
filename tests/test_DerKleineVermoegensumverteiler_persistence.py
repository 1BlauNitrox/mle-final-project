"""Tests for task 1 model persistence."""

import json
from pathlib import Path

import numpy as np
import pytest

from agent_code.DerKleineVermoegensumverteiler.config import (
    ACTIONS,
)
from agent_code.DerKleineVermoegensumverteiler.model import (
    QTable,
)
from agent_code.DerKleineVermoegensumverteiler.persistence import (
    load_model,
    save_model,
)

STATE = (1, 1, 1, 1, 1, 1, 0, 2)


def test_save_load_round_trip(
    tmp_path: Path,
) -> None:
    model = QTable(
        learning_rate=0.25,
        discount_factor=0.8,
    )

    model.update(
        state=STATE,
        action="RIGHT",
        reward=4.0,
        next_state=None,
        terminal=True,
    )

    model_path = tmp_path / "model.npz"

    save_model(
        model,
        epsilon=0.4,
        completed_episodes=12,
        path=model_path,
    )

    loaded = load_model(model_path)

    assert loaded.epsilon == pytest.approx(0.4)
    assert loaded.completed_episodes == 12

    assert loaded.q_table.learning_rate == pytest.approx(0.25)
    assert loaded.q_table.discount_factor == pytest.approx(0.8)

    np.testing.assert_array_equal(
        loaded.q_table.q_values(STATE),
        model.q_values(STATE),
    )


def test_empty_q_table_round_trip(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.npz"

    save_model(
        QTable(),
        epsilon=1.0,
        completed_episodes=0,
        path=model_path,
    )

    loaded = load_model(model_path)

    assert len(loaded.q_table) == 0
    assert loaded.completed_episodes == 0


def test_save_replaces_existing_model(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.npz"
    first_model = QTable()
    second_model = QTable(learning_rate=1.0)

    save_model(
        first_model,
        epsilon=1.0,
        completed_episodes=0,
        path=model_path,
    )

    second_model.update(
        state=STATE,
        action="UP",
        reward=5.0,
        next_state=None,
        terminal=True,
    )

    save_model(
        second_model,
        epsilon=0.5,
        completed_episodes=1,
        path=model_path,
    )

    loaded = load_model(model_path)

    assert loaded.completed_episodes == 1
    assert loaded.q_table.q_values(STATE)[0] == 5.0


def test_missing_model_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
        match="does not exist",
    ):
        load_model(tmp_path / "missing.npz")


def test_corrupted_model_is_rejected(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.npz"
    model_path.write_bytes(b"not a numpy archive")

    with pytest.raises(
        ValueError,
        match="Could not load model archive",
    ):
        load_model(model_path)


def test_incompatible_action_order_is_rejected(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.npz"
    model = QTable()

    save_model(
        model,
        epsilon=1.0,
        completed_episodes=0,
        path=model_path,
    )

    with np.load(
        model_path,
        allow_pickle=False,
    ) as archive:
        metadata = json.loads(str(archive["metadata"].item()))

        metadata["actions"] = list(reversed(ACTIONS))

        states = archive["states"].copy()
        q_values = archive["q_values"].copy()

    np.savez_compressed(
        model_path,
        states=states,
        q_values=q_values,
        metadata=np.array(json.dumps(metadata)),
    )

    with pytest.raises(
        ValueError,
        match="Actions mismatch",
    ):
        load_model(model_path)


def test_non_finite_q_values_are_rejected(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.npz"
    model = QTable()

    model.values[STATE] = np.array([np.nan, 0.0, 0.0, 0.0, 0.0])

    with pytest.raises(
        ValueError,
        match="must all be finite",
    ):
        save_model(
            model,
            epsilon=1.0,
            completed_episodes=0,
            path=model_path,
        )


def test_scalar_state_array_is_rejected(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.npz"

    save_model(
        QTable(),
        epsilon=1.0,
        completed_episodes=0,
        path=model_path,
    )

    with np.load(
        model_path,
        allow_pickle=False,
    ) as archive:
        metadata = archive["metadata"].copy()
        q_values = archive["q_values"].copy()

    np.savez_compressed(
        model_path,
        states=np.array(1),
        q_values=q_values,
        metadata=metadata,
    )

    with pytest.raises(
        ValueError,
        match="two-dimensional",
    ):
        load_model(model_path)


def test_different_state_and_q_value_counts_are_rejected(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.npz"

    save_model(
        QTable(),
        epsilon=1.0,
        completed_episodes=0,
        path=model_path,
    )

    with np.load(
        model_path,
        allow_pickle=False,
    ) as archive:
        metadata = archive["metadata"].copy()

    states = np.zeros(
        (1, 8),
        dtype=np.int64,
    )
    q_values = np.zeros(
        (0, len(ACTIONS)),
        dtype=np.float64,
    )

    np.savez_compressed(
        model_path,
        states=states,
        q_values=q_values,
        metadata=metadata,
    )

    with pytest.raises(
        ValueError,
        match="row counts differ",
    ):
        load_model(model_path)
