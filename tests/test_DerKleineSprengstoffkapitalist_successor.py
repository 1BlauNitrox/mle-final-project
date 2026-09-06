"""Behavior-preservation tests for the tabular Task 2 successor."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from agent_code.DerKleineSprengstoffkapitalist import callbacks as successor_callbacks
from agent_code.DerKleineSprengstoffkapitalist.config import (
    ACTIONS as SUCCESSOR_ACTIONS,
)
from agent_code.DerKleineSprengstoffkapitalist.config import (
    DISCOUNT_FACTOR as SUCCESSOR_DISCOUNT_FACTOR,
)
from agent_code.DerKleineSprengstoffkapitalist.config import (
    LEARNING_RATE as SUCCESSOR_LEARNING_RATE,
)
from agent_code.DerKleineSprengstoffkapitalist.config import (
    REWARDS as SUCCESSOR_REWARDS,
)
from agent_code.DerKleineSprengstoffkapitalist.features import (
    FEATURE_COUNT as SUCCESSOR_FEATURE_COUNT,
)
from agent_code.DerKleineSprengstoffkapitalist.features import (
    FEATURE_SCHEMA_VERSION as SUCCESSOR_FEATURE_SCHEMA_VERSION,
)
from agent_code.DerKleineSprengstoffkapitalist.features import (
    state_to_features as successor_state_to_features,
)
from agent_code.DerKleineSprengstoffkapitalist.persistence import (
    load_model as load_successor_model,
)
from agent_code.DerKleineVermoegensumverteiler import callbacks as parent_callbacks
from agent_code.DerKleineVermoegensumverteiler.config import (
    ACTIONS as PARENT_ACTIONS,
)
from agent_code.DerKleineVermoegensumverteiler.config import (
    DISCOUNT_FACTOR as PARENT_DISCOUNT_FACTOR,
)
from agent_code.DerKleineVermoegensumverteiler.config import (
    LEARNING_RATE as PARENT_LEARNING_RATE,
)
from agent_code.DerKleineVermoegensumverteiler.config import (
    REWARDS as PARENT_REWARDS,
)
from agent_code.DerKleineVermoegensumverteiler.features import (
    FEATURE_COUNT as PARENT_FEATURE_COUNT,
)
from agent_code.DerKleineVermoegensumverteiler.features import (
    FEATURE_SCHEMA_VERSION as PARENT_FEATURE_SCHEMA_VERSION,
)
from agent_code.DerKleineVermoegensumverteiler.features import (
    state_to_features as parent_state_to_features,
)
from agent_code.DerKleineVermoegensumverteiler.persistence import (
    load_model as load_parent_model,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

PARENT_ROOT = REPOSITORY_ROOT / "agent_code" / "DerKleineVermoegensumverteiler"

SUCCESSOR_ROOT = REPOSITORY_ROOT / "agent_code" / "DerKleineSprengstoffkapitalist"

PARENT_MODEL = PARENT_ROOT / "model.npz"
SUCCESSOR_MODEL = SUCCESSOR_ROOT / "model.npz"
SUCCESSOR_MANIFEST = SUCCESSOR_ROOT / "artifact.json"

EXPECTED_SHA256 = "4e1da63a819ef8f51b112ffaf422ab251b853915375fe486538be8595b988307"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_game_state(
    *,
    position: tuple[int, int] = (3, 3),
    coins: list[tuple[int, int]] | None = None,
    bombs: list[tuple[tuple[int, int], int]] | None = None,
    others: list[tuple] | None = None,
) -> dict:
    """Create a synthetic framework-compatible game state."""

    field = np.zeros((7, 7), dtype=int)
    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    field[:, -1] = -1

    return {
        "round": 1,
        "step": 1,
        "field": field,
        "self": ("test-agent", 0, True, position),
        "coins": [] if coins is None else coins,
        "bombs": [] if bombs is None else bombs,
        "others": [] if others is None else others,
        "explosion_map": np.zeros_like(field),
    }


@pytest.fixture
def representative_states() -> list[dict]:
    return [
        make_game_state(coins=[]),
        make_game_state(coins=[(5, 3)]),
        make_game_state(coins=[(1, 3), (5, 3)]),
        make_game_state(
            coins=[(5, 4)],
            bombs=[((3, 2), 3)],
            others=[("opponent", 0, True, (4, 3))],
        ),
        make_game_state(position=(1, 1), coins=[(5, 5)]),
    ]


def test_successor_artifact_keeps_parent_lineage() -> None:
    successor = load_successor_model(SUCCESSOR_MODEL)

    assert successor.parent_model_sha256 == EXPECTED_SHA256
    assert sha256_file(PARENT_MODEL) == EXPECTED_SHA256
    assert SUCCESSOR_MODEL.is_file()


def test_successor_manifest_records_parent_lineage() -> None:
    with SUCCESSOR_MANIFEST.open(encoding="utf-8") as file:
        manifest = json.load(file)

    assert manifest["status"] == "task2_tabular_successor"
    assert manifest["capability"] == "task2_tabular_successor"

    assert manifest["parent"]["agent"] == "DerKleineVermoegensumverteiler"
    assert manifest["parent"]["artifact_sha256"] == EXPECTED_SHA256

    assert manifest["model_contract"]["model_schema_version"] == 3
    assert manifest["model_contract"]["feature_schema_version"] == 2
    assert manifest["model_contract"]["action_order"] == [
        "UP",
        "RIGHT",
        "DOWN",
        "LEFT",
        "WAIT",
        "BOMB",
    ]
    assert manifest["model_contract"]["forbidden_actions"] == []

    assert manifest["policy"]["task2_features_present"] is True
    assert manifest["policy"]["bomb_action_present"] is True
    assert manifest["policy"]["scientific_training_authorized"] is False


def test_successor_extends_parent_configuration() -> None:
    assert SUCCESSOR_ACTIONS == (
        "UP",
        "RIGHT",
        "DOWN",
        "LEFT",
        "WAIT",
        "BOMB",
    )

    assert SUCCESSOR_LEARNING_RATE == PARENT_LEARNING_RATE
    assert SUCCESSOR_DISCOUNT_FACTOR == PARENT_DISCOUNT_FACTOR

    for event, value in PARENT_REWARDS.items():
        assert SUCCESSOR_REWARDS[event] == value

    assert "CRATE_DESTROYED" in SUCCESSOR_REWARDS
    assert "COIN_FOUND" in SUCCESSOR_REWARDS
    assert "SURVIVED_ROUND" in SUCCESSOR_REWARDS


def test_successor_extends_parent_feature_contract() -> None:
    assert SUCCESSOR_FEATURE_COUNT == 17
    assert PARENT_FEATURE_COUNT == 8
    assert SUCCESSOR_FEATURE_SCHEMA_VERSION == 2
    assert PARENT_FEATURE_SCHEMA_VERSION == 1


def test_successor_features_preserve_parent_prefix(
    representative_states: list[dict],
) -> None:
    assert successor_state_to_features(None) is None

    for game_state in representative_states:
        parent_features = parent_state_to_features(game_state)
        successor_features = successor_state_to_features(game_state)

        assert parent_features is not None
        assert successor_features is not None

        assert len(parent_features) == 8
        assert len(successor_features) == 17
        assert successor_features[:8] == parent_features


def test_successor_q_table_uses_parent_prior() -> None:
    parent = load_parent_model(PARENT_MODEL)
    successor = load_successor_model(SUCCESSOR_MODEL)

    assert successor.q_table.learning_rate == pytest.approx(parent.q_table.learning_rate)
    assert successor.q_table.discount_factor == pytest.approx(parent.q_table.discount_factor)

    parent_state = next(iter(parent.q_table.values))
    parent_values = parent.q_table.q_values(parent_state)

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

    successor_values = successor.q_table.q_values(task2_state)

    assert successor_values.shape == (len(SUCCESSOR_ACTIONS),)
    np.testing.assert_array_equal(
        successor_values[:5],
        parent_values,
    )


def test_read_only_actions_match_parent(
    representative_states: list[dict],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BOMBERMAN_AGENT_SEED", "65001")

    parent_agent = SimpleNamespace(
        train=False,
        logger=Mock(),
    )
    successor_agent = SimpleNamespace(
        train=False,
        logger=Mock(),
    )

    parent_callbacks.setup(parent_agent)
    successor_callbacks.setup(successor_agent)

    parent_hash_before = sha256_file(PARENT_MODEL)
    successor_hash_before = sha256_file(SUCCESSOR_MODEL)

    parent_actions = [parent_callbacks.act(parent_agent, state) for state in representative_states]
    successor_actions = [
        successor_callbacks.act(successor_agent, state) for state in representative_states
    ]

    assert all(action in SUCCESSOR_ACTIONS for action in successor_actions)
    assert all(action in PARENT_ACTIONS for action in parent_actions)

    assert sha256_file(PARENT_MODEL) == parent_hash_before
    assert sha256_file(SUCCESSOR_MODEL) == successor_hash_before
