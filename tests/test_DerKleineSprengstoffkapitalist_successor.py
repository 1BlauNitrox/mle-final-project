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


def test_successor_artifact_is_byte_identical_to_parent() -> None:
    assert PARENT_MODEL.read_bytes() == SUCCESSOR_MODEL.read_bytes()
    assert sha256_file(PARENT_MODEL) == EXPECTED_SHA256
    assert sha256_file(SUCCESSOR_MODEL) == EXPECTED_SHA256
    assert SUCCESSOR_MODEL.stat().st_size == 6845


def test_successor_manifest_records_parent_lineage() -> None:
    with SUCCESSOR_MANIFEST.open(encoding="utf-8") as file:
        manifest = json.load(file)

    assert manifest["status"] == "task2_successor_scaffold"
    assert manifest["capability"] == "task1_behavior_only"
    assert manifest["artifact"]["sha256"] == EXPECTED_SHA256
    assert manifest["artifact"]["size_bytes"] == 6845
    assert manifest["parent"]["agent"] == "DerKleineVermoegensumverteiler"
    assert manifest["parent"]["artifact_sha256"] == EXPECTED_SHA256
    assert manifest["policy"]["task2_features_present"] is False
    assert manifest["policy"]["bomb_action_present"] is False
    assert manifest["policy"]["scientific_training_authorized"] is False


def test_successor_preserves_configuration() -> None:
    assert SUCCESSOR_ACTIONS == PARENT_ACTIONS
    assert SUCCESSOR_ACTIONS == (
        "UP",
        "RIGHT",
        "DOWN",
        "LEFT",
        "WAIT",
    )
    assert "BOMB" not in SUCCESSOR_ACTIONS
    assert SUCCESSOR_LEARNING_RATE == PARENT_LEARNING_RATE
    assert SUCCESSOR_DISCOUNT_FACTOR == PARENT_DISCOUNT_FACTOR
    assert SUCCESSOR_REWARDS == PARENT_REWARDS


def test_successor_preserves_feature_contract() -> None:
    assert SUCCESSOR_FEATURE_COUNT == PARENT_FEATURE_COUNT == 8
    assert SUCCESSOR_FEATURE_SCHEMA_VERSION == PARENT_FEATURE_SCHEMA_VERSION == 1


def test_successor_features_match_parent(
    representative_states: list[dict],
) -> None:
    assert successor_state_to_features(None) is None

    for game_state in representative_states:
        assert successor_state_to_features(game_state) == parent_state_to_features(game_state)


def test_successor_q_table_matches_parent() -> None:
    parent = load_parent_model(PARENT_MODEL)
    successor = load_successor_model(SUCCESSOR_MODEL)

    assert parent.epsilon == successor.epsilon
    assert parent.completed_episodes == successor.completed_episodes
    assert parent.q_table.learning_rate == successor.q_table.learning_rate
    assert parent.q_table.discount_factor == successor.q_table.discount_factor
    assert parent.q_table.values.keys() == successor.q_table.values.keys()

    for state in parent.q_table.values:
        np.testing.assert_array_equal(
            parent.q_table.q_values(state),
            successor.q_table.q_values(state),
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

    assert successor_actions == parent_actions
    assert all(action in SUCCESSOR_ACTIONS for action in successor_actions)
    assert "BOMB" not in successor_actions

    assert sha256_file(PARENT_MODEL) == parent_hash_before
    assert sha256_file(SUCCESSOR_MODEL) == successor_hash_before
