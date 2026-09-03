"""Behavior-preservation tests for the DQN Task 2 successor."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from agent_code.DagobertDuckDQN import callbacks as parent_callbacks
from agent_code.DagobertDuckDQN.config import ACTIONS as PARENT_ACTIONS
from agent_code.DagobertDuckDQN.config import DEFAULT_CONFIG as PARENT_CONFIG
from agent_code.DagobertDuckDQN.config import FEATURE_COUNT as PARENT_FEATURE_COUNT
from agent_code.DagobertDuckDQN.config import (
    FEATURE_SCHEMA_VERSION as PARENT_FEATURE_SCHEMA_VERSION,
)
from agent_code.DagobertDuckDQN.config import REWARDS as PARENT_REWARDS
from agent_code.DagobertDuckDQN.features import state_to_features as parent_state_to_features
from agent_code.DagobertDuckDQN.persistence import (
    load_evaluation_checkpoint as load_parent_checkpoint,
)
from agent_code.DagobertDuckDQNTask2 import callbacks as successor_callbacks
from agent_code.DagobertDuckDQNTask2.config import ACTIONS as SUCCESSOR_ACTIONS
from agent_code.DagobertDuckDQNTask2.config import DEFAULT_CONFIG as SUCCESSOR_CONFIG
from agent_code.DagobertDuckDQNTask2.config import FEATURE_COUNT as SUCCESSOR_FEATURE_COUNT
from agent_code.DagobertDuckDQNTask2.config import (
    FEATURE_SCHEMA_VERSION as SUCCESSOR_FEATURE_SCHEMA_VERSION,
)
from agent_code.DagobertDuckDQNTask2.config import REWARDS as SUCCESSOR_REWARDS
from agent_code.DagobertDuckDQNTask2.features import (
    state_to_features as successor_state_to_features,
)
from agent_code.DagobertDuckDQNTask2.persistence import (
    load_evaluation_checkpoint as load_successor_checkpoint,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

PARENT_ROOT = REPOSITORY_ROOT / "agent_code" / "DagobertDuckDQN"
SUCCESSOR_ROOT = REPOSITORY_ROOT / "agent_code" / "DagobertDuckDQNTask2"

PARENT_CHECKPOINT = PARENT_ROOT / "checkpoint.pt"
SUCCESSOR_CHECKPOINT = SUCCESSOR_ROOT / "checkpoint.pt"
SUCCESSOR_MANIFEST = SUCCESSOR_ROOT / "artifact.json"

EXPECTED_SHA256 = "45e38fa8900acd0783a84c339bf81d7e718de7797fbeeb147b5db94da3e96649"


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
    assert PARENT_CHECKPOINT.read_bytes() == SUCCESSOR_CHECKPOINT.read_bytes()
    assert sha256_file(PARENT_CHECKPOINT) == EXPECTED_SHA256
    assert sha256_file(SUCCESSOR_CHECKPOINT) == EXPECTED_SHA256
    assert SUCCESSOR_CHECKPOINT.stat().st_size == 47280


def test_successor_manifest_records_parent_lineage() -> None:
    with SUCCESSOR_MANIFEST.open(encoding="utf-8") as file:
        manifest = json.load(file)

    assert manifest["status"] == "task2_successor_scaffold"
    assert manifest["capability"] == "task1_behavior_only"
    assert manifest["artifact"]["sha256"] == EXPECTED_SHA256
    assert manifest["artifact"]["size_bytes"] == 47280
    assert manifest["parent"]["agent"] == "DagobertDuckDQN"
    assert manifest["parent"]["artifact_sha256"] == EXPECTED_SHA256
    assert manifest["policy"]["task2_features_present"] is False
    assert manifest["policy"]["bomb_action_present"] is False
    assert manifest["policy"]["scientific_training_authorized"] is False


def test_successor_preserves_configuration() -> None:
    assert SUCCESSOR_ACTIONS == PARENT_ACTIONS
    assert SUCCESSOR_ACTIONS == ("UP", "RIGHT", "DOWN", "LEFT", "WAIT")
    assert "BOMB" not in SUCCESSOR_ACTIONS
    assert asdict(SUCCESSOR_CONFIG) == asdict(PARENT_CONFIG)
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


def test_successor_network_matches_parent(
    representative_states: list[dict],
) -> None:
    parent = load_parent_checkpoint(PARENT_CHECKPOINT)
    successor = load_successor_checkpoint(SUCCESSOR_CHECKPOINT)

    assert parent.completed_episodes == successor.completed_episodes

    probe_states = torch.randn(64, PARENT_FEATURE_COUNT)

    with torch.no_grad():
        parent_values = parent.network(probe_states)
        successor_values = successor.network(probe_states)

    assert torch.equal(parent_values, successor_values)


def test_read_only_actions_match_parent(
    representative_states: list[dict],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BOMBERMAN_AGENT_SEED", "43001")

    parent_agent = SimpleNamespace(train=False, logger=Mock())
    successor_agent = SimpleNamespace(train=False, logger=Mock())

    parent_callbacks.setup(parent_agent)
    successor_callbacks.setup(successor_agent)

    parent_hash_before = sha256_file(PARENT_CHECKPOINT)
    successor_hash_before = sha256_file(SUCCESSOR_CHECKPOINT)

    parent_actions = [parent_callbacks.act(parent_agent, state) for state in representative_states]
    successor_actions = [
        successor_callbacks.act(successor_agent, state) for state in representative_states
    ]

    assert successor_actions == parent_actions
    assert all(action in SUCCESSOR_ACTIONS for action in successor_actions)
    assert "BOMB" not in successor_actions

    assert sha256_file(PARENT_CHECKPOINT) == parent_hash_before
    assert sha256_file(SUCCESSOR_CHECKPOINT) == successor_hash_before
