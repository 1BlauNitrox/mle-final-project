"""Task 1 regression and lineage tests for the DQN Task 2 successor.

Issue #43 made this successor a byte-identical, behavior-preserving copy of
the frozen parent. Issue #44 changes that deliberately: the feature count,
action space, and checkpoint shape all grow. What must still hold is the
Task 1 *prefix* -- the first eight features, the five original action
indices, and the migrated network's continuity with the parent's weights.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from agent_code.DagobertDuckDQN.features import normalize_features as normalize_parent_features
from agent_code.DagobertDuckDQN.features import state_to_features as parent_state_to_features
from agent_code.DagobertDuckDQN.persistence import (
    load_evaluation_checkpoint as load_parent_checkpoint,
)
from agent_code.DagobertDuckDQNTask2.config import ACTIONS as SUCCESSOR_ACTIONS
from agent_code.DagobertDuckDQNTask2.config import FEATURE_COUNT as SUCCESSOR_FEATURE_COUNT
from agent_code.DagobertDuckDQNTask2.config import (
    FEATURE_SCHEMA_VERSION as SUCCESSOR_FEATURE_SCHEMA_VERSION,
)
from agent_code.DagobertDuckDQNTask2.features import (
    normalize_features as normalize_successor_features,
)
from agent_code.DagobertDuckDQNTask2.features import (
    state_to_features as successor_state_to_features,
)
from agent_code.DagobertDuckDQNTask2.migration import INHERITED_Q_VALUE_TOLERANCE
from agent_code.DagobertDuckDQNTask2.persistence import (
    load_evaluation_checkpoint as load_successor_checkpoint,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

PARENT_ROOT = REPOSITORY_ROOT / "agent_code" / "DagobertDuckDQN"
SUCCESSOR_ROOT = REPOSITORY_ROOT / "agent_code" / "DagobertDuckDQNTask2"

PARENT_CHECKPOINT = PARENT_ROOT / "checkpoint.pt"
SUCCESSOR_CHECKPOINT = SUCCESSOR_ROOT / "checkpoint.pt"
CORRECTED_SUCCESSOR_CHECKPOINT = (
    SUCCESSOR_ROOT / "checkpoint-issue85-zero-suffix.pt"
)
SUCCESSOR_MANIFEST = SUCCESSOR_ROOT / "artifact.json"

PARENT_SHA256 = "eb08e3f67b620ac2a253a2af4db3d5b4c6ea9e667a2aaf1d91e3fccf4ba8b05e"


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
        make_game_state(position=(1, 1), coins=[(5, 5)]),
    ]


def test_parent_checkpoint_is_unmodified() -> None:
    """Issue #44 must not touch the frozen Task 1 baseline."""
    assert sha256_file(PARENT_CHECKPOINT) == PARENT_SHA256


def test_successor_manifest_records_migration_lineage() -> None:
    with SUCCESSOR_MANIFEST.open(encoding="utf-8") as file:
        manifest = json.load(file)

    assert manifest["status"] == "task2_capability_in_development"
    assert manifest["capability"] == "task1_navigation_plus_task2_bombs_and_crates"
    assert manifest["artifact"]["sha256"] == sha256_file(SUCCESSOR_CHECKPOINT)
    assert manifest["corrected_migration_artifact"]["sha256"] == sha256_file(
        CORRECTED_SUCCESSOR_CHECKPOINT
    )
    assert manifest["corrected_migration_artifact"]["path"] == (
        CORRECTED_SUCCESSOR_CHECKPOINT.name
    )
    assert manifest["parent"]["agent"] == "DagobertDuckDQN"
    assert manifest["parent"]["artifact_sha256"] == PARENT_SHA256
    assert manifest["migration"]["parent_input_dim"] == 8
    assert manifest["migration"]["parent_output_dim"] == 5
    assert manifest["policy"]["task2_features_present"] is True
    assert manifest["policy"]["bomb_action_present"] is True
    assert manifest["policy"]["scientific_training_authorized"] is False


def test_action_space_preserves_task1_indices() -> None:
    assert SUCCESSOR_ACTIONS == ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")
    assert SUCCESSOR_ACTIONS[:5] == ("UP", "RIGHT", "DOWN", "LEFT", "WAIT")


def test_feature_schema_extends_task1() -> None:
    assert SUCCESSOR_FEATURE_COUNT == 21
    assert SUCCESSOR_FEATURE_SCHEMA_VERSION == 2


def test_task1_prefix_matches_parent_on_task1_states(
    representative_states: list[dict],
) -> None:
    """The first eight features must stay byte-for-byte what the parent computes."""
    assert successor_state_to_features(None) is None

    for game_state in representative_states:
        successor_features = successor_state_to_features(game_state)
        parent_features = parent_state_to_features(game_state)

        assert successor_features is not None
        assert successor_features[:8] == parent_features
        assert len(successor_features) == SUCCESSOR_FEATURE_COUNT


def test_task1_state_has_neutral_task2_features(
    representative_states: list[dict],
) -> None:
    """No bombs, no crates: every Task 2 feature must read as neutral/absent."""
    for game_state in representative_states:
        features = successor_state_to_features(game_state)
        assert features is not None

        (
            bomb_available,
            danger_bin,
            safe_up,
            safe_right,
            safe_down,
            safe_left,
            escape_after_bomb,
            crate_visible,
            crate_dx,
            crate_dy,
            crate_distance_bin,
            crates_here,
            useful_target,
        ) = features[8:]

        assert bomb_available == 1
        assert danger_bin == 0
        assert (safe_up, safe_right, safe_down, safe_left) == (
            features[0],
            features[1],
            features[2],
            features[3],
        )
        assert escape_after_bomb == 1
        assert crate_visible == 0
        assert (crate_dx, crate_dy, crate_distance_bin) == (0, 0, 0)
        assert crates_here == 0
        assert useful_target == 0


def test_migrated_network_continues_the_parent_weights() -> None:
    parent = load_parent_checkpoint(PARENT_CHECKPOINT)
    successor = load_successor_checkpoint(SUCCESSOR_CHECKPOINT)

    assert successor.completed_episodes == 0

    parent_linear = [layer for layer in parent.network.layers if hasattr(layer, "weight")]
    successor_linear = [layer for layer in successor.network.layers if hasattr(layer, "weight")]

    import torch

    assert torch.equal(parent_linear[0].weight, successor_linear[0].weight[:, :8])
    assert torch.equal(parent_linear[-1].weight, successor_linear[-1].weight[:5, :])


def test_corrected_artifact_preserves_parent_q_values(
    representative_states: list[dict],
) -> None:
    """The Issue #85 artifact, unlike the preserved #46 start, is function-safe."""
    parent = load_parent_checkpoint(PARENT_CHECKPOINT)
    corrected = load_successor_checkpoint(CORRECTED_SUCCESSOR_CHECKPOINT)

    for game_state in representative_states:
        parent_features = parent_state_to_features(game_state)
        successor_features = successor_state_to_features(game_state)
        assert parent_features is not None
        assert successor_features is not None

        with torch.no_grad():
            parent_q_values = parent.network(
                torch.from_numpy(normalize_parent_features(parent_features))
            )
            corrected_q_values = corrected.network(
                torch.from_numpy(normalize_successor_features(successor_features))
            )[:5]

        torch.testing.assert_close(
            corrected_q_values,
            parent_q_values,
            rtol=0.0,
            atol=INHERITED_Q_VALUE_TOLERANCE,
        )
