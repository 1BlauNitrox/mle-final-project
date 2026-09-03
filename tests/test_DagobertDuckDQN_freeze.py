"""Contract tests for the frozen Task 1 DQN baseline."""

from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import yaml

from agent_code.DagobertDuckDQN import train
from agent_code.DagobertDuckDQN.config import ACTIONS, DEFAULT_CONFIG, REWARDS
from agent_code.DagobertDuckDQN.persistence import (
    CHECKPOINT_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION,
    MODEL_SCHEMA_VERSION,
    load_evaluation_checkpoint,
)
from scripts.package_agent import package_agent

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AGENT_ROOT = REPOSITORY_ROOT / "agent_code" / "DagobertDuckDQN"
MANIFEST_PATH = AGENT_ROOT / "artifact.json"
CHECKPOINT_PATH = AGENT_ROOT / "checkpoint.pt"
CONFIG_PATH = AGENT_ROOT / "baseline-config.yaml"
REFERENCE_PATH = AGENT_ROOT / "reference-results.csv"

EXPECTED_CHECKPOINT_SHA256 = (
    "45e38fa8900acd0783a84c339bf81d7e718de7797fbeeb147b5db94da3e96649"
)


@pytest.fixture
def manifest() -> dict:
    with MANIFEST_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def test_frozen_artifact_integrity(manifest: dict) -> None:
    checkpoint_bytes = CHECKPOINT_PATH.read_bytes()

    assert len(checkpoint_bytes) == manifest["artifact"]["size_bytes"]
    assert hashlib.sha256(checkpoint_bytes).hexdigest() == (
        manifest["artifact"]["sha256"]
    )
    assert manifest["artifact"]["sha256"] == EXPECTED_CHECKPOINT_SHA256


def test_frozen_model_loads_and_matches_schema(manifest: dict) -> None:
    loaded = load_evaluation_checkpoint(CHECKPOINT_PATH)
    contract = manifest["model_contract"]

    assert loaded.completed_episodes == 10_000
    assert contract["checkpoint_schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert contract["model_schema_version"] == MODEL_SCHEMA_VERSION
    assert contract["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert contract["action_order"] == list(ACTIONS)
    assert contract["forbidden_actions"] == ["BOMB"]
    assert "BOMB" not in ACTIONS
    assert loaded.config == DEFAULT_CONFIG


def test_frozen_configuration_matches_agent() -> None:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        configuration = yaml.safe_load(file)

    hyperparameters = configuration["agent"]["hyperparameters"]
    rewards = configuration["agent"]["rewards"]

    assert hyperparameters["learning_rate"] == DEFAULT_CONFIG.learning_rate
    assert hyperparameters["discount_factor"] == DEFAULT_CONFIG.discount_factor
    assert hyperparameters["batch_size"] == DEFAULT_CONFIG.batch_size
    assert hyperparameters["replay_capacity"] == DEFAULT_CONFIG.replay_capacity
    assert rewards["COIN_COLLECTED"] == REWARDS["COIN_COLLECTED"]
    assert rewards["INVALID_ACTION"] == REWARDS["INVALID_ACTION"]
    assert rewards["WAITED"] == REWARDS["WAITED"]
    assert rewards["MOVED_TOWARDS_COIN"] == REWARDS["MOVED_TOWARDS_COIN"]
    assert rewards["MOVED_AWAY_FROM_COIN"] == REWARDS["MOVED_AWAY_FROM_COIN"]


def test_reference_results_match_selected_run() -> None:
    with REFERENCE_PATH.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 40
    assert [int(row["world_seed"]) for row in rows] == list(range(31001, 31041))
    assert {int(row["agent_seed"]) for row in rows} == {25002}
    assert sum(int(row["coins_collected"]) for row in rows) == 1633
    assert sum(int(row["episode_steps"]) for row in rows) == 12713
    assert sum(int(row["invalid_actions"]) for row in rows) == 4
    assert sum(int(row["action_bomb"]) for row in rows) == 0
    assert sum(1 for row in rows if row["full_clear"] == "True") == 12


def test_training_is_rejected_without_modifying_artifact() -> None:
    hash_before = hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest()

    agent = SimpleNamespace(train=True, logger=Mock())

    with pytest.raises(RuntimeError, match="frozen Task 1 baseline"):
        train.setup_training(agent)

    hash_after = hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest()

    assert hash_after == hash_before == EXPECTED_CHECKPOINT_SHA256


def test_frozen_files_are_in_agent_package(tmp_path: Path) -> None:
    output_path = tmp_path / "agent.zip"

    package_agent("DagobertDuckDQN", REPOSITORY_ROOT, output_path)

    prefix = "DagobertDuckDQN/"

    with zipfile.ZipFile(output_path) as archive:
        names = set(archive.namelist())
        archived_checkpoint = archive.read(prefix + "checkpoint.pt")

    required_names = {
        prefix + "README.md",
        prefix + "artifact.json",
        prefix + "baseline-config.yaml",
        prefix + "requirements.txt",
        prefix + "reference-results.csv",
        prefix + "callbacks.py",
        prefix + "config.py",
        prefix + "features.py",
        prefix + "model.py",
        prefix + "persistence.py",
        prefix + "replay.py",
        prefix + "rewards.py",
        prefix + "train.py",
        prefix + "checkpoint.pt",
    }

    assert required_names <= names
    assert hashlib.sha256(archived_checkpoint).hexdigest() == (
        EXPECTED_CHECKPOINT_SHA256
    )


def test_frozen_network_is_deterministic_and_greedy() -> None:
    """The frozen policy must select actions deterministically at eval time."""
    loaded = load_evaluation_checkpoint(CHECKPOINT_PATH)

    assert not loaded.network.training

    for parameter in loaded.network.parameters():
        assert not parameter.requires_grad

    probe_state = torch.zeros(DEFAULT_CONFIG.input_dim)

    with torch.no_grad():
        first = loaded.network(probe_state)
        second = loaded.network(probe_state)

    assert torch.equal(first, second)
