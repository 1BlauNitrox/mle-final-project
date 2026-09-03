"""Contract tests for the frozen Task 1 baseline."""

from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from agent_code.DerKleineVermoegensumverteiler import train
from agent_code.DerKleineVermoegensumverteiler.config import (
    ACTIONS,
    DISCOUNT_FACTOR,
    LEARNING_RATE,
    REWARDS,
)
from agent_code.DerKleineVermoegensumverteiler.features import (
    FEATURE_SCHEMA_VERSION,
)
from agent_code.DerKleineVermoegensumverteiler.persistence import (
    MODEL_SCHEMA_VERSION,
    load_model,
)
from scripts.package_agent import package_agent

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AGENT_ROOT = REPOSITORY_ROOT / "agent_code" / "DerKleineVermoegensumverteiler"
MANIFEST_PATH = AGENT_ROOT / "artifact.json"
MODEL_PATH = AGENT_ROOT / "model.npz"
CONFIG_PATH = AGENT_ROOT / "baseline-config.yaml"
REFERENCE_PATH = AGENT_ROOT / "reference-results.csv"

EXPECTED_MODEL_SHA256 = "4e1da63a819ef8f51b112ffaf422ab251b853915375fe486538be8595b988307"


@pytest.fixture
def manifest() -> dict:
    with MANIFEST_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def test_frozen_artifact_integrity(manifest: dict) -> None:
    model_bytes = MODEL_PATH.read_bytes()

    assert len(model_bytes) == manifest["artifact"]["size_bytes"]
    assert hashlib.sha256(model_bytes).hexdigest() == (manifest["artifact"]["sha256"])
    assert manifest["artifact"]["sha256"] == EXPECTED_MODEL_SHA256


def test_frozen_model_loads_and_matches_schema(
    manifest: dict,
) -> None:
    loaded = load_model(MODEL_PATH)
    contract = manifest["model_contract"]

    assert len(loaded.q_table) > 0
    assert contract["model_schema_version"] == MODEL_SCHEMA_VERSION
    assert contract["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert contract["action_order"] == list(ACTIONS)
    assert contract["forbidden_actions"] == ["BOMB"]
    assert "BOMB" not in ACTIONS


def test_frozen_configuration_matches_agent() -> None:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        configuration = yaml.safe_load(file)

    hyperparameters = configuration["agent"]["hyperparameters"]
    rewards = configuration["agent"]["rewards"]

    assert hyperparameters["alpha"] == LEARNING_RATE
    assert hyperparameters["gamma"] == DISCOUNT_FACTOR
    assert rewards["COIN_COLLECTED"] == REWARDS["COIN_COLLECTED"]
    assert rewards["INVALID_ACTION"] == REWARDS["INVALID_ACTION"]
    assert rewards["WAITED"] == REWARDS["WAITED"]
    assert rewards["MOVED_TOWARDS_COIN"] == (REWARDS["MOVED_TOWARDS_COIN"])
    assert rewards["MOVED_AWAY_FROM_COIN"] == (REWARDS["MOVED_AWAY_FROM_COIN"])


def test_reference_results_match_selected_run() -> None:
    with REFERENCE_PATH.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 40
    assert [int(row["world_seed"]) for row in rows] == list(range(31001, 31041))
    assert {int(row["agent_seed"]) for row in rows} == {21006}
    assert sum(int(row["coins_collected"]) for row in rows) == 2000
    assert sum(int(row["episode_steps"]) for row in rows) == 5213
    assert sum(int(row["invalid_actions"]) for row in rows) == 2
    assert sum(int(row["action_bomb"]) for row in rows) == 0
    assert all(row["full_clear"] == "True" for row in rows)


def test_training_is_rejected_without_modifying_artifact() -> None:
    hash_before = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()

    agent = SimpleNamespace(
        train=True,
        logger=Mock(),
    )

    with pytest.raises(
        RuntimeError,
        match="frozen Task 1 baseline",
    ):
        train.setup_training(agent)

    hash_after = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()

    assert hash_after == hash_before == EXPECTED_MODEL_SHA256


def test_frozen_files_are_in_agent_package(tmp_path: Path) -> None:
    output_path = tmp_path / "agent.zip"

    package_agent(
        "DerKleineVermoegensumverteiler",
        REPOSITORY_ROOT,
        output_path,
    )

    prefix = "DerKleineVermoegensumverteiler/"

    with zipfile.ZipFile(output_path) as archive:
        names = set(archive.namelist())
        archived_model = archive.read(prefix + "model.npz")

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
        prefix + "model.npz",
        prefix + "persistence.py",
        prefix + "rewards.py",
        prefix + "train.py",
    }

    assert required_names <= names
    assert hashlib.sha256(archived_model).hexdigest() == (EXPECTED_MODEL_SHA256)
