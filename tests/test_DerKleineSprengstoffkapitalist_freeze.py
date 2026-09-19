"""Contract tests for the frozen tabular Task 2 agent."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent_code.DerKleineSprengstoffkapitalist import callbacks, train
from agent_code.DerKleineSprengstoffkapitalist.persistence import load_model

MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "agent_code"
    / "DerKleineSprengstoffkapitalist"
    / "model.npz"
)
EXPECTED_SHA256 = "93470f92082597b1848f2fa65265c7288dbb1b50ac90370e5747e085e775ea3e"


def model_hash() -> str:
    return hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()


def test_frozen_model_integrity_and_configuration() -> None:
    assert MODEL_PATH.stat().st_size == 41780
    assert model_hash() == EXPECTED_SHA256

    loaded = load_model(MODEL_PATH)
    assert loaded.state_representation == "compact_decision"
    assert loaded.initialization == "zeros"
    assert loaded.action_masking == "none"
    assert loaded.potential_shaping == "none"
    assert loaded.exploration_mode == "standard"
    assert loaded.completed_episodes == 10000
    assert len(loaded.q_table) > 0


def test_evaluation_loads_without_treatment_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        callbacks.STATE_REPRESENTATION_ENV,
        callbacks.INITIALIZATION_ENV,
        callbacks.ACTION_MASKING_ENV,
        callbacks.POTENTIAL_SHAPING_ENV,
        callbacks.EXPLORATION_MODE_ENV,
        callbacks.USEFUL_BOMB_REWARD_ENV,
    ):
        monkeypatch.delenv(name, raising=False)

    before = model_hash()
    agent = SimpleNamespace(train=False, logger=Mock())
    callbacks.setup(agent)

    assert agent.state_representation == "compact_decision"
    assert agent.initialization == "zeros"
    assert agent.epsilon == 0.0
    assert len(agent.q_table) > 0
    assert model_hash() == before


def test_training_is_blocked_without_changing_model() -> None:
    before = model_hash()

    with pytest.raises(RuntimeError, match="frozen Task 2 agent"):
        train.setup_training(SimpleNamespace(train=True, logger=Mock()))

    assert model_hash() == before


def test_freeze_confirmation_evidence() -> None:
    evidence_path = MODEL_PATH.parent / "freeze-reference-results.csv"
    with evidence_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 600

    primary = {
        (row["scenario"], row["world_seed"], row["agent_seed"]): row
        for row in rows
        if row["phase"] == "primary"
    }
    repeats = [
        row for row in rows if row["phase"] == "repeat"
    ]
    assert len(primary) == len(repeats) == 300

    for row in repeats:
        key = (row["scenario"], row["world_seed"], row["agent_seed"])
        original = primary[key]
        for field in (
            "coins_collected",
            "collection_fraction",
            "self_kills",
            "action_bomb",
            "termination_reason",
            "executed_action_sequence_sha256",
        ):
            assert row[field] == original[field]

    classic = [row for row in primary.values() if row["scenario"] == "classic"]
    coin_heaven = [
        row for row in primary.values() if row["scenario"] == "coin-heaven"
    ]
    loot_crate = [
        row for row in primary.values() if row["scenario"] == "loot-crate"
    ]
    assert len(classic) == len(coin_heaven) == len(loot_crate) == 100

    for scenario_rows, expected in (
        (classic, 0.2133333333),
        (coin_heaven, 0.9716),
        (loot_crate, 0.2626),
    ):
        mean_collection = (
            sum(float(row["collection_fraction"]) for row in scenario_rows) / 100
        )
        assert mean_collection == pytest.approx(expected)

    assert sum(int(row["self_kills"]) for row in classic) == 13
    assert sum(int(row["action_bomb"]) for row in coin_heaven) == 33
    assert sum(int(row["self_kills"]) for row in loot_crate) == 28