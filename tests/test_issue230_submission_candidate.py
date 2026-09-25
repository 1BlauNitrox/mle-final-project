"""Integrity checks for the tabular candidate evaluated by Issue #230."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent_code.DerKleineKonkurrenzvernichter import callbacks, train

ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "agent_code/DerKleineKonkurrenzvernichter"
MODEL = AGENT / "model.npz"
EXPECTED_SHA256 = "945b2cf0176b4ed57922aa6e12347f1ebbc19d675021dadb38eed6449075eb2d"


def _model_hash() -> str:
    return hashlib.sha256(MODEL.read_bytes()).hexdigest()


def test_confirmation_candidate_integrity() -> None:
    assert MODEL.stat().st_size == 197177
    assert _model_hash() == EXPECTED_SHA256

    artifact = json.loads((AGENT / "artifact.json").read_text(encoding="utf-8"))
    freeze = json.loads((AGENT / "freeze.json").read_text(encoding="utf-8"))
    assert artifact["artifact"]["sha256"] == EXPECTED_SHA256
    assert artifact["policy"]["training_allowed"] is False
    assert artifact["policy"]["registered_gates_passed"] is False
    assert freeze["artifact"]["sha256"] == EXPECTED_SHA256
    assert freeze["selection"]["selected_by"] == "explicit_human_deadline_override"
    assert freeze["selection"]["registered_gates_passed"] is False


def test_candidate_loads_in_evaluation_without_modification(
    monkeypatch,
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

    before = _model_hash()
    agent = SimpleNamespace(train=False, logger=Mock())
    callbacks.setup(agent)

    assert agent.state_representation == "compact_opponent"
    assert agent.initialization == "task2_prior"
    assert agent.epsilon == 0.0
    assert agent.completed_episodes == 10000
    assert len(agent.q_table) == 7909
    assert _model_hash() == before


def test_deadline_frozen_agent_rejects_training() -> None:
    before = _model_hash()

    with pytest.raises(RuntimeError, match="frozen final tabular agent"):
        train.setup_training(SimpleNamespace(train=True, logger=Mock()))

    assert _model_hash() == before


def test_evaluation_code_has_no_forbidden_runtime_imports() -> None:
    forbidden = {"training", "experiments"}
    for path in AGENT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(alias.name.split(".")[0] not in forbidden for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in forbidden
