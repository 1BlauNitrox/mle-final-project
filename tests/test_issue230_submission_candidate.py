"""Integrity checks for the tabular candidate evaluated by Issue #230."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from agent_code.DerKleineKonkurrenzvernichter import callbacks

ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "agent_code/DerKleineKonkurrenzvernichter"
MODEL = AGENT / "model.npz"
EXPECTED_SHA256 = "9ef02537efd75b70cbfdd5f973d391ed4924bec91735e07a51b866f3beeee031"


def _model_hash() -> str:
    return hashlib.sha256(MODEL.read_bytes()).hexdigest()


def test_confirmation_candidate_integrity() -> None:
    assert MODEL.stat().st_size == 1493
    assert _model_hash() == EXPECTED_SHA256


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
