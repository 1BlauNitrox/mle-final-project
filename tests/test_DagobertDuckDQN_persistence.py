"""Tests for DagobertDuckDQN checkpoint persistence."""

import os
from pathlib import Path

import pytest
import torch

import agent_code.DagobertDuckDQN.persistence as persistence
from agent_code.DagobertDuckDQN.config import DEFAULT_CONFIG
from agent_code.DagobertDuckDQN.model import build_q_network
from agent_code.DagobertDuckDQN.persistence import (
    CHECKPOINT_PATH,
    load_evaluation_checkpoint,
    save_evaluation_artifact,
)


def test_checkpoint_path_is_relative_to_agent_module() -> None:
    assert CHECKPOINT_PATH.parent == Path(
        __file__
    ).resolve().parents[1] / "agent_code" / "DagobertDuckDQN"


def test_evaluation_checkpoint_round_trip(tmp_path: Path) -> None:
    network = build_q_network(DEFAULT_CONFIG, seed=123)
    path = tmp_path / "checkpoint.pt"

    save_evaluation_artifact(
        network=network,
        config=DEFAULT_CONFIG,
        completed_episodes=3,
        path=path,
    )
    loaded = load_evaluation_checkpoint(path)

    assert loaded.config == DEFAULT_CONFIG
    assert loaded.completed_episodes == 3
    assert not loaded.network.training
    assert all(
        not parameter.requires_grad
        for parameter in loaded.network.parameters()
    )

    for original, restored in zip(
        network.parameters(),
        loaded.network.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(original, restored)


def test_evaluation_loading_does_not_modify_checkpoint(tmp_path: Path) -> None:
    network = build_q_network(DEFAULT_CONFIG, seed=123)
    path = tmp_path / "checkpoint.pt"

    save_evaluation_artifact(
        network=network,
        config=DEFAULT_CONFIG,
        completed_episodes=3,
        path=path,
    )
    bytes_before = path.read_bytes()

    load_evaluation_checkpoint(path)

    assert path.read_bytes() == bytes_before


def test_missing_checkpoint_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_evaluation_checkpoint(tmp_path / "missing.pt")


def test_incompatible_schema_is_rejected(tmp_path: Path) -> None:
    network = build_q_network(DEFAULT_CONFIG, seed=123)
    path = tmp_path / "checkpoint.pt"

    save_evaluation_artifact(
        network=network,
        config=DEFAULT_CONFIG,
        completed_episodes=3,
        path=path,
    )

    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )
    payload["artifact_schema_version"] = 999
    torch.save(payload, path)

    with pytest.raises(ValueError):
        load_evaluation_checkpoint(path)


def test_save_rejects_negative_completed_episodes(tmp_path: Path) -> None:
    network = build_q_network(DEFAULT_CONFIG, seed=123)

    with pytest.raises(ValueError):
        save_evaluation_artifact(
            network=network,
            config=DEFAULT_CONFIG,
            completed_episodes=-1,
            path=tmp_path / "checkpoint.pt",
        )


def test_failed_save_preserves_existing_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    network = build_q_network(DEFAULT_CONFIG, seed=123)
    path = tmp_path / "checkpoint.pt"
    original_bytes = b"existing-checkpoint"
    path.write_bytes(original_bytes)

    def fail_during_serialization(*args, **kwargs) -> None:
        del args, kwargs
        raise RuntimeError("simulated serialization failure")

    monkeypatch.setattr(
        "agent_code.DagobertDuckDQN.persistence.torch.save",
        fail_during_serialization
    )

    with pytest.raises(RuntimeError, match="simulated"):
        save_evaluation_artifact(
            network=network,
            config=DEFAULT_CONFIG,
            completed_episodes=3,
            path=path,
        )

    assert path.read_bytes() == original_bytes
    assert list(tmp_path.iterdir()) == [path]


def test_checkpoint_replace_retries_transient_windows_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A brief antivirus lock must not end a multi-hour training run."""
    source = tmp_path / "source.tmp"
    destination = tmp_path / "checkpoint.pt"
    source.write_bytes(b"payload")
    attempts = 0
    original_replace = os.replace

    def flaky_replace(a, b):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("temporarily locked")
        return original_replace(a, b)

    monkeypatch.setattr(persistence.os, "replace", flaky_replace)
    monkeypatch.setattr(persistence, "sleep", lambda _seconds: None)

    persistence._replace_with_retry(source, destination)

    assert attempts == 3
    assert destination.read_bytes() == b"payload"


def test_checkpoint_replace_gives_up_after_the_registered_attempts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    attempts = 0

    def always_locked(a, b):
        nonlocal attempts
        attempts += 1
        raise PermissionError("locked")

    monkeypatch.setattr(persistence.os, "replace", always_locked)
    monkeypatch.setattr(persistence, "sleep", lambda _seconds: None)

    with pytest.raises(PermissionError):
        persistence._replace_with_retry(tmp_path / "a", tmp_path / "b")

    assert attempts == persistence.CHECKPOINT_REPLACE_ATTEMPTS
