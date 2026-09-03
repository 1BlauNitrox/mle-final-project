"""Regression tests for the successor's checkpoint-replace retry.

Ported from tests/test_DagobertDuckDQN_persistence.py (issue #71) since the
successor's persistence.py was copied from the parent before that fix
merged into main; this pins that the copy actually carries it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import agent_code.DagobertDuckDQNTask2.persistence as persistence


def test_checkpoint_replace_retries_transient_windows_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A brief antivirus lock must not end a multi-hour training run."""
    source = tmp_path / "source.tmp"
    destination = tmp_path / "checkpoint.pt"
    source.write_bytes(b"payload")
    attempts = 0
    original_replace = persistence.os.replace

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
