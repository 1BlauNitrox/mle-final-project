"""Tests for the compact Issue #41 evidence bundle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import training.dqn_task1_evidence as evidence


def test_gzip_evidence_is_reproducible(tmp_path: Path) -> None:
    rows = [{"model": "run-01", "duration_ms": 1.25}]
    first = tmp_path / "first.csv.gz"
    second = tmp_path / "second.csv.gz"

    evidence._write_gzip_csv(first, rows)
    evidence._write_gzip_csv(second, rows)

    assert evidence._read_gzip_csv(first) == [
        {"model": "run-01", "duration_ms": "1.25"}
    ]
    assert first.read_bytes() == second.read_bytes()


def test_evidence_checksum_mismatch_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "observations.csv"
    path.write_text("original\n", encoding="utf-8")
    manifest = {
        "schema_version": evidence.EVIDENCE_SCHEMA_VERSION,
        "files": {
            path.name: {
                "sha256": evidence._sha256(path),
                "size_bytes": path.stat().st_size,
            }
        },
    }
    path.write_text("changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        evidence._validate_evidence_files(tmp_path, manifest)


def test_json_writer_rejects_non_finite_values(tmp_path: Path) -> None:
    path = tmp_path / "result.json"

    with pytest.raises(ValueError, match="Out of range float"):
        evidence._write_json(path, {"invalid": float("nan")})

    assert json.loads('{"valid": null}') == {"valid": None}


def test_write_csv_survives_a_clean_checkout(tmp_path: Path) -> None:
    """The manifest checksum is computed immediately after writing, before the
    file is ever committed. Git's `text=auto` normalizes any CRLF to LF the
    moment the file is committed, so if the writer emits CRLF (csv's own
    default line terminator, independent of platform), the checksum recorded
    here goes stale the instant the file is checked into git: every later
    checkout of the same content comes back a few bytes shorter than what was
    hashed, and `dqn_task1_evidence verify` reports a checksum mismatch on a
    perfectly clean checkout, as happened for issue #58.

    Pinning the writer to `\\n` up front means the bytes it hands to
    `_sha256` are already what git will store, so the round trip through
    commit and checkout is a no-op and the recorded checksum never drifts.
    """
    path = tmp_path / "evaluation-episodes.csv"

    evidence._write_csv(path, [{"model": "run-01", "world_seed": 31001}])

    written = path.read_bytes()
    assert b"\r\n" not in written
    assert written.count(b"\n") == 2  # header + one data row


def test_verify_evidence_rejects_a_manifest_from_another_issue(
    tmp_path: Path,
) -> None:
    """--issue selects which profile's models/seeds are used to rebuild the
    record; silently accepting evidence from a different issue would compare
    against the wrong agent seeds without any warning.
    """
    evidence_directory = tmp_path / "evidence"
    evidence_directory.mkdir()
    (evidence_directory / evidence.MANIFEST_FILE).write_text(
        json.dumps({"issue": 41}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="issue 41.*issue 58"):
        evidence.verify_evidence(tmp_path, evidence.PROFILES[58])
