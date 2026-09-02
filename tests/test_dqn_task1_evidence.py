"""Tests for the compact Issue #41 evidence bundle."""

from __future__ import annotations

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
