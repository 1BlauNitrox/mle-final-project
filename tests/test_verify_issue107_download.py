"""Integrity checks for cross-platform Issue #107 evidence verification."""

import hashlib
import json

import pytest

from scripts.verify_issue107_download import (
    check_dependency_record,
    contained,
    digest,
    linux_directory_digest,
)


def test_linux_digest_uses_case_sensitive_order_and_ignores_runtime_files(tmp_path):
    (tmp_path / "README.md").write_bytes(b"readme")
    (tmp_path / "artifact.json").write_bytes(b"artifact")
    (tmp_path / ".evaluation-checkpoint.pt").write_bytes(b"temporary")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "cache.pyc").write_bytes(b"cache")
    expected = hashlib.sha256(b"README.md\0readme\0artifact.json\0artifact\0").hexdigest()
    assert linux_directory_digest(tmp_path) == expected
    (tmp_path / "artifact.json").write_bytes(b"tampered")
    assert linux_directory_digest(tmp_path) != expected


def test_dependency_inventory_cannot_change_without_invalidating_hash():
    record = {"python": "3.14.4", "numpy": "2.5.3"}
    fingerprints = {
        "dependencies": record,
        "dependencies_sha256": hashlib.sha256(
            json.dumps(record, sort_keys=True).encode()
        ).hexdigest(),
    }
    assert check_dependency_record(fingerprints) == record
    record["numpy"] = "changed"
    with pytest.raises(ValueError, match="checksum mismatch"):
        check_dependency_record(fingerprints)


def test_evidence_cannot_read_outside_its_plan(tmp_path):
    assert contained(tmp_path, "jobs/job/metadata.json").is_relative_to(tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        contained(tmp_path, "../outside.json")


def test_cached_checksum_does_not_hide_changed_file(tmp_path):
    path = tmp_path / "artifact.pt"
    path.write_bytes(b"original")
    first = digest(path)
    assert digest(path) == first
    path.write_bytes(b"changed and longer")
    assert digest(path) != first
