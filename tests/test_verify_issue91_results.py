"""Portable evidence checks reject corruption, missing models and escaping paths."""

import hashlib
import json

import pytest

from training.verify_issue91_results import verify_manifest


def test_manifest_checks_actual_bytes_and_size(tmp_path):
    (tmp_path / "analysis").mkdir()
    artifact = tmp_path / "checkpoint.pt"
    artifact.write_bytes(b"checkpoint fixture")
    manifest = {
        "checkpoint.pt": {
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "size_bytes": artifact.stat().st_size,
        }
    }
    (tmp_path / "analysis/evidence-files.json").write_text(json.dumps(manifest))
    verify_manifest(tmp_path)
    artifact.write_bytes(b"wrong model bytes!")
    with pytest.raises(ValueError, match="checkpoint.pt"):
        verify_manifest(tmp_path)


def test_missing_model_cannot_be_treated_as_verified(tmp_path):
    (tmp_path / "analysis").mkdir()
    (tmp_path / "analysis/evidence-files.json").write_text(
        json.dumps({"missing.pt": {"sha256": "0" * 64, "size_bytes": 1}})
    )
    with pytest.raises(ValueError, match="missing.pt"):
        verify_manifest(tmp_path)


def test_manifest_cannot_reference_paths_outside_campaign(tmp_path):
    (tmp_path / "analysis").mkdir()
    (tmp_path / "analysis/evidence-files.json").write_text(
        json.dumps({"../outside.pt": {"sha256": "0" * 64, "size_bytes": 1}})
    )
    with pytest.raises(ValueError, match="is_relative_to"):
        verify_manifest(tmp_path)
