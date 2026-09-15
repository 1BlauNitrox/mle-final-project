"""Contract checks for the descriptively selected Task 3 incumbent."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "agent_code" / "DagobertDuckDQNTask3"


def test_frozen_incumbent_manifest_matches_checkpoint() -> None:
    artifact = json.loads((AGENT / "artifact.json").read_text(encoding="utf-8"))
    freeze = json.loads((AGENT / "freeze.json").read_text(encoding="utf-8"))
    checkpoint = (AGENT / "checkpoint.pt").read_bytes()
    digest = hashlib.sha256(checkpoint).hexdigest()

    assert artifact["status"] == "task3_closed_incumbent_selected_descriptively"
    assert artifact["artifact"]["sha256"] == digest
    assert artifact["artifact"]["size_bytes"] == len(checkpoint)
    assert freeze["selection"]["decision"] == "descriptive_owner_selection"
    assert freeze["selection"]["gate_status"] == "not_passed"
    assert freeze["training"]["scientific_training_authorized"] is False
    assert freeze["artifact"]["sha256"] == digest
    assert freeze["artifact"]["previous_fixture_sha256"] != digest
