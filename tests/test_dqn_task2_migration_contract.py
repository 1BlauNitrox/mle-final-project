"""Tests for the preregistered Issue #85 Q-value contract analyzer."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from training.dqn_task2_migration_contract import (
    create_probe_report,
    main,
    verify_probe_report,
)


def test_probe_report_is_complete_and_verifiable(tmp_path) -> None:
    output = tmp_path / "report.json"

    assert main(["--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))

    assert len(report["probes"]) == 4
    verify_probe_report(report)


@pytest.mark.parametrize("mutation", ["missing", "altered"])
def test_probe_report_rejects_missing_or_altered_evidence(mutation: str) -> None:
    report = deepcopy(create_probe_report())

    if mutation == "missing":
        del report["probes"][0]["corrected_q_values"]
    else:
        report["probes"][0]["bomb_q_value"] += 1.0

    with pytest.raises(ValueError):
        verify_probe_report(report)
