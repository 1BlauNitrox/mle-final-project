"""Regression tests for the registered reduced matrix and retained laptop evidence."""

import gzip
import json
from pathlib import Path

import pytest

from training.analyze_issue124_reduced import analyze, verify

EVIDENCE = (
    Path(__file__).resolve().parents[1]
    / "experiments/2026-09-10-task2-rehearsal-mask/laptop-results/evidence.json.gz"
)


@pytest.fixture(scope="module")
def evidence():
    with gzip.open(EVIDENCE, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def test_reduced_result_is_reproducible_and_d_cannot_be_selected(evidence):
    result, rows = analyze(evidence)
    assert len(rows) == 4880
    assert result["analysis_valid"]
    assert result["selection"]["cell"] == "A"
    assert result["selection"]["replica"] == "r5"
    assert result["selection"]["task2_complete"] is False
    assert result["absolute_gates"]["C"]["task2_passed"]
    assert not result["absolute_gates"]["C"]["task1_passed"]
    assert "D" not in result["eligibility"]
    assert result["comparisons"]["rehearsal_b_minus_a"]["paired_models"] == 5
    assert result["comparisons"]["mask_c_minus_a"]["mean_difference"] == pytest.approx(
        0.05333333333333334
    )


def test_missing_job_is_rejected(evidence, monkeypatch):
    status = evidence["records"]["laptop-evaluation/run-plans/issue124-reduced-eval-a/status.json"]
    monkeypatch.delitem(status["jobs"], next(iter(status["jobs"])))
    with pytest.raises(ValueError, match="Incomplete jobs"):
        verify(evidence)


def test_changed_seed_is_rejected(evidence, monkeypatch):
    key = next(
        k
        for k in evidence["records"]
        if k.startswith("laptop-evaluation/run-plans/") and k.endswith("/metadata.json")
    )
    monkeypatch.setitem(evidence["records"][key], "world_seed", -1)
    with pytest.raises(ValueError, match="Job condition changed"):
        verify(evidence)


def test_wrong_artifact_is_rejected(evidence, monkeypatch):
    key = next(iter(evidence["artifact_verification"]))
    monkeypatch.setitem(evidence["artifact_verification"][key], "sha256", "0" * 64)
    with pytest.raises(ValueError, match="Artifact checksum mismatch"):
        verify(evidence)


def test_resource_reset_is_rejected(evidence, monkeypatch):
    monkeypatch.setitem(
        evidence["records"]["laptop-evaluation/resources.json"], "cpu_seconds_consumed", 0
    )
    with pytest.raises(ValueError, match="CPU accounting reset"):
        verify(evidence)


def test_repeat_mismatch_invalidates_selection(evidence, monkeypatch):
    key = next(
        k
        for k in evidence["records"]
        if k.startswith("laptop-evaluation/run-plans/")
        and "-repeat-" in k
        and k.endswith("/episodes.csv")
    )
    import csv
    import io

    content = evidence["records"][key]
    reader = csv.DictReader(io.StringIO(content))
    row = next(reader)
    row["executed_action_sequence_sha256"] = "0" * 64
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=reader.fieldnames)
    writer.writeheader()
    writer.writerow(row)
    monkeypatch.setitem(evidence["records"], key, output.getvalue())
    result, _ = analyze(evidence)
    assert not result["analysis_valid"]
    assert result["selection"] is None
    assert result["determinism_mismatches"]
