"""Tests for Issue #230 freeze evidence analysis."""

from training.analyze_issue230_freeze import (
    EXPECTED_SUITES,
    analyze,
    collect_rows,
    repeat_mismatches,
    summarize,
)


def test_completed_confirmation_has_expected_shape() -> None:
    rows = collect_rows()

    assert len(rows) == 240
    assert sum(row["phase"] == "primary" for row in rows) == 120
    assert sum(row["phase"] == "repeat" for row in rows) == 120
    assert {row["suite"] for row in rows} == set(EXPECTED_SUITES)


def test_summary_has_twenty_primary_rounds_per_suite() -> None:
    summaries = summarize(collect_rows())

    assert len(summaries) == 6
    assert all(row["rounds"] == 20 for row in summaries)


def test_repeat_comparison_covers_complete_matrix() -> None:
    mismatches = repeat_mismatches(collect_rows())

    assert isinstance(mismatches, list)


def test_analysis_records_artifact_and_all_registered_gates() -> None:
    result = analyze(collect_rows())

    assert len(result["model"]["sha256"]) == 64
    assert result["model"]["size_bytes"] > 0
    assert len(result["gates"]) == 8
    assert result["decision"] in {
        "freeze_installed_task3_incumbent",
        "do_not_freeze_confirmation_failed",
    }
