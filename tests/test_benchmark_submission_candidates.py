"""Contract tests for the final checkpoint benchmark summary."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_submission_candidates.py"
SPEC = importlib.util.spec_from_file_location("submission_benchmark", SCRIPT)
assert SPEC and SPEC.loader
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


def test_report_recommends_warm_only_when_every_registered_gate_passes() -> None:
    candidates = ("fallback", "warm_lineup", "targeted_unqualified")
    suites = ("classic", "mixed", "coins", "crates")
    rows = []
    for candidate in candidates:
        for suite in suites:
            row = {
                "candidate": candidate,
                "suite": suite,
                "seed": 1,
                "score": 1,
                "kills": 1,
                "self_kills": 0,
                "survived": 1,
                "coins": 1,
                "collection_fraction": 0.2,
                "invalid": 0,
            }
            if candidate == "warm_lineup" and suite == "coins":
                row["collection_fraction"] = 0.3
            rows.append(row)
    cfg = {
        "suites": {suite: {} for suite in suites},
        "candidates": {
            "fallback": {},
            "warm_lineup": {},
            "targeted_unqualified": {"ineligibility_reason": "failed confirmation"}
        },
        "selection_rule": "human review",
    }

    report = BENCHMARK.report(rows, cfg)

    assert report["automatic_recommendation"] == "warm_lineup"
    assert all(report["warm_lineup_screen"].values())
    assert report["targeted_unqualified"] == "failed confirmation"
