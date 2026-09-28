"""Tests for Issue #228 Task 4 baseline plots."""

from __future__ import annotations

import json
from pathlib import Path

from training.plot_issue228_task4_baseline import plot_results


def test_plot_results_creates_registered_figures(tmp_path: Path) -> None:
    summaries = [
        {
            "replica": f"r{replica}",
            "scenario": scenario,
            "mean_collection_fraction": 0.5,
            "self_kill_rate": 0.1,
            "survival_rate": 0.8,
            "first_place_rate": 0.1 if scenario == "competitive" else None,
            "tied_first_rate": 0.05 if scenario == "competitive" else None,
        }
        for replica in range(1, 6)
        for scenario in (
            "competitive",
            "mixed",
            "peaceful",
            "classic",
            "coin-heaven",
            "loot-crate",
        )
    ]
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps({"summaries": summaries}), encoding="utf-8")

    paths = plot_results(result_path, tmp_path / "figures")

    assert {path.name for path in paths} == {
        "competitive_outcomes.png",
        "capability_retention.png",
        "safety_and_survival.png",
    }
    assert all(path.read_bytes().startswith(b"\x89PNG") for path in paths)
