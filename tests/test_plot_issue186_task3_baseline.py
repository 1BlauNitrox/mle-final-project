"""Tests for the Issue #186 result plots."""

from __future__ import annotations

import json
from pathlib import Path

from training.plot_issue186_task3_baseline import plot_results


def test_plot_results_creates_both_figures(tmp_path: Path) -> None:
    summaries = [
        {
            "replica": f"r{replica}",
            "scenario": scenario,
            "mean_collection_fraction": 0.5,
            "mean_opponents_eliminated": 0.1 if scenario == "peaceful" else 0.0,
            "self_kill_rate": 0.05,
        }
        for replica in range(1, 6)
        for scenario in ("peaceful", "classic", "coin-heaven", "loot-crate")
    ]
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps({"summaries": summaries}), encoding="utf-8")

    paths = plot_results(result_path, tmp_path / "figures")

    assert {path.name for path in paths} == {
        "peaceful_hunting.png",
        "retention_and_safety.png",
    }
    for path in paths:
        assert path.read_bytes().startswith(b"\x89PNG")
