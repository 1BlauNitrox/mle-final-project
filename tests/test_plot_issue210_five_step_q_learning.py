"""Tests for the Issue #210 result plots."""

from __future__ import annotations

import json
from pathlib import Path

from training.plot_issue210_five_step_q_learning import plot_results


def test_plot_results_creates_three_figures(tmp_path: Path) -> None:
    summaries = [
        {
            "treatment": treatment,
            "replica": f"r{replica}",
            "scenario": scenario,
            "mean_collection_fraction": 0.5,
            "mean_opponents_eliminated": 0.2,
            "self_kill_rate": 0.05,
        }
        for treatment in ("control", "candidate")
        for replica in range(1, 6)
        for scenario in (
            "peaceful",
            "coincollector",
            "classic",
            "coin-heaven",
            "loot-crate",
        )
    ]
    diagnostics = [
        {
            "treatment": treatment,
            "q_table_size": 100,
            "mean_visits_per_state": 2.0,
            "singleton_state_fraction": 0.5,
        }
        for treatment in ("control", "candidate")
        for _ in range(5)
    ]
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps({"summaries": summaries, "training_diagnostics": diagnostics}),
        encoding="utf-8",
    )

    paths = plot_results(result_path, tmp_path / "figures")

    assert {path.name for path in paths} == {
        "peaceful_hunting_comparison.png",
        "retention_and_safety.png",
        "learning_efficiency.png",
    }
    assert all(path.read_bytes().startswith(b"\x89PNG") for path in paths)
