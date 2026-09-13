"""Tests for the Issue #144 plots."""

from __future__ import annotations

import json
from pathlib import Path

from training.plot_issue144_post_bomb_escape import plot_results


def test_plot_results_creates_both_figures(tmp_path: Path) -> None:
    summaries = [
        {
            "treatment": treatment,
            "model": f"r{model}",
            "scenario": scenario,
            "mean_collection_fraction": 0.4,
            "self_kill_rate": 0.02,
        }
        for treatment in ("control", "candidate")
        for model in range(1, 6)
        for scenario in ("classic", "coin-heaven", "loot-crate")
    ]
    diagnostics = [
        {
            "treatment": treatment,
            "model": f"r{model}",
            "q_table_size": 1000,
            "mean_visits_per_state": 500.0,
            "singleton_state_fraction": 0.05,
            "q_table_artifact_size_bytes": 40000,
        }
        for treatment in ("control", "candidate")
        for model in range(1, 6)
    ]
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps({"summaries": summaries, "training_diagnostics": diagnostics}),
        encoding="utf-8",
    )
    paths = plot_results(result, tmp_path / "figures")
    assert {path.name for path in paths} == {
        "performance_and_safety.png",
        "learning_efficiency.png",
    }
    assert all(path.read_bytes().startswith(b"\x89PNG") for path in paths)
