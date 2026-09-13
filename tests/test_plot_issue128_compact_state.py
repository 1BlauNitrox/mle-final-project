"""Tests for the Issue #128 result plots."""

from __future__ import annotations

import json
from pathlib import Path

from training.plot_issue128_compact_state import plot_results


def test_plot_results_creates_both_figures(
    tmp_path: Path,
) -> None:
    summaries = []

    for treatment in ("control", "candidate"):
        for model_index in range(5):
            for scenario in (
                "classic",
                "coin-heaven",
                "loot-crate",
            ):
                summaries.append(
                    {
                        "treatment": treatment,
                        "model": f"r{model_index + 1}",
                        "scenario": scenario,
                        "mean_collection_fraction": (
                            0.4
                            if treatment == "candidate"
                            else 0.2
                        ),
                        "self_kill_rate": (
                            0.1
                            if treatment == "candidate"
                            else 0.01
                        ),
                    }
                )

    diagnostics = [
        {
            "treatment": treatment,
            "model": f"r{model_index + 1}",
            "q_table_size": (
                1000 if treatment == "candidate" else 4000
            ),
            "mean_visits_per_state": (
                600.0 if treatment == "candidate" else 150.0
            ),
            "singleton_state_fraction": (
                0.06 if treatment == "candidate" else 0.24
            ),
            "q_table_artifact_size_bytes": (
                40_000 if treatment == "candidate" else 90_000
            ),
        }
        for treatment in ("control", "candidate")
        for model_index in range(5)
    ]

    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "summaries": summaries,
                "training_diagnostics": diagnostics,
            }
        ),
        encoding="utf-8",
    )

    output_directory = tmp_path / "figures"
    paths = plot_results(result_path, output_directory)

    assert {path.name for path in paths} == {
        "performance_and_safety.png",
        "learning_efficiency.png",
    }

    for path in paths:
        assert path.is_file()
        assert path.stat().st_size > 0
        assert path.read_bytes().startswith(b"\x89PNG")