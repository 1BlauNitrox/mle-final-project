"""Plot the recorded Issue #144 post-bomb escape experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import training.plot_issue135_escape_distance as base
from training.run_experiment import REPOSITORY_ROOT

EXPERIMENT = (
    REPOSITORY_ROOT
    / "experiments"
    / "2026-09-11-task2-post-bomb-escape-status-DerKleineSprengstoffkapitalist"
)
DEFAULT_RESULT = EXPERIMENT / "result.json"
DEFAULT_OUTPUT = EXPERIMENT / "figures"


def plot_results(
    result_path: Path = DEFAULT_RESULT, output_directory: Path = DEFAULT_OUTPUT
) -> list[Path]:
    """Create performance and learning-efficiency figures."""
    result = json.loads(Path(result_path).read_text(encoding="utf-8"))
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    original_labels = base.LABELS
    base.LABELS = {"control": "Compact", "candidate": "Escape status"}
    try:
        return [
            base._plot_performance(result["summaries"], output_directory),
            base._plot_efficiency(result["training_diagnostics"], output_directory),
        ]
    finally:
        base.LABELS = original_labels


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)
    for path in plot_results(arguments.result, arguments.output):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
