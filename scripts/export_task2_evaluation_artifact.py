"""Export a Task 2 DQN checkpoint without training-only state."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from agent_code.DagobertDuckDQNTask2.persistence import (  # noqa: E402
    export_evaluation_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Resumable training checkpoint")
    parser.add_argument("destination", type=Path, help="Evaluation artifact to write")
    arguments = parser.parse_args()
    result = export_evaluation_checkpoint(arguments.source, arguments.destination)
    print(f"Created evaluation artifact {result}")


if __name__ == "__main__":
    main()
