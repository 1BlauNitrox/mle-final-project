"""Watch any checkpoint play, with the GUI on.

The agent resolves its evaluation policy through BOMBERMAN_EVALUATION_CHECKPOINT,
which must be a bare file name inside the agent directory. This copies the chosen
checkpoint in under a temporary name, plays, and removes it again, so watching a
mid-training milestone never disturbs the installed policy.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WATCH_NAME = "watch-checkpoint.pt"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Checkpoint to watch (any path).")
    parser.add_argument("--agent", default="DagobertDuckDQNTask3")
    parser.add_argument(
        "--opponents",
        nargs="*",
        default=["rule_based_agent", "rule_based_agent", "rule_based_agent"],
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--scenario", default="classic")
    parser.add_argument(
        "--no-gui", action="store_true", help="Run headless (for scoring, not watching)."
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        print(f"FAIL: checkpoint not found: {checkpoint}", file=sys.stderr)
        return 1

    agent_dir = REPO_ROOT / "agent_code" / args.agent
    if not agent_dir.is_dir():
        print(f"FAIL: agent directory not found: {agent_dir}", file=sys.stderr)
        return 1

    staged = agent_dir / WATCH_NAME
    staged.write_bytes(checkpoint.read_bytes())

    env = dict(os.environ)
    env["BOMBERMAN_EVALUATION_CHECKPOINT"] = WATCH_NAME

    command = [
        sys.executable,
        "main.py",
        "play",
        "--agents",
        args.agent,
        *args.opponents,
        "--n-rounds",
        str(args.rounds),
        "--scenario",
        args.scenario,
    ]
    if args.no_gui:
        command.append("--no-gui")

    print(f"watching   : {checkpoint}")
    print(f"as agent   : {args.agent}")
    print(f"opponents  : {' '.join(args.opponents)}")
    print()

    try:
        return subprocess.run(command, cwd=REPO_ROOT, env=env).returncode
    finally:
        staged.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
