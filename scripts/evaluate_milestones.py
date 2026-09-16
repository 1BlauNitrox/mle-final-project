"""Evaluate retained milestone checkpoints so a long run can be watched as it trains.

This is monitoring, not the registered evaluation. It reads the development
worlds only; the held-out suite is never touched here, so nothing this script
prints can leak into the number the report quotes.

Every checkpoint is played on the same worlds from the same base seed, and the
unchanged reference is played on those worlds too, because absolute scores are
not comparable across world sets - the unchanged incumbent has scored anywhere
between 1.375 and 3.100 on different forty-world draws.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGED_NAME = "eval-milestone.pt"


def discover(root: Path, latest_only: bool) -> list[tuple[str, int, Path]]:
    """Return (job, episode, path) for every retained milestone."""
    found = []
    for job_dir in sorted((root / "training-resume").glob("*")):
        if not job_dir.is_dir():
            continue
        milestones = sorted(job_dir.glob("milestone-*.pt"))
        if latest_only and milestones:
            milestones = milestones[-1:]
        for path in milestones:
            match = re.search(r"milestone-(\d+)\.pt$", path.name)
            if match:
                found.append((job_dir.name, int(match.group(1)), path))
    return found


def play(agent: str, checkpoint: Path, seed: int, rounds: int, opponents: list[str]) -> dict:
    agent_dir = REPO_ROOT / "agent_code" / agent
    staged = agent_dir / STAGED_NAME
    staged.write_bytes(checkpoint.read_bytes())

    env = {**os.environ, "BOMBERMAN_EVALUATION_CHECKPOINT": STAGED_NAME}
    handle, stats_path = tempfile.mkstemp(suffix=".json")
    os.close(handle)
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "main.py",
                "play",
                "--agents",
                agent,
                *opponents,
                "--n-rounds",
                str(rounds),
                "--seed",
                str(seed),
                "--no-gui",
                "--save-stats",
                stats_path,
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"play failed: {(proc.stdout + proc.stderr)[-600:]}")
        data = json.loads(Path(stats_path).read_text(encoding="utf-8"))
    finally:
        staged.unlink(missing_ok=True)
        Path(stats_path).unlink(missing_ok=True)

    by_agent = data["by_agent"]
    key = next((k for k in by_agent if k.startswith(agent)), None)
    if key is None:
        raise RuntimeError(f"agent {agent} absent from stats: {list(by_agent)}")
    entry = by_agent[key]
    played = entry.get("rounds", rounds) or rounds
    return {
        "rounds": played,
        "score": entry.get("score", 0) / played,
        "kills": entry.get("kills", 0) / played,
        "self_kills": entry.get("suicides", 0) / played,
        "coins": entry.get("coins", 0) / played,
        "invalid": entry.get("invalid", 0) / played,
        "survived": entry.get("survived", 0) / played,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--agent", default="Bomb-omb")
    parser.add_argument("--seed", type=int, default=930100001)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--latest-only", action="store_true")
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--opponents",
        nargs="*",
        default=["rule_based_agent", "rule_based_agent", "rule_based_agent"],
    )
    args = parser.parse_args()

    root = args.root.resolve()
    milestones = discover(root, args.latest_only)
    if not milestones:
        print(f"no milestones retained yet under {root / 'training-resume'}")
        return 0

    rows = []
    if not args.skip_reference and (root / "reference.pt").is_file():
        stats = play(args.agent, root / "reference.pt", args.seed, args.rounds, args.opponents)
        rows.append({"job": "reference", "episode": 0, **stats})
        print(
            f"  reference          score={stats['score']:.3f} kills={stats['kills']:.3f} "
            f"self_kills={stats['self_kills']:.3f} survived={stats['survived']:.3f}"
        )

    for job, episode, path in milestones:
        stats = play(args.agent, path, args.seed, args.rounds, args.opponents)
        rows.append({"job": job, "episode": episode, **stats})
        print(
            f"  {job:<12} {episode:>6}  score={stats['score']:.3f} kills={stats['kills']:.3f} "
            f"self_kills={stats['self_kills']:.3f} survived={stats['survived']:.3f}"
        )

    out = args.out or (root / "milestone-monitoring.csv")
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print()
    print(f"worlds : {args.rounds} rounds from base seed {args.seed} (development, not held out)")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
