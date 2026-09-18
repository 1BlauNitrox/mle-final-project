"""Evaluate each new milestone of the continuation run, while it is still running.

The run produces a milestone every 1,000 episodes for all six jobs. This finds
the ones that are complete and not yet evaluated, plays them on the registered
100-world monitoring suite, and prints each arm against the agent the run
continues from. It is meant to be called on a schedule, so it is deliberately
cheap and idempotent:

  * the run root is only ever read, and milestones are copied to a staging root
  * a lock file stops two instances from writing the same output folder
  * evaluation resumes from games.jsonl, so an interrupted pass costs nothing
  * --jobs defaults to 1, which leaves the six training workers their cores

Every number here is paired per world against `reference.pt`, which for this run
is the episode-8,000 milestone the continuation started from. So "+0.3 on score"
means better than the agent we would otherwise ship, not better than untrained.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN = Path(r"C:\task4-warm-lineup")
REGISTRATION = REPO_ROOT / "experiments/2026-09-17-final-training-monitoring/config.json"
JOBS = ("control-r1", "control-r2", "control-r3", "hard-r1", "hard-r2", "hard-r3")
SETTLE_SECONDS = 60
METRICS = ("score", "coins", "kills", "self_kills", "survived", "collection_fraction", "invalid")


def complete_episodes(root: Path) -> list[int]:
    """Episodes every job has written, left alone long enough to be fully flushed."""
    per_job = []
    now = time.time()
    for job in JOBS:
        found = set()
        for path in (root / "training-resume" / job).glob("milestone-*.pt"):
            if now - path.stat().st_mtime < SETTLE_SECONDS:
                continue
            found.add(int(re.search(r"milestone-(\d+)\.pt$", path.name).group(1)))
        per_job.append(found)
    return sorted(set.intersection(*per_job)) if per_job else []


def evaluated_episodes(out: Path) -> set[int]:
    log = out / "games.jsonl"
    if not log.is_file():
        return set()
    episodes = set()
    for line in log.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            row = json.loads(line)
            if row["artifact"] != "reference":
                episodes.add(int(row["episode"]))
    return episodes


def stage(root: Path, staging: Path, episodes: list[int]) -> None:
    staging.mkdir(parents=True, exist_ok=True)
    reference = staging / "reference.pt"
    if not reference.is_file():
        shutil.copyfile(root / "reference.pt", reference)
    for job in JOBS:
        target = staging / "training-resume" / job
        target.mkdir(parents=True, exist_ok=True)
        for episode in episodes:
            name = f"milestone-{episode:06d}.pt"
            if not (target / name).is_file():
                shutil.copyfile(root / "training-resume" / job / name, target / name)


def evaluate(staging: Path, out: Path, episodes: list[int], jobs: int, registration: Path) -> int:
    command = [
        sys.executable, str(REPO_ROOT / "scripts/evaluate_milestones.py"),
        "--root", str(staging), "--registration", str(registration),
        "--agent", "Bomb-omb", "--jobs", str(jobs), "--out", str(out),
        "--episodes", *[str(e) for e in episodes],
    ]
    environment = {**os.environ, "BOMBERMAN_EVALUATION_VARIANT": "warm-lineup"}
    return subprocess.run(command, cwd=REPO_ROOT, env=environment).returncode


def summarise(out: Path) -> None:
    rows = [json.loads(line) for line in (out / "games.jsonl").read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]
    by_artifact = defaultdict(dict)
    for row in rows:
        by_artifact[row["artifact"]][row["world_seed"]] = row
    worlds = sorted(by_artifact["reference"])
    episodes = sorted({int(a.split("@")[1]) for a in by_artifact if a != "reference"})
    rng = np.random.default_rng(20260919)

    print(f"\nagainst the episode-8,000 agent this run continues from, paired on {len(worlds)} worlds")
    print(f"{'episode':>8}{'arm':>9}" + "".join(f"{m:>22}" for m in ("score", "coins", "kills", "self_kills")))
    for episode in episodes:
        for arm in ("control", "hard"):
            candidates = [a for a in by_artifact if a.startswith(arm) and a.endswith(f"@{episode}")]
            if not candidates:
                continue
            cells = []
            for metric in ("score", "coins", "kills", "self_kills"):
                diffs = np.array([
                    np.mean([by_artifact[c][w][metric] for c in candidates if w in by_artifact[c]])
                    - by_artifact["reference"][w][metric]
                    for w in worlds
                    if all(w in by_artifact[c] for c in candidates)
                ])
                draws = np.array([diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(2000)])
                low, high = np.percentile(draws, [2.5, 97.5])
                mark = "" if low <= 0 <= high else "*"
                cells.append(f"{diffs.mean():+.3f} [{low:+.2f},{high:+.2f}]{mark}".rjust(22))
            print(f"{episode:>8}{arm:>9}" + "".join(cells))
    print("\n* marks an interval that clears zero. Kills are the metric this run has to move.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=RUN)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--registration", type=Path, default=REGISTRATION)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--max-episodes", type=int, default=2,
                        help="Evaluate at most this many new milestone levels per call.")
    args = parser.parse_args()

    lock = args.out.parent / f".{args.out.name}.lock"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if lock.is_file() and time.time() - lock.stat().st_mtime < 3 * 3600:
        print(f"ANOTHER PASS IS RUNNING (lock {lock}); doing nothing")
        return 0

    complete = complete_episodes(args.root)
    done = evaluated_episodes(args.out)
    pending = [e for e in complete if e not in done][: args.max_episodes]
    print(f"complete milestones: {complete} | already evaluated: {sorted(done)} | to evaluate: {pending}")
    if not pending:
        print("NO NEW MILESTONES")
        if (args.out / "games.jsonl").is_file():
            summarise(args.out)
        return 0

    lock.write_text(str(os.getpid()), encoding="utf-8")
    try:
        stage(args.root, args.staging, pending)
        code = evaluate(args.staging, args.out, pending, args.jobs, args.registration)
        if code != 0:
            print(f"EVALUATION FAILED with exit code {code}; the next pass retries it")
            return code
    finally:
        lock.unlink(missing_ok=True)

    print(f"EVALUATED {pending}")
    summarise(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
