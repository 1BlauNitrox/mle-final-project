"""Evaluate retained milestone checkpoints so a long run can be watched as it trains.

This is monitoring, not the registered evaluation. Every checkpoint plays the
development worlds of the registered classic-rule-based suite, one game per
world, and the unchanged reference plays the same worlds, because absolute
scores are not comparable across world sets. The held-out suite is never read.

Each game is its own process with its own --seed. The framework draws from the
world generator every step to shuffle move order over however many agents are
still alive, so a multi-round game seeded once hands different checkpoints
different arenas from round two onward. Opponents are not seeded, so paired
worlds still carry some behavioural noise.

Games are appended to games.jsonl as they finish and skipped on a rerun, so an
interrupted evaluation resumes where it stopped.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "experiments/2026-09-17-task4-final-training/config.json"
METRICS = ("score", "kills", "self_kills", "survived", "coins", "collection_fraction", "invalid")
REFERENCE = "reference"


def discover(root: Path, latest_only: bool, episodes: set[int] | None) -> dict[str, Path]:
    found = {}
    if (root / "reference.pt").is_file():
        found[REFERENCE] = root / "reference.pt"
    for job_dir in sorted((root / "training-resume").glob("*")):
        milestones = []
        for path in sorted(job_dir.glob("milestone-*.pt")):
            episode = int(re.search(r"milestone-(\d+)\.pt$", path.name).group(1))
            if episodes is None or episode in episodes:
                milestones.append((episode, path))
        if latest_only:
            milestones = milestones[-1:]
        for episode, path in milestones:
            found[f"{job_dir.name}@{episode}"] = path
    return found


def arm_and_episode(artifact: str) -> tuple[str, int]:
    if artifact == REFERENCE:
        return REFERENCE, 0
    job, episode = artifact.split("@")
    return job.rsplit("-r", 1)[0], int(episode)


def round_record(stats: dict, agent: str) -> dict:
    rounds = list(stats["by_round"].values())
    if len(rounds) != 1:
        raise ValueError(f"expected one round per game, got {len(rounds)}")
    agents = rounds[0]["agents"]
    name = next((key for key in agents if key.startswith(agent)), None)
    if name is None:
        raise ValueError(f"{agent} absent from round statistics: {list(agents)}")
    entry = agents[name]
    available = entry.get("initially_available_coins") or 0
    return {
        "score": entry["score"],
        "kills": entry["kills"],
        "self_kills": entry["self_kills"],
        "survived": int(bool(entry["survived"])),
        "survival_steps": entry["survival_steps"],
        "coins": entry["coins"],
        "collection_fraction": entry["coins"] / available if available else None,
        "invalid": entry["invalid"],
        "termination_reason": entry["termination_reason"],
    }


def play(agent: str, staged: str, seed: int, opponents: list[str]) -> dict:
    handle, stats_path = tempfile.mkstemp(suffix=".json")
    os.close(handle)
    try:
        proc = subprocess.run(
            [
                sys.executable, "main.py", "play",
                "--agents", agent, *opponents,
                "--n-rounds", "1", "--seed", str(seed),
                "--no-gui", "--save-stats", stats_path,
            ],
            cwd=REPO_ROOT,
            env={**os.environ, "BOMBERMAN_EVALUATION_CHECKPOINT": staged},
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"seed {seed} failed: {(proc.stdout + proc.stderr)[-800:]}")
        return round_record(json.loads(Path(stats_path).read_text(encoding="utf-8")), agent)
    finally:
        Path(stats_path).unlink(missing_ok=True)


def paired_difference(games: list[dict], artifacts: list[str], metric: str,
                      resamples: int = 5000, seed: int = 20260917) -> dict | None:
    """Mean over worlds of (candidate - reference), pooling the given artifacts per world."""
    by_world = defaultdict(dict)
    for game in games:
        by_world[game["world_seed"]][game["artifact"]] = game[metric]
    diffs = []
    for values in by_world.values():
        reference = values.get(REFERENCE)
        pooled = [values[a] for a in artifacts if values.get(a) is not None]
        if reference is None or len(pooled) != len(artifacts):
            continue
        diffs.append(mean(pooled) - reference)
    if not diffs:
        return None
    rng = random.Random(seed)
    boots = sorted(
        mean(rng.choice(diffs) for _ in diffs) for _ in range(resamples)
    )
    return {
        "worlds": len(diffs),
        "mean_difference": mean(diffs),
        "ci_low": boots[int(0.025 * resamples)],
        "ci_high": boots[int(0.975 * resamples) - 1],
    }


def summarize(games: list[dict], out: Path) -> None:
    by_artifact = defaultdict(list)
    for game in games:
        by_artifact[game["artifact"]].append(game)

    with (out / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["artifact", "arm", "episode", "games", *METRICS])
        for artifact in sorted(by_artifact, key=lambda a: arm_and_episode(a)[::-1]):
            rows = by_artifact[artifact]
            arm, episode = arm_and_episode(artifact)
            means = []
            for metric in METRICS:
                values = [row[metric] for row in rows if row[metric] is not None]
                means.append(round(mean(values), 4) if values else "")
            writer.writerow([artifact, arm, episode, len(rows), *means])

    groups = {artifact: [artifact] for artifact in by_artifact if artifact != REFERENCE}
    pooled = defaultdict(list)
    for artifact in by_artifact:
        if artifact != REFERENCE:
            arm, episode = arm_and_episode(artifact)
            pooled[f"{arm}@{episode} (pooled)"].append(artifact)
    groups.update(pooled)

    with (out / "paired.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["candidate", "metric", "worlds", "mean_difference_vs_reference", "ci95_low", "ci95_high"])
        for name in sorted(groups):
            for metric in ("score", "kills", "self_kills", "survived", "collection_fraction"):
                result = paired_difference(games, groups[name], metric)
                if result:
                    writer.writerow([
                        name, metric, result["worlds"], round(result["mean_difference"], 4),
                        round(result["ci_low"], 4), round(result["ci_high"], 4),
                    ])

    print()
    print(f"{'artifact':<24}{'games':>6}{'score':>8}{'kills':>7}{'selfkill':>9}{'survived':>9}{'coins':>7}")
    with (out / "summary.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            print(f"{row['artifact']:<24}{row['games']:>6}{row['score']:>8}{row['kills']:>7}"
                  f"{row['self_kills']:>9}{row['survived']:>9}{row['coins']:>7}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True,
                        help="Directory holding reference.pt and training-resume/*/milestone-*.pt")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--suite", default="classic-rule-based")
    parser.add_argument("--worlds", type=int, default=40)
    parser.add_argument("--agent", default="Bomb-omb")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--latest-only", action="store_true")
    parser.add_argument("--episodes", type=int, nargs="*")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if args.suite not in cfg["evaluation_suites"] or "holdout" in args.suite:
        raise SystemExit(f"refusing suite {args.suite!r}: monitoring reads development suites only")
    suite = cfg["evaluation_suites"][args.suite]
    seeds = suite["world_seeds"][: args.worlds]
    opponents = suite["opponents"]

    root = args.root.resolve()
    out = (args.out or root / "milestone-evaluation").resolve()
    out.mkdir(parents=True, exist_ok=True)
    artifacts = discover(root, args.latest_only, set(args.episodes) if args.episodes else None)
    if REFERENCE not in artifacts:
        raise SystemExit(f"no reference.pt under {root}: nothing to pair against")
    if len(artifacts) == 1:
        raise SystemExit(f"no milestone checkpoints under {root / 'training-resume'}")

    log = out / "games.jsonl"
    games = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.is_file() else []
    done = {(g["artifact"], g["world_seed"]) for g in games}
    pending = [(a, s) for a in artifacts for s in seeds if (a, s) not in done]
    print(f"artifacts: {len(artifacts)}  worlds: {len(seeds)}  games done: {len(done)}  pending: {len(pending)}")

    agent_dir = REPO_ROOT / "agent_code" / args.agent
    staged = {}
    for artifact, path in artifacts.items():
        name = "eval-" + re.sub(r"[^A-Za-z0-9_-]", "-", artifact) + ".pt"
        (agent_dir / name).write_bytes(path.read_bytes())
        staged[artifact] = name

    lock = threading.Lock()
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            futures = {pool.submit(play, args.agent, staged[a], s, opponents): (a, s) for a, s in pending}
            for index, future in enumerate(as_completed(futures), 1):
                artifact, seed = futures[future]
                arm, episode = arm_and_episode(artifact)
                row = {"artifact": artifact, "arm": arm, "episode": episode, "world_seed": seed, **future.result()}
                with lock, log.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                games.append(row)
                if index % 20 == 0 or index == len(pending):
                    print(f"  {index}/{len(pending)} games", flush=True)
    finally:
        for name in staged.values():
            (agent_dir / name).unlink(missing_ok=True)

    summarize(games, out)
    print()
    print(f"worlds : {args.suite} development seeds {seeds[0]}..{seeds[-1]} (not held out)")
    print(f"results: {out}  (games.jsonl, summary.csv, paired.csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
