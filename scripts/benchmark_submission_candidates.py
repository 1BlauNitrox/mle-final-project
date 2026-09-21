"""Run a resumable, common-seed benchmark for final submission candidates."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import fmean

WORKTREE = Path(__file__).resolve().parents[1]
CONFIG = WORKTREE / "experiments/2026-09-21-submission-selection-benchmark/config.json"
STAGED_CHECKPOINT = "benchmark-checkpoint.pt"
METRICS = ("score", "kills", "self_kills", "survived", "coins", "collection_fraction", "invalid")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (WORKTREE / value).resolve()


def candidate_paths(candidate: dict) -> tuple[Path, Path, Path]:
    root = resolve(candidate["root"])
    checkpoint = resolve(candidate["checkpoint"])
    agent_dir = root / "agent_code" / candidate["agent"]
    if not root.is_dir() or not checkpoint.is_file() or not agent_dir.is_dir():
        raise ValueError(f"Invalid candidate paths for {candidate['agent']}")
    return root, checkpoint, agent_dir


def round_record(stats: dict, agent: str) -> dict:
    rounds = list(stats["by_round"].values())
    if len(rounds) != 1:
        raise ValueError(f"Expected one round, got {len(rounds)}")
    agents = rounds[0]["agents"]
    name = next((key for key in agents if key.startswith(agent)), None)
    if name is None:
        raise ValueError(f"{agent} absent from stats: {list(agents)}")
    row = agents[name]
    available = row.get("initially_available_coins") or 0
    return {
        "score": row["score"],
        "kills": row["kills"],
        "self_kills": row["self_kills"],
        "survived": int(bool(row["survived"])),
        "coins": row["coins"],
        "collection_fraction": row["coins"] / available if available else None,
        "invalid": row["invalid"],
    }


@contextlib.contextmanager
def stage(candidate: dict):
    root, checkpoint, agent_dir = candidate_paths(candidate)
    staged = agent_dir / STAGED_CHECKPOINT
    if staged.exists():
        raise ValueError(f"Refusing to overwrite existing staged checkpoint: {staged}")
    shutil.copy2(checkpoint, staged)
    try:
        yield root, staged.name
    finally:
        staged.unlink(missing_ok=True)


def play(root: Path, candidate: dict, staged: str, suite: dict, seed: int) -> dict:
    handle, stats_name = tempfile.mkstemp(suffix=".json")
    os.close(handle)
    try:
        environment = {
            **os.environ,
            "BOMBERMAN_EVALUATION_CHECKPOINT": staged,
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": "",
            **candidate["environment"],
        }
        command = [
            sys.executable,
            "main.py",
            "play",
            "--agents",
            candidate["agent"],
            *suite["opponents"],
            "--scenario",
            suite["scenario"],
            "--n-rounds",
            "1",
            "--seed",
            str(seed),
            "--no-gui",
            "--save-stats",
            stats_name,
        ]
        completed = subprocess.run(command, cwd=root, env=environment, capture_output=True, text=True)
        if completed.returncode:
            message = (completed.stdout + completed.stderr)[-1000:]
            raise RuntimeError(f"Seed {seed} failed for {candidate['agent']}: {message}")
        return round_record(read(Path(stats_name)), candidate["agent"])
    finally:
        Path(stats_name).unlink(missing_ok=True)


def load_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def aggregate(rows: list[dict]) -> dict:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["candidate"], row["suite"]].append(row)
    return {
        f"{candidate}.{suite}": {
            "games": len(values),
            **{
                metric: fmean([row[metric] for row in values if row[metric] is not None])
                for metric in METRICS
                if any(row[metric] is not None for row in values)
            },
        }
        for (candidate, suite), values in sorted(grouped.items())
    }


def paired(rows: list[dict], candidate: str, suite: str, metric: str) -> float | None:
    values: dict[int, dict[str, float | None]] = defaultdict(dict)
    for row in rows:
        if row["suite"] == suite:
            values[row["seed"]][row["candidate"]] = row[metric]
    differences = [
        entry[candidate] - entry["fallback"]
        for entry in values.values()
        if entry.get(candidate) is not None and entry.get("fallback") is not None
    ]
    return fmean(differences) if differences else None


def report(rows: list[dict], cfg: dict) -> dict:
    effects = {
        candidate: {
            suite: {metric: paired(rows, candidate, suite, metric) for metric in METRICS}
            for suite in cfg["suites"]
        }
        for candidate in cfg["candidates"]
        if candidate != "fallback"
    }
    warm = effects.get("warm_lineup", {})
    overall = {
        metric: fmean(
            value
            for suite in warm.values()
            if (value := suite.get(metric)) is not None
        )
        for metric in ("score", "kills", "self_kills", "survived", "invalid")
    }
    warm_screen = {
        "score": overall["score"] >= 0,
        "kills": overall["kills"] >= 0,
        "survival": overall["survived"] >= 0,
        "self_kills": overall["self_kills"] <= 0,
        "invalid": overall["invalid"] <= 0,
        "coins_collection": warm["coins"]["collection_fraction"] is not None
        and warm["coins"]["collection_fraction"] >= 0.05,
    }
    return {
        "aggregates": aggregate(rows),
        "paired_effects_vs_fallback": effects,
        "warm_lineup_screen": warm_screen,
        "automatic_recommendation": (
            "warm_lineup" if all(warm_screen.values()) else "fallback"
        ),
        "targeted_unqualified": cfg["candidates"]["targeted_unqualified"]["ineligibility_reason"],
        "selection_rule": cfg["selection_rule"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument(
        "--out",
        type=Path,
        default=WORKTREE / "training_outputs/issue225-submission-selection-benchmark",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = read(args.config.resolve())
    for candidate in cfg["candidates"].values():
        root, _, agent_dir = candidate_paths(candidate)
        if candidate["environment"].get("BOMBERMAN_PURSUIT_HEAD"):
            head = agent_dir / candidate["environment"]["BOMBERMAN_PURSUIT_HEAD"]
            if not head.is_file():
                raise ValueError(f"Missing pursuit head: {head}")
        print(f"ready: {candidate['agent']} from {root}")
    if args.dry_run:
        return

    args.out.mkdir(parents=True, exist_ok=True)
    rows_path = args.out / "games.jsonl"
    rows = load_rows(rows_path)
    recorded = {(row["candidate"], row["suite"], row["seed"]) for row in rows}
    for name, candidate in cfg["candidates"].items():
        with stage(candidate) as (root, staged):
            for suite_name, suite in cfg["suites"].items():
                for seed in range(suite["seeds"][0], suite["seeds"][1] + 1):
                    key = name, suite_name, seed
                    if key in recorded:
                        continue
                    row = {
                        "candidate": name,
                        "suite": suite_name,
                        "seed": seed,
                        **play(root, candidate, staged, suite, seed),
                    }
                    append(rows_path, row)
                    rows.append(row)
                    recorded.add(key)
    (args.out / "analysis.json").write_text(json.dumps(report(rows, cfg), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
