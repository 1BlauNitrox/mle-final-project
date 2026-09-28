"""Issue #234: play the head-to-head benchmark between the frozen tabular agent and Bomb-omb.

Both agents are exported unchanged from their recorded commits and verified by SHA-256.
Games run as separate `main.py play` processes with tournament defaults (no agent
environment overrides). Per-game rows are appended to the experiment's per-game.jsonl.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/2026-09-27-final-agent-comparison"
OUTPUT = ROOT / "training_outputs/issue234-final-agent-comparison"
FRAMEWORK_FILES = (
    "agents.py",
    "environment.py",
    "events.py",
    "fallbacks.py",
    "items.py",
    "main.py",
    "replay.py",
    "settings.py",
)
RULE_BASED = "rule_based_agent"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def export_agent(commit: str, source_path: str, target: Path) -> None:
    """Extract one agent directory from git without touching the working tree."""
    archive = subprocess.run(
        ["git", "archive", "--format=tar", commit, source_path],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(target, filter="data")


def prepare_runtime(config: dict, runtime: Path) -> dict:
    """Build the runtime once; an existing runtime is reused after re-verifying all inputs.

    Deleting a runtime is avoided on purpose: synced folders (OneDrive) can lock __pycache__.
    """
    if not runtime.exists():
        (runtime / "agent_code").mkdir(parents=True)
        for name in FRAMEWORK_FILES:
            shutil.copy2(ROOT / name, runtime / name)
        shutil.copytree(ROOT / "assets", runtime / "assets")
        shutil.copytree(ROOT / "agent_code" / RULE_BASED, runtime / "agent_code" / RULE_BASED)
        for spec in config["agents"].values():
            export_agent(spec["source_commit"], spec["source_path"], runtime)
    for name in FRAMEWORK_FILES:
        if sha256(ROOT / name) != sha256(runtime / name):
            raise SystemExit(f"Runtime framework file {name} differs from the repository")
    verified = {}
    for name, spec in config["agents"].items():
        artifact = runtime / spec["source_path"] / spec["artifact"]
        digest = sha256(artifact)
        if digest != spec["sha256"]:
            raise SystemExit(f"{name}: artifact SHA-256 {digest} does not match {spec['sha256']}")
        verified[name] = digest
    return verified


def game_plan(config: dict) -> list[dict]:
    first, count = config["world_seeds"]["first"], config["world_seeds"]["count"]
    suites = config["suites"]
    games = []
    base = suites["head_to_head"]["lineup"]
    for index in range(count):
        seed = first + index
        for rotation in suites["head_to_head"]["rotations"]:
            lineup = base[rotation:] + base[:rotation]
            games.append(
                {
                    "suite": "head_to_head",
                    "world_seed": seed,
                    "rotation": rotation,
                    "lineup": lineup,
                }
            )
        for agent in suites["versus_rule_based"]["learned_agents"]:
            lineup = list(suites["versus_rule_based"]["opponents"])
            lineup.insert(index % 4, agent)
            games.append(
                {
                    "suite": "versus_rule_based",
                    "world_seed": seed,
                    "rotation": index % 4,
                    "lineup": lineup,
                }
            )
    return games


def game_tag(game: dict) -> str:
    learned = "-".join(a for a in game["lineup"] if a != RULE_BASED)
    return f"{game['suite']}-w{game['world_seed']}-r{game['rotation']}-{learned}"


def clean_env() -> dict:
    """Tournament conditions: single thread, no agent-specific overrides."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("BOMBERMAN_")}
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    return env


def placement(scores: dict[str, float], name: str) -> tuple[int, bool]:
    better = sum(1 for other, value in scores.items() if other != name and value > scores[name])
    tied = sum(1 for other, value in scores.items() if other != name and value == scores[name])
    return better + 1, better == 0 and tied == 0


def extract_row(game: dict, stats: dict) -> dict:
    rounds = list(stats["by_round"].values())
    if len(rounds) != 1:
        raise ValueError("expected exactly one round")
    agents = rounds[0]["agents"]
    if len(agents) != 4:
        raise ValueError("expected four agents")
    scores = {name: values["score"] for name, values in agents.items()}
    rows = {}
    for name, values in agents.items():
        place, strict_first = placement(scores, name)
        rows[name] = {
            "score": values["score"],
            "coins": values.get("coins", 0),
            "kills": values.get("kills", 0),
            "self_kills": values.get("self_kills", values.get("suicides", 0)),
            "survived": bool(values.get("survived", False)),
            "invalid_actions": values.get("invalid", values.get("invalid_actions", 0)),
            "attempted_actions": values.get("attempted_actions", 0),
            "decision_time_p95_ms": values.get("decision_time_p95_ms"),
            "decision_time_max_ms": values.get("decision_time_max_ms"),
            "placement": place,
            "strict_first": strict_first,
        }
    return {**game, "agents": rows}


def run_game(python: Path, runtime: Path, game: dict) -> dict:
    tag = game_tag(game)
    stats_path = OUTPUT / "games" / f"{tag}.json"
    log_dir = OUTPUT / "framework-logs" / tag
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    if not stats_path.exists():
        cmd = [
            str(python),
            str(runtime / "main.py"),
            "play",
            "--no-gui",
            "--n-rounds",
            "1",
            "--scenario",
            "classic",
            "--seed",
            str(game["world_seed"]),
            "--agents",
            *game["lineup"],
            "--save-stats",
            str(stats_path),
            "--log-dir",
            str(log_dir),
        ]
        with (log_dir / "stdout.txt").open("w", encoding="utf-8") as out:
            process = subprocess.run(
                cmd,
                cwd=runtime,
                env=clean_env(),
                stdout=out,
                stderr=subprocess.STDOUT,
                timeout=1800,
            )
        if process.returncode != 0:
            raise RuntimeError(f"{tag} failed with exit code {process.returncode}")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    return extract_row(game, stats)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--limit", type=int, default=None, help="only run the first N games")
    args = parser.parse_args()

    config = json.loads((EXPERIMENT / "config.json").read_text(encoding="utf-8"))
    runtime = OUTPUT / "runtime"
    verified = prepare_runtime(config, runtime)
    games = game_plan(config)[: args.limit]
    rows = []
    with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
        futures = [pool.submit(run_game, args.python, runtime, game) for game in games]
        for number, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if number % 50 == 0 or number == len(games):
                print(f"{number}/{len(games)} games done", flush=True)
    rows.sort(key=lambda r: (r["suite"], r["world_seed"], r["rotation"], "-".join(r["lineup"])))
    results = EXPERIMENT / "results"
    results.mkdir(parents=True, exist_ok=True)
    with (results / "per-game.jsonl").open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, sort_keys=True) + "\n")
    (results / "verified-artifacts.json").write_text(
        json.dumps({"verified_sha256": verified, "games": len(rows)}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} games to {results / 'per-game.jsonl'}")


if __name__ == "__main__":
    main()
