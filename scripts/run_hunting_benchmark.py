"""Prepare and run the prospectively registered Issue #205 benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments/2026-09-19-hunting-agent-screen/config.json"
GIB = 1024**3
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_inputs(config: dict, handoff: Path) -> dict:
    archive_name = "laptop-hunting-handoff-20260919.zip"
    archives = [parent / archive_name for parent in handoff.parents]
    archive = next((candidate for candidate in archives if candidate.is_file()), archives[0])
    reference = handoff / "reference.pt"
    external_zip = handoff / "RUEHL_BASED_AGENT.zip"
    external = handoff / "external-inspection/RUEHL_BASED_AGENT"
    expected = config["source"]
    for path in (
        archive,
        reference,
        external_zip,
        external / "callbacks.py",
        external / "Networks.py",
        external / "z_best-model.pt",
    ):
        require(path.is_file(), f"Missing required input: {path}")
    require(sha256(archive) == expected["handoff_archive_sha256"], "Handoff hash mismatch")
    require(sha256(reference) == expected["fallback_checkpoint_sha256"], "Fallback hash mismatch")
    require(
        sha256(external_zip) == expected["external_archive_sha256"],
        "External archive hash mismatch",
    )
    return {"archive": archive, "reference": reference, "external": external}


def prepare_runtime(config: dict, handoff: Path, runtime: Path) -> dict:
    inputs = verify_inputs(config, handoff)
    if runtime.exists():
        require(
            (runtime / "runtime-manifest.json").is_file(), "Refusing unmanifested runtime directory"
        )
        manifest = load_json(runtime / "runtime-manifest.json")
        require(manifest["source"] == config["source"], "Existing runtime has different inputs")
        return manifest

    runtime.mkdir(parents=True)
    for name in FRAMEWORK_FILES:
        shutil.copy2(ROOT / name, runtime / name)
    shutil.copytree(ROOT / "assets", runtime / "assets")
    (runtime / "agent_code").mkdir()
    shutil.copytree(ROOT / "agent_code/rule_based_agent", runtime / "agent_code/rule_based_agent")
    shutil.copytree(ROOT / "agent_code/Bomb-omb", runtime / "agent_code/benchmark_fallback")
    default_fixture = runtime / "agent_code/benchmark_fallback/checkpoint.pt"
    if default_fixture.exists():
        default_fixture.unlink()
    shutil.copy2(inputs["reference"], runtime / "agent_code/benchmark_fallback/reference.pt")
    shutil.copytree(inputs["external"], runtime / "agent_code/RUEHL_BASED_AGENT")

    manifest = {
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "source": config["source"],
        "fallback_runtime_files": {
            p.relative_to(runtime / "agent_code/benchmark_fallback").as_posix(): sha256(p)
            for p in sorted((runtime / "agent_code/benchmark_fallback").rglob("*"))
            if p.is_file()
        },
        "external_files": {
            p.relative_to(runtime / "agent_code/RUEHL_BASED_AGENT").as_posix(): sha256(p)
            for p in sorted((runtime / "agent_code/RUEHL_BASED_AGENT").rglob("*"))
            if p.is_file()
        },
    }
    (runtime / "runtime-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def base_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "BOMBERMAN_EVALUATION_CHECKPOINT": "reference.pt",
            "BOMBERMAN_AGENT_SEED": "2052026",
            "BOMBERMAN_DQN_ACTION_MASKING": "framework_legal",
            "BOMBERMAN_DQN_ESCAPE_CONTINUATIONS": "on",
        }
    )
    return env


def command(
    python: Path, runtime: Path, lineup: list[str], seed: int, stats: Path, log_dir: Path
) -> list[str]:
    return [
        str(python),
        str(runtime / "main.py"),
        "play",
        "--no-gui",
        "--n-rounds",
        "1",
        "--scenario",
        "classic",
        "--seed",
        str(seed),
        "--agents",
        *lineup,
        "--save-stats",
        str(stats),
        "--log-dir",
        str(log_dir),
    ]


def rotated(values: list[str], index: int) -> list[str]:
    return values[index:] + values[:index]


def validate_stats(path: Path, lineup: list[str]) -> dict:
    data = load_json(path)
    rounds = list(data.get("by_round", {}).values())
    require(len(rounds) == 1, f"Expected one round in {path}")
    agents = rounds[0].get("agents", {})
    require(len(agents) == 4, f"Expected four agents in {path}")
    expected = {name for name in lineup if name != "rule_based_agent"}
    require(
        expected.issubset(agents), f"Learned agent missing from {path}: {expected - set(agents)}"
    )
    return data


def run_game(
    *,
    python: Path,
    runtime: Path,
    output: Path,
    lineup_name: str,
    lineup: list[str],
    seed: int,
    rotation: int,
    timing: bool = False,
) -> dict:
    tag = f"{lineup_name}-s{seed}-r{rotation}"
    subdir = "timing" if timing else "games"
    stats = output / subdir / f"{tag}.json"
    stdout = output / "logs" / f"{tag}.txt"
    stats.parent.mkdir(parents=True, exist_ok=True)
    stdout.parent.mkdir(parents=True, exist_ok=True)
    (output / "framework-logs").mkdir(parents=True, exist_ok=True)
    ordered = rotated(lineup, rotation)
    if stats.exists():
        validate_stats(stats, ordered)
        return {"tag": tag, "resumed": True, "seconds": None}
    require(psutil.virtual_memory().available >= 3 * GIB, "Available RAM fell below 3 GiB")
    cmd = command(python, runtime, ordered, seed, stats, output / "framework-logs")
    started = time.monotonic()
    with stdout.open("w", encoding="utf-8") as stream:
        process = subprocess.run(
            cmd,
            cwd=runtime,
            env=base_env(),
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=1800,
        )
    elapsed = time.monotonic() - started
    require(process.returncode == 0, f"Game failed ({process.returncode}); see {stdout}")
    validate_stats(stats, ordered)
    return {"tag": tag, "resumed": False, "seconds": elapsed, "command": cmd}


def verify_smoke(runtime: Path, output: Path) -> None:
    fallback_log = runtime / "agent_code/benchmark_fallback/logs/benchmark_fallback.log"
    external_log = runtime / "agent_code/RUEHL_BASED_AGENT/logs/RUEHL_BASED_AGENT.log"
    require(
        "Loaded frozen DQN policy after 8000 training episodes"
        in fallback_log.read_text(encoding="utf-8"),
        "Fallback did not log a loaded 8000-episode policy",
    )
    require(
        "Loaded model from" in external_log.read_text(encoding="utf-8"),
        "External agent did not log successful model loading",
    )
    report = {
        "verified_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "fallback_log": str(fallback_log),
        "external_log": str(external_log),
    }
    (output / "smoke-verification.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "smoke", "scientific", "timing"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    args = parser.parse_args()
    config = load_json(args.config)
    require(config["status"] == "prospective", "Protocol is not prospective")
    require(sys.version_info[:2] == (3, 13), "Harness itself must run under Python 3.13")
    require(
        args.python.resolve() == Path(sys.executable).resolve(),
        "Child and harness Python must match",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    prepare_runtime(config, args.handoff, args.runtime)
    if args.phase == "prepare":
        print(f"Prepared immutable evaluation runtime: {args.runtime}")
        return

    lineups = config["design"]["lineups"]
    if args.phase == "smoke":
        result = run_game(
            python=args.python,
            runtime=args.runtime,
            output=args.output,
            lineup_name="smoke",
            lineup=[
                "benchmark_fallback",
                "RUEHL_BASED_AGENT",
                "rule_based_agent",
                "rule_based_agent",
            ],
            seed=config["design"]["smoke_seed"],
            rotation=0,
        )
        verify_smoke(args.runtime, args.output)
        print(json.dumps(result, indent=2))
        return

    require((args.output / "smoke-verification.json").is_file(), "Smoke verification is required")
    started = datetime.now(timezone.utc).astimezone()
    results = []
    if args.phase == "timing":
        for name in ("A", "B", "C"):
            results.append(
                run_game(
                    python=args.python,
                    runtime=args.runtime,
                    output=args.output,
                    lineup_name=name,
                    lineup=lineups[name],
                    seed=954099998,
                    rotation=0,
                    timing=True,
                )
            )
    else:
        tasks = iter(
            (name, seed, rotation)
            for seed in config["design"]["world_seeds"]
            for rotation in range(config["design"]["rotations"])
            for name in ("A", "B", "C")
        )
        total = config["design"]["total_games"]
        deadline = time.monotonic() + config["runtime"]["wall_clock_limit_hours"] * 3600
        with ThreadPoolExecutor(max_workers=2) as pool:
            active = set()

            def submit_next() -> bool:
                try:
                    name, seed, rotation = next(tasks)
                except StopIteration:
                    return False
                active.add(
                    pool.submit(
                        run_game,
                        python=args.python,
                        runtime=args.runtime,
                        output=args.output,
                        lineup_name=name,
                        lineup=lineups[name],
                        seed=seed,
                        rotation=rotation,
                    )
                )
                return True

            submit_next()
            submit_next()
            while active:
                completed, active = wait(active, return_when=FIRST_COMPLETED)
                for future in completed:
                    results.append(future.result())
                    print(f"completed {len(results)}/{total}: {results[-1]['tag']}", flush=True)
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Registered three-hour wall-clock limit reached")
                    submit_next()
    run = {
        "phase": args.phase,
        "started_at": started.isoformat(),
        "ended_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "python": sys.version,
        "runtime": str(args.runtime),
        "output": str(args.output),
        "games": results,
        "cut_short": False,
    }
    (args.output / f"run-{args.phase}.json").write_text(
        json.dumps(run, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
