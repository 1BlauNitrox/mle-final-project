"""Replay the registered #162 matrix using immutable #150 code and #151 instrumentation."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/2026-09-12-task3-hunting-diagnosis/config.json"
SOURCE = "6f014485a3026cc3707fa2cc3a379880dd0b74bd"
INSTRUMENT = "e960bcc21dbe8b03343b85587a3af725c415616d"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def summarize(rows):
    """Descriptive counts; denominator is retained pre-action states, never efficacy."""
    result = dict(
        steps=len(rows),
        opponent_steps=0,
        attack_steps=0,
        safe_attack_steps=0,
        available_safe_attack_steps=0,
        bombs=0,
        safe_attack_bombs=0,
        penalized_safe_attack_bombs=0,
        crate_free_safe_attack_steps=0,
        approach_steps=0,
    )
    distances, bomb_gaps = [], []
    for row in rows:
        f = row["features"]
        if len(f) != 39:
            raise ValueError("Expected frozen schema-four features")
        state = row["state"]
        opponents = [o[3] for o in state["others"]]
        pos = state["self"][3]
        if opponents:
            result["opponent_steps"] += 1
            distance = min(abs(pos[0] - o[0]) + abs(pos[1] - o[1]) for o in opponents)
            distances.append(distance)
            after = row["next_position"]
            result["approach_steps"] += int(
                min(abs(after[0] - o[0]) + abs(after[1] - o[1]) for o in opponents) < distance
            )
        safe = bool(f[31])
        available = safe and bool(state["self"][2]) and bool(row["legal_mask"][5])
        dropped = "BOMB_DROPPED" in row["events"]
        result["attack_steps"] += int(bool(f[30]))
        result["safe_attack_steps"] += int(safe)
        result["available_safe_attack_steps"] += int(available)
        result["crate_free_safe_attack_steps"] += int(safe and f[19] == 0)
        result["bombs"] += int(dropped)
        result["safe_attack_bombs"] += int(safe and dropped)
        result["penalized_safe_attack_bombs"] += int(
            safe and dropped and row["reward_components"].get("WASTEFUL_BOMB_PLACED", 0) < 0
        )
        if available:
            legal_q = [
                q for q, allowed in zip(row["q_values"], row["legal_mask"], strict=True) if allowed
            ]
            bomb_gaps.append(max(legal_q) - row["q_values"][5])
    result["opponent_manhattan_distances"] = distances
    result["available_attack_bomb_q_gaps"] = bomb_gaps
    return result


def instrument_source(repo):
    source = subprocess.check_output(
        ["git", "show", f"{INSTRUMENT}:training/diagnose_dqn_stalls.py"], cwd=repo
    ).decode("utf-8")
    old = "world_seed in {1460001, 1460002} and agent_seed == world_seed + 1000000"
    new = "world_seed in {1501101, 1501102, 1501201} and agent_seed == world_seed + 1000000"
    if source.count(old) != 1:
        raise ValueError("Reviewed instrumentation seed contract changed")
    return source.replace(old, new)


def run(evidence, output):
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["executed_source"] != SOURCE or config["instrumentation_source"] != INSTRUMENT:
        raise ValueError("Unregistered source")
    if config["max_episodes"] != 15 or len(config["replicas"]) != 5:
        raise ValueError("Unregistered matrix")
    inputs = []
    for r in config["replicas"]:
        path = (evidence / r["archive_path"]).resolve()
        path.relative_to(evidence.resolve())
        if digest(path) != r["sha256"] or path.stat().st_size != int(r["size_bytes"]):
            raise ValueError(f"Wrong input: {r['replica']}")
        inputs.append((r, path))
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_root = output / "source"
    source_root.mkdir()
    archive = output / "source.tar"
    subprocess.run(
        ["git", "archive", "--format=tar", f"--output={archive}", SOURCE], cwd=ROOT, check=True
    )
    with tarfile.open(archive) as tar:
        tar.extractall(source_root, filter="data")
    instrument = source_root / "training/diagnose_issue162_worker.py"
    instrument.write_text(instrument_source(ROOT), encoding="utf-8")
    source_manifest = {
        str(p.relative_to(source_root)): digest(p) for p in source_root.rglob("*") if p.is_file()
    }
    write(output / "source-manifest.json", source_manifest)
    write(output / "registration.json", config)
    report = {
        "scope": config["scope"],
        "execution_source": SOURCE,
        "tool_source": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config_sha256": digest(CONFIG),
        "jobs": [],
        "cpu_seconds": 0.0,
        "peak_memory_bytes": 0,
        "status": "running",
    }
    started = time.monotonic()
    try:
        for r, checkpoint in inputs:
            for case in config["cases"]:
                name = f"{r['replica']}-{case['world_seed']}"
                target = output / name
                job = {"replica": r["replica"], **case, "status": "running"}
                report["jobs"].append(job)
                command = [
                    sys.executable,
                    "-m",
                    "training.diagnose_issue162_worker",
                    "--agent",
                    "DagobertDuckDQNTask3",
                    "--checkpoint",
                    str(checkpoint),
                    "--world-seed",
                    str(case["world_seed"]),
                    "--agent-seed",
                    str(case["world_seed"] + 1000000),
                    "--output",
                    str(target),
                ]
                if case["opponent"]:
                    command += ["--opponent", case["opponent"]]
                environment = {
                    **os.environ,
                    "BOMBERMAN_COMPACT_LOGS": "1",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                }
                job_start, cpu = time.monotonic(), 0.0
                with (output / f"{name}.log").open("w", encoding="utf-8") as log:
                    child = subprocess.Popen(
                        command,
                        cwd=source_root,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    try:
                        process = psutil.Process(child.pid)
                        while child.poll() is None:
                            try:
                                times = process.cpu_times()
                                cpu = max(cpu, times.user + times.system)
                                memory = process.memory_info().rss
                                report["peak_memory_bytes"] = max(
                                    report["peak_memory_bytes"], memory
                                )
                            except psutil.NoSuchProcess:
                                break
                            if (
                                time.monotonic() - started >= config["wall_seconds"]
                                or time.monotonic() - job_start >= config["per_episode_seconds"]
                                or report["cpu_seconds"] + cpu >= config["cpu_seconds"]
                                or memory >= config["memory_bytes"]
                            ):
                                raise RuntimeError("Registered diagnostic resource limit")
                            time.sleep(0.1)
                        if child.wait() != 0:
                            raise RuntimeError(f"Diagnostic failed; retained log: {name}")
                    finally:
                        if child.poll() is None:
                            child.kill()
                            child.wait()
                        report["cpu_seconds"] += cpu
                if digest(checkpoint) != r["sha256"]:
                    raise ValueError("Input checkpoint changed")
                rows = json.loads(gzip.decompress((target / "trajectory.json.gz").read_bytes()))
                native = json.loads((target / "summary.json").read_text(encoding="utf-8"))
                job.update(
                    status="completed",
                    checkpoint_sha256=digest(checkpoint),
                    trajectory_sha256=digest(target / "trajectory.json.gz"),
                    metrics=summarize(rows),
                    stall_windows=native["stall_windows"],
                )
                # The reviewed instrument's git field observes the enclosing tool checkout.
                # Preserve it verbatim; the frozen execution hash is explicit in this report.
                report["wall_seconds"] = time.monotonic() - started
                write(output / "summary.json", report)
                print(
                    name,
                    {k: v for k, v in job["metrics"].items() if not isinstance(v, list)},
                    flush=True,
                )
        report["status"] = "completed"
    finally:
        report["wall_seconds"] = time.monotonic() - started
        for r, checkpoint in inputs:
            if digest(checkpoint) != r["sha256"]:
                report["input_integrity_failure"] = r["replica"]
        if report["status"] != "completed":
            report["status"] = "incomplete"
        write(output / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.evidence, args.output)


if __name__ == "__main__":
    main()
