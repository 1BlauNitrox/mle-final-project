"""Audit an interrupted final-training run and resume it.

The supervisor refuses to resume while `training-state.json` still lists owned
workers, because a supervisor that died without reaping them left its CPU and
wall accounting short of what was actually spent. The refusal asks for an audit,
not for the state to be deleted.

This performs that audit explicitly: it proves every recorded worker is gone,
writes what was lost to a timestamped record beside the run, and only then
clears the roster so the stage can resume. The accounting it carries forward is
a floor, and the record says so, because the unreaped workers' final samples
cannot be recovered.

Refuses to touch anything while a worker is still alive.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE_OF_MODE = {"train": "training", "evaluate": "evaluation", "latency": "latency"}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def live_workers(state: dict) -> list[dict]:
    """Recorded workers whose process really is still the one that was started."""
    alive = []
    for active in state.get("active") or []:
        try:
            process = psutil.Process(active["pid"])
            if process.create_time() == active["created"]:
                alive.append(active)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return alive


def chain_supervisor_alive(root: Path) -> bool:
    chain_state = read(root / "chain-state.json")
    if not chain_state or chain_state.get("status") != "running":
        return False
    pid = chain_state.get("active_pid")
    if not pid:
        return False
    try:
        return psutil.Process(pid).is_running()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", default="train", choices=sorted(STAGE_OF_MODE))
    parser.add_argument("--profile", default="final-training")
    parser.add_argument(
        "--dry-run", action="store_true", help="Audit and report without resuming."
    )
    args = parser.parse_args()

    root = args.root.resolve()
    stage = STAGE_OF_MODE[args.mode]
    state_path = root / f"{stage}-state.json"
    state = read(state_path)

    if state is None:
        print(f"no {stage}-state.json under {root}: nothing to resume")
        return 1
    if state.get("status") == "completed":
        print(f"{stage} already completed; nothing to resume")
        return 0

    alive = live_workers(state)
    if alive:
        print(f"REFUSING: {len(alive)} recorded worker(s) still alive: {[a['pid'] for a in alive]}")
        return 1
    if chain_supervisor_alive(root):
        print("REFUSING: the chain supervisor is still running; it owns this stage")
        return 1

    stamp = time.strftime("%Y%m%dT%H%M%S")
    roster = state.get("active") or []
    audit = {
        "performed_unix": time.time(),
        "performed_readable": time.strftime("%Y-%m-%d %H:%M:%S"),
        "root": str(root),
        "stage": stage,
        "reason": "supervisor exited without reaping its workers; resume requires an audit",
        "unreaped_workers": roster,
        "accounting_carried_forward": {
            "cpu_seconds": state.get("cpu_seconds"),
            "wall_seconds": state.get("wall_seconds"),
            "peak_memory_bytes": state.get("peak_memory_bytes"),
        },
        "accounting_caveat": (
            "The unreaped workers' final CPU and wall samples were never folded in, "
            "so the carried figures are a floor and the true spend is higher. The "
            "registered ceilings are therefore enforced against an undercount for the "
            "remainder of this stage."
        ),
        "completed_jobs_before_resume": len(state.get("completed", {})),
        "attempts_before_resume": len(state.get("attempts", [])),
    }
    write(root / f"resume-audit-{stamp}.json", audit)
    print(f"audit written: resume-audit-{stamp}.json")
    print(f"  unreaped workers      : {len(roster)}")
    print(f"  completed jobs        : {audit['completed_jobs_before_resume']}")
    print(f"  cpu_seconds (floor)   : {state.get('cpu_seconds')}")
    print(f"  wall_seconds (floor)  : {state.get('wall_seconds')}")

    if args.dry_run:
        print("dry run: state not modified, stage not resumed")
        return 0

    state["active"] = []
    state["status"] = "running"
    write(state_path, state)
    print("roster cleared; resuming stage")

    env = {
        **os.environ,
        "TASK4_PROFILE": args.profile,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    command = [
        sys.executable,
        "-m",
        "scripts.pilot_task4_competition",
        args.mode,
        "--root",
        str(root),
        "--resume",
    ]
    print("running:", " ".join(command))
    return subprocess.run(command, cwd=REPO_ROOT, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
