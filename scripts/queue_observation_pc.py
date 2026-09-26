"""Wait for the existing PC campaign, then launch its independent issue211 triplet."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_observation_screen import awake, bound, read, write  # noqa: E402


def previous_complete(root):
    chain = root / "recovery-chain.json"
    if not chain.exists() or read(chain).get("status") != "completed":
        return False
    for name in ("training", "evaluation", "latency"):
        path = root / f"{name}-state.json"
        if not path.exists() or read(path).get("status") != "completed":
            return False
    return (root / "task4-competition-evidence.tar.gz.manifest.json").is_file()


def previous_processes(root):
    me = psutil.Process()
    excluded = {me.pid, *(p.pid for p in me.parents())}
    needle = str(root.resolve()).replace("\\", "/").lower()
    found = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if process.pid in excluded or "python" not in (process.info["name"] or "").lower():
                continue
            command = " ".join(process.info["cmdline"] or []).replace("\\", "/").lower()
            if needle in command:
                found.append(process.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--wait-for", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--authorize-compute", action="store_true")
    parser.add_argument("--detached", action="store_true")
    args = parser.parse_args()
    args.root, args.wait_for = args.root.resolve(), args.wait_for.resolve()
    if not args.authorize_compute:
        raise ValueError("Explicit authorization is required")
    _cfg, binding = bound(args.root)
    if binding["replica"] != 3:
        raise ValueError("This queue is only for the PC's replica 3")
    record_path = args.root / "queue.json"
    if args.detached:
        if record_path.exists():
            record = read(record_path)
            if psutil.pid_exists(record["pid"]):
                raise ValueError("Inspect existing queue before relaunching")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--root",
            str(args.root),
            "--wait-for",
            str(args.wait_for),
            "--workers",
            str(args.workers),
            "--authorize-compute",
        ]
        flags = (
            subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
            if os.name == "nt"
            else 0
        )
        with (args.root / "queue.log").open("a") as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                creationflags=flags,
                close_fds=True,
                start_new_session=os.name != "nt",
            )
        write(
            record_path,
            {"pid": process.pid, "status": "waiting", "previous_root": str(args.wait_for)},
        )
        print(f"Queued replica 3: PID {process.pid}")
        return
    with awake():
        while datetime.now().astimezone() < datetime.fromisoformat("2026-09-20T04:00:00+02:00"):
            active = previous_processes(args.wait_for)
            ready = previous_complete(args.wait_for)
            write(
                record_path,
                {
                    "pid": os.getpid(),
                    "status": "waiting",
                    "previous_complete": ready,
                    "previous_pids": active,
                    "updated_at": datetime.now().astimezone().isoformat(),
                },
            )
            if ready and not active:
                command = [
                    sys.executable,
                    str(Path(__file__).with_name("run_observation_screen.py")),
                    "launch",
                    "--root",
                    str(args.root),
                    "--workers",
                    str(args.workers),
                    "--authorize-compute",
                ]
                subprocess.run(command, check=True)
                write(
                    record_path,
                    {"pid": os.getpid(), "status": "launched", "previous_root": str(args.wait_for)},
                )
                return
            time.sleep(30)
    write(
        record_path,
        {"pid": os.getpid(), "status": "not_launched", "reason": "04:00 latest-start deadline"},
    )


if __name__ == "__main__":
    main()
