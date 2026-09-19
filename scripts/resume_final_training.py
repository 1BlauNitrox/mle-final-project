"""Report on, recover and continue the final-training run.

The chain supervisor refuses to restart once it has stopped ("no automatic
retries"), and a stage supervisor refuses to resume while its state still lists
owned workers, because a supervisor that died without reaping them left its CPU
and wall accounting short of what was really spent. Both refusals ask for a
deliberate recovery, not for state to be deleted. This tool is that recovery.

  --status       Read-only, stage-aware health verdict for a watchdog.
  (default)      Refuse while any process still works on the run; audit a stage
                 its supervisor abandoned; then hand every remaining stage to a
                 detached orchestrator and return immediately.
  --orchestrate  Internal. Run the remaining stages in order, as the chain would.

The audit's carried accounting is a floor, and its record says so, because the
unreaped workers' final samples cannot be recovered.
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
STAGES = (
    ("train", "training"),
    ("evaluate", "evaluation"),
    ("latency", "latency"),
    ("results", None),
)
EVIDENCE = "task4-competition-evidence.tar.gz"
RECOVERY_STATE = "recovery-chain.json"
STALE_MINUTES = {"training": 12, "evaluation": 30, "latency": 30}
MEMORY_WAIT_SECONDS = 45 * 60


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def durable_write(path: Path, value) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def normalized(text: str) -> str:
    return text.lower().replace("/", "\\").rstrip("\\")


def stage_complete(root: Path, mode: str, stage: str | None) -> bool:
    if stage is not None:
        state = read(root / f"{stage}-state.json")
        return bool(state) and state.get("status") == "completed"
    manifest = read(root / f"{EVIDENCE}.manifest.json")
    archive = root / EVIDENCE
    return (
        bool(manifest)
        and archive.is_file()
        and archive.stat().st_size == manifest.get("size_bytes")
    )


def first_incomplete(root: Path):
    for mode, stage in STAGES:
        if not stage_complete(root, mode, stage):
            return mode, stage
    return None


def processes_on_run(root: Path) -> list[dict]:
    """Every live process, other than this one and its ancestors, that references the run."""
    me = psutil.Process()
    excluded = {me.pid, *(parent.pid for parent in me.parents())}
    target = normalized(str(root))
    found = []
    for process in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        if process.info["pid"] in excluded:
            continue
        try:
            command = " ".join(process.info["cmdline"] or [])
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        if target in normalized(command):
            found.append(
                {
                    "pid": process.info["pid"],
                    "name": process.info["name"],
                    "cmdline": command[-160:],
                }
            )
    return found


def live_orchestrator(root: Path):
    chain = read(root / "chain-state.json")
    if (
        chain
        and chain.get("status") == "running"
        and chain.get("active_pid")
        and psutil.pid_exists(chain["active_pid"])
    ):
        return "chain, stage supervisor", chain["active_pid"]
    recovery = read(root / RECOVERY_STATE)
    if recovery and recovery.get("status") == "running":
        try:
            if psutil.Process(recovery["pid"]).create_time() == recovery["created"]:
                return "recovery", recovery["pid"]
        except psutil.NoSuchProcess:
            pass
    return None


def job_finished(root: Path, job: str) -> bool:
    return any((root / "training" / job).glob("attempt-*/result.json"))


def progress_age_minutes(root: Path, stage: str | None):
    now = time.time()
    if stage == "training":
        ages = [
            (now - (job / "episodes.json.gz").stat().st_mtime) / 60
            for job in (root / "training-resume").glob("*")
            if (job / "episodes.json.gz").is_file() and not job_finished(root, job.name)
        ]
        return max(ages) if ages else None
    if stage in ("evaluation", "latency"):
        files = (
            [p for p in (root / stage).rglob("*") if p.is_file()] if (root / stage).is_dir() else []
        )
        newest = max((p.stat().st_mtime for p in files), default=None)
        if newest is None and (root / f"{stage}-state.json").is_file():
            newest = (root / f"{stage}-state.json").stat().st_mtime
        return None if newest is None else (now - newest) / 60
    return None


def status(root: Path) -> tuple[str, dict]:
    report = {"root": str(root)}
    pending = first_incomplete(root)
    report["stage"] = pending[0] if pending else "all complete"
    report["completed_stages"] = [
        mode for mode, stage in STAGES if stage_complete(root, mode, stage)
    ]
    owner = live_orchestrator(root)
    report["orchestrator"] = f"{owner[0]} pid {owner[1]}" if owner else "none alive"

    if (root / "training-resume").is_dir():
        report["milestones"] = len(list((root / "training-resume").glob("*/milestone-*.pt")))
    played_holdout = (root / "evaluation").is_dir() and any(
        "holdout" in p.name for p in (root / "evaluation").iterdir()
    )
    if played_holdout:
        report["warning"] = (
            "a held-out suite appears under evaluation/ - it must never be played there"
        )
    report["audits"] = len(list(root.glob("resume-audit-*.json")))
    report["ram_available_gib"] = round(psutil.virtual_memory().available / 1024**3, 2)

    if pending is None:
        return "COMPLETED", report
    if owner is None:
        return "NEEDS_RECOVERY", report
    age = progress_age_minutes(root, pending[1])
    report["progress_age_minutes"] = None if age is None else round(age, 1)
    limit = STALE_MINUTES.get(pending[1] or "")
    if age is not None and limit is not None and age > limit:
        return "STALLED", report
    return "HEALTHY", report


def roster_alive(state: dict) -> list[int]:
    alive = []
    for active in state.get("active") or []:
        try:
            if psutil.Process(active["pid"]).create_time() == active["created"]:
                alive.append(active["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return alive


def audit(root: Path, stage: str) -> Path | None:
    state_path = root / f"{stage}-state.json"
    state = read(state_path)
    if not state or not state.get("active"):
        return None
    if roster_alive(state):
        raise RuntimeError(f"recorded {stage} workers are still alive: {roster_alive(state)}")
    stamp = time.strftime("%Y%m%dT%H%M%S")
    record = {
        "performed_unix": time.time(),
        "performed_readable": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": stage,
        "reason": "supervisor exited without reaping its workers; resume requires an audit",
        "verified_gone": "no live process references the run root",
        "unreaped_workers": state["active"],
        "accounting_carried_forward": {
            "cpu_seconds": state.get("cpu_seconds"),
            "wall_seconds": state.get("wall_seconds"),
            "peak_memory_bytes": state.get("peak_memory_bytes"),
        },
        "accounting_caveat": (
            "The unreaped workers' final CPU and wall samples were never folded in, "
            "so the carried figures are a floor and the true spend is higher. The "
            "registered ceilings are enforced against an undercount for the rest of "
            "this stage."
        ),
        "completed_jobs_before_resume": len(state.get("completed", {})),
        "attempts_before_resume": len(state.get("attempts", [])),
    }
    path = root / f"resume-audit-{stamp}.json"
    durable_write(path, record)
    state["active"] = []
    state["status"] = "running"
    durable_write(state_path, state)
    return path


def stage_environment(profile: str) -> dict:
    return {
        **os.environ,
        "TASK4_PROFILE": profile,
        "TASK4_AUTHORIZED": "yes",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }


def launch_orchestrator(root: Path, profile: str) -> int:
    stamp = time.strftime("%Y%m%dT%H%M%S")
    log = (root / f"recovery-{stamp}.log").open("w", encoding="utf-8")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--root",
        str(root),
        "--profile",
        profile,
        "--orchestrate",
    ]
    flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    common = dict(
        cwd=REPO_ROOT,
        env=stage_environment(profile),
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        close_fds=True,
    )
    try:
        process = subprocess.Popen(
            command, creationflags=flags | subprocess.CREATE_BREAKAWAY_FROM_JOB, **common
        )
    except OSError:
        process = subprocess.Popen(command, creationflags=flags, **common)
    return process.pid


def wait_for_memory(root: Path, record: dict) -> None:
    minimum = read(root / "config.json")["minimum_free_memory_bytes"]
    started = time.monotonic()
    while (
        psutil.virtual_memory().available < minimum
        and time.monotonic() - started < MEMORY_WAIT_SECONDS
    ):
        time.sleep(30)
    record.setdefault("memory_waits_seconds", []).append(round(time.monotonic() - started))


def set_aside_partial_archive(root: Path) -> None:
    archive = root / EVIDENCE
    if archive.exists() and not stage_complete(root, "results", None):
        archive.rename(root / f"{EVIDENCE}.partial-{time.strftime('%Y%m%dT%H%M%S')}")


def orchestrate(root: Path, profile: str, runner=subprocess.run) -> int:
    me = psutil.Process()
    path = root / RECOVERY_STATE
    record = {
        "status": "running",
        "pid": me.pid,
        "created": me.create_time(),
        "started_readable": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": None,
        "completed": [],
    }
    durable_write(path, record)
    try:
        for mode, stage in STAGES:
            if stage_complete(root, mode, stage):
                record["completed"].append(mode)
                continue
            record["stage"] = mode
            durable_write(path, record)
            wait_for_memory(root, record)
            command = [
                sys.executable,
                "-m",
                "scripts.pilot_task4_competition",
                mode,
                "--root",
                str(root),
            ]
            if stage is not None and (root / f"{stage}-state.json").is_file():
                command.append("--resume")
            if mode == "results":
                set_aside_partial_archive(root)
                command += ["--output", str(root / EVIDENCE)]
            returncode = runner(command, cwd=REPO_ROOT, env=stage_environment(profile)).returncode
            if returncode != 0 or not stage_complete(root, mode, stage):
                record.update(status="failed", error=f"{mode} exited {returncode}")
                durable_write(path, record)
                return 1
            record["completed"].append(mode)
            durable_write(path, record)
        record.update(status="completed", stage=None)
        durable_write(path, record)
        return 0
    except BaseException as error:
        record.update(status="failed", error=repr(error))
        durable_write(path, record)
        raise


def recover(root: Path, profile: str) -> int:
    busy = processes_on_run(root)
    if busy:
        print(f"REFUSING: {len(busy)} live process(es) still reference {root}:")
        for process in busy[:12]:
            print(f"  pid {process['pid']} {process['name']}: ...{process['cmdline']}")
        return 1
    pending = first_incomplete(root)
    if pending is None:
        print("every stage is already complete; nothing to recover")
        return 0
    mode, stage = pending
    audited = audit(root, stage) if stage else None
    if audited:
        print(f"audit written: {audited.name}")
    pid = launch_orchestrator(root, profile)
    print(
        f"resuming from stage '{mode}'; detached orchestrator pid {pid}; state in {RECOVERY_STATE}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, required=True)
    # The tool sets TASK4_PROFILE for every stage it launches, so this default
    # silently overrides the caller's own environment. Honouring the variable
    # first means `TASK4_PROFILE=warm-lineup resume_final_training.py ...` does
    # what it looks like it does: on 19 September it did not, and the resumed
    # stage died on "Prepared protocol copy changed" because it verified the
    # warm-lineup run against the final-training registration.
    parser.add_argument("--profile", default=os.environ.get("TASK4_PROFILE", "final-training"))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--status", action="store_true")
    group.add_argument("--orchestrate", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()

    if args.status:
        verdict, report = status(root)
        for key, value in report.items():
            print(f"{key:<22}: {value}")
        print(f"VERDICT: {verdict}")
        return {"HEALTHY": 0, "COMPLETED": 0, "NEEDS_RECOVERY": 2, "STALLED": 3}[verdict]
    if args.orchestrate:
        return orchestrate(root, args.profile)
    return recover(root, args.profile)


if __name__ == "__main__":
    raise SystemExit(main())
