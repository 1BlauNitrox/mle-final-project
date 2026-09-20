"""Attach to issue 217; bounded recovery without changing scientific inputs.

Run beside the pinned bundle, not by editing its runner. Unknown failures remain
stopped. This process must stay alive; it does not install a Windows startup task.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import psutil


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    temporary = path.with_suffix(".watch-tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    for attempt in range(10):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.2)


def process(record):
    """PID alone is never proof of ownership, including after a reboot."""
    try:
        owner = psutil.Process(record["pid"])
        if abs(owner.create_time() - record["created"]) < 0.01 and owner.is_running():
            return owner
    except (psutil.NoSuchProcess, KeyError):
        pass
    return None


@contextlib.contextmanager
def exclusive(root):
    with (root / "watchdog.lock").open("a+b") as stream:
        stream.seek(0)
        stream.write(b"0")
        stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def validate(root, runner):
    binding = read(root / "binding.json")
    expected = [(root / "config.json", binding["config_sha256"])]
    expected += [(runner.parent.parent / p, h) for p, h in binding["tools"].items()]
    expected += [(root / p, h) for p, h in binding["inputs"].items()]
    for path, checksum in expected:
        if digest(path) != checksum:
            raise ValueError(f"Integrity failure: {path}")
    for pointer in root.glob("pairs/*/*/resume.json"):
        record = read(pointer)
        generation = (pointer.parent / record["generation"]).resolve()
        if not generation.is_relative_to(pointer.parent.resolve()):
            raise ValueError("Unsafe generation path")
        for name, key in (("checkpoint.pt", "checkpoint_sha256"), ("state.json", "state_sha256")):
            if digest(generation / name) != record[key]:
                raise ValueError(f"Corrupt recovery generation: {generation}")
    if not read(root / "smoke.json")["passed"]:
        raise ValueError("Passing mechanics check required")


def recovery_reason(root):
    """Allow only unexpected process loss or narrowly identified sharing errors."""
    if (root / "STOP.request").exists():
        return None
    if not (root / "STOP.json").exists():
        return "Unexpected process exit; recover last verified generation"
    reason = read(root / "STOP.json")["reason"]
    if reason.startswith("Worker failed: "):
        key = reason.removeprefix("Worker failed: ").split(";", 1)[0]
        if Path(key).name != key:
            return None
        log = root / f"{key}.log"
        if not log.exists():
            return None
        reason = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-1]
    if ("[WinError 32]" in reason or "[WinError 33]" in reason) and (
        "PermissionError" in reason or "OSError" in reason or reason.startswith("[WinError")
    ):
        return "Transient Windows file sharing error"
    return None


def budget_reason(root, overhead=0):
    cfg = read(root / "config.json")
    device = cfg["devices"][read(root / "binding.json")["device"]]
    usage = read(root / "resources.json") if (root / "resources.json").exists() else None
    now = time.time()
    training_stop = cfg["limits"].get("training_stop_utc")
    if (training_stop and now >= datetime.fromisoformat(training_stop).timestamp()
            and any(read(p)["stage"] == "training" for p in root.glob("pairs/*/work-stage.json"))):
        return "Training absolute deadline reached"
    if now >= datetime.fromisoformat(cfg["limits"]["stop_utc"]).timestamp():
        return "Absolute deadline reached"
    if usage:
        if usage["cpu_seconds"] + overhead >= device["cpu_seconds"]:
            return "CPU budget exhausted (including watcher overhead)"
        if now - usage["first_start"] >= cfg["limits"]["wall_seconds"]:
            return "Elapsed budget exhausted"
    return None


def observe_family(owner):
    records = []
    try:
        family = [owner, *owner.children(recursive=True)]
    except psutil.NoSuchProcess:
        return records
    for member in family:
        with contextlib.suppress(psutil.NoSuchProcess):
            records.append({"pid": member.pid, "created": member.create_time()})
    return records


def stale_heartbeat(root, owner, now):
    ledger = root / "resources.json"
    heartbeat = max(ledger.stat().st_mtime if ledger.exists() else 0, owner.create_time())
    return now - heartbeat > 300


def stop_orphans(root, known):
    """Conservatively debit full orphan CPU, even if some was already counted."""
    records = list(known)
    if (root / "workers.json").exists():
        records += list(read(root / "workers.json").values())
    family = {}
    for record in records:
        owner = process(record)
        if owner:
            for item in observe_family(owner):
                family[(item["pid"], item["created"])] = item
    debit = 0.0
    remaining = []
    for record in family.values():
        owner = process(record)
        if owner:
            with contextlib.suppress(psutil.NoSuchProcess):
                debit += sum(owner.cpu_times()[:2])
                owner.terminate()
                remaining.append(owner)
    _, alive = psutil.wait_procs(remaining, timeout=10)
    for owner in alive:
        owner.kill()
    _, alive = psutil.wait_procs(alive, timeout=10)
    if alive:
        raise RuntimeError("Orphan processes remain; refusing duplicate workers")
    return debit


def launch(root, runner, python):
    log = root / f"watchdog-launch-{time.time_ns()}.log"
    with log.open("wb") as stream:
        child = subprocess.Popen(
            [str(python), str(runner), "run", "--root", str(root)],
            cwd=runner.parent.parent,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    return {"pid": child.pid, "created": psutil.Process(child.pid).create_time()}


def evidence(root, repo):
    """Commit only lossless JSON evidence, never models, prose or unrelated changes."""
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()

    if git("branch", "--show-current") in ("", "main", "master"):
        raise ValueError("Evidence commit requires an isolated non-main branch")
    if git("diff", "--cached", "--name-only"):
        raise ValueError("Index contains other work; refusing automatic commit")
    exported = read(root / "export.json")
    archive = Path(exported["path"])
    if digest(archive) != exported["sha256"] or archive.stat().st_size != exported["bytes"]:
        raise ValueError("Export checksum/size mismatch")
    device = read(root / "binding.json")["device"]
    destination = repo / "experiments/2026-09-19-hunting-curriculum/evidence" / device
    files = {}
    with zipfile.ZipFile(archive) as stream:
        manifest = json.loads(stream.read("manifest.json"))
        for name, metadata in manifest.items():
            content = stream.read(name)
            if (hashlib.sha256(content).hexdigest() != metadata["sha256"]
                    or len(content) != metadata["bytes"]):
                raise ValueError(f"Export member mismatch: {name}")
            if name.endswith(".json"):
                target = (destination / name).resolve()
                if not target.is_relative_to(destination.resolve()):
                    raise ValueError("Unsafe evidence path")
                files[name] = content
    records = ["complete.json", "export.json", "resources.json", "watchdog.json"]
    records += [p.name for p in root.glob("watchdog-recovery-*.json")]
    records += [p.name for p in root.glob("watchdog-stall-*.json")]
    for name in records:
        if (root / name).exists():
            files[name] = (root / name).read_bytes()
    # Large per-episode observations remain lossless without bloating Git.
    index, compact = {}, {}
    for name, content in files.items():
        stored = name + ".gz" if len(content) > 100_000 else name
        packed = gzip.compress(content, mtime=0) if stored != name else content
        compact[stored] = packed
        index[name] = {"stored": stored, "bytes": len(content),
                       "sha256": hashlib.sha256(content).hexdigest(),
                       "stored_sha256": hashlib.sha256(packed).hexdigest()}
    compact["evidence-index.json"] = json.dumps(index, indent=2).encode("utf-8")
    files = compact
    if sum(map(len, files.values())) > 50_000_000:
        raise ValueError("Evidence needs compact review before commit (>50MB)")
    paths = []
    for name, content in files.items():
        target = destination / name
        if target.exists() and target.read_bytes() != content:
            raise ValueError(f"Preserve prior evidence: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        paths.append(target.relative_to(repo).as_posix())
    # Explicit paths exclude any untracked README, AI log or user's work.
    for path in paths:
        git("add", "--", path)
    if git("diff", "--cached", "--name-only"):
        git("commit", "-m", f"chore(experiment): retain issue 217 {device} machine evidence")
    return git("rev-parse", "HEAD")


def watch(root, runner, python, *, interval=20, results_repo=None):
    own = psutil.Process()
    state_path = root / "watchdog.json"
    state = read(state_path) if state_path.exists() else {"restarts": 0, "cpu_seconds": 0.0}
    previous_cpu = state["cpu_seconds"]
    state.update(pid=own.pid, created=own.create_time(), runner_sha256=digest(runner))
    validate(root, runner)
    while True:
        state.update(time=time.time(), cpu_seconds=previous_cpu + sum(own.cpu_times()[:2]))
        overhead = max(0, state["cpu_seconds"] - state.get("cpu_debited", 0))
        record = read(root / "supervisor.json") if (root / "supervisor.json").exists() else {}
        owner = process(record)
        launching = process(state.get("launching", {}))
        if owner or launching:
            state["family"] = observe_family(owner or launching)
            reason = budget_reason(root, overhead)
            if reason:
                (root / "STOP.request").touch()
            ledger = root / "resources.json"
            if stale_heartbeat(root, owner or launching, time.time()):
                # The supervisor writes every 0.5s even during long evaluations.
                # Stop only verified owned processes and debit full orphan CPU.
                debit = stop_orphans(root, state["family"])
                if ledger.exists():
                    usage = read(ledger)
                    usage["cpu_seconds"] += debit
                    write(ledger, usage)
                write(root / f"watchdog-stall-{time.time_ns()}.json", {
                    "reason": "Supervisor heartbeat absent for over 300 seconds",
                    "orphan_cpu_debit": debit, "family": state["family"],
                })
                # Normal dead-process recovery applies next iteration, including
                # all persistent stop, budget, integrity and retry-limit checks.
                continue
            state["status"] = reason or "monitoring"
            write(state_path, state)
            time.sleep(interval)
            continue
        if (root / "complete.json").exists() and not (root / "STOP.json").exists():
            exported = read(root / "export.json")
            if "final_pipeline_resources" not in exported:
                raise ValueError("Completion ledger incomplete; manual review required")
            if digest(Path(exported["path"])) != exported["sha256"]:
                raise ValueError("Export checksum mismatch")
            state["status"] = "pipeline_complete"
            write(state_path, state)
            if results_repo:
                commit = evidence(root, results_repo)
                state["evidence_commit"] = commit
                write(state_path, state)
            return state
        reason = recovery_reason(root)
        budget = budget_reason(root, overhead)
        if not reason or budget or state["restarts"] >= 3 or state.get("launches", 0) >= 4:
            debit = stop_orphans(root, state.get("family", []))
            if debit and (root / "resources.json").exists():
                usage = read(root / "resources.json")
                usage["cpu_seconds"] += debit
                write(root / "resources.json", usage)
            state.update(status="stopped", reason=budget or "Persistent stop or retry limit")
            write(state_path, state)
            return state
        orphan_cpu = stop_orphans(root, state.get("family", []))
        validate(root, runner)
        # A fresh launch needs no crash debit. A recovery preserves elapsed start
        # and adds an intentionally conservative 60s sampling/termination margin.
        recovering = (root / "supervisor.json").exists()
        if recovering:
            usage = read(root / "resources.json")
            usage["cpu_seconds"] += 60 + orphan_cpu + overhead
            usage["wall_seconds"] = time.time() - usage["first_start"]
            write(root / "resources.json", usage)
            state["cpu_debited"] = state["cpu_seconds"]
            state["restarts"] += 1
        recovery = root / f"watchdog-recovery-{time.time_ns()}.json"
        write(recovery, {"reason": reason, "orphan_cpu_debit": orphan_cpu,
                         "state": state, "recovering": recovering})
        if budget_reason(root):
            state.update(status="stopped", reason="Budget exhausted before recovery")
            write(state_path, state)
            return state
        if (root / "STOP.json").exists():
            (root / "STOP.json").rename(recovery.with_suffix(".STOP.json"))
        state["launching"] = launch(root, runner, python)
        state["launches"] = state.get("launches", 0) + 1
        state["status"] = "resuming" if recovering else "starting"
        write(state_path, state)
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--results-repo", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    with exclusive(root):
        try:
            if os.name == "nt":
                import ctypes

                ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
            watch(root, args.runner.resolve(), args.python.resolve(),
                  results_repo=args.results_repo.resolve() if args.results_repo else None)
        except Exception as error:
            write(root / "watchdog-error.json", {"error": str(error), "time": time.time()})
            raise
        finally:
            if os.name == "nt":
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    main()
