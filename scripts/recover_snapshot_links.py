"""Offline byte-preserving recovery of #163 hard-link saturation (no campaign imports)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
from pathlib import Path

import psutil


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def plain(path, root):
    path = Path(path).absolute()
    if not path.is_relative_to(root):
        raise ValueError("Path escapes campaign root")
    for item in (path, *path.parents):
        if item.is_symlink() or item.is_junction():
            raise ValueError("Refuse symlink/junction")
        if item == root:
            break
    return path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def quiescent(root):
    if (root / ".supervisor.lock").exists():
        raise ValueError("Supervisor lock exists")
    for path in (root / "supervisor-resources.json", root / "runs/resources.json"):
        if read(plain(path, root))["active_root_pids"]:
            raise ValueError("Active resource roots")
    ancestors = {p.pid for p in psutil.Process().parents()} | {os.getpid()}
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        if process.pid in ancestors or "python" not in (process.info["name"] or "").lower():
            continue
        command = " ".join(process.info["cmdline"] or []).lower().replace("\\", "/")
        if str(root).lower().replace("\\", "/") in command or "task3_attack_campaign" in command:
            raise ValueError(f"Campaign process still present: {process.pid}")


def inventory(root):
    root = Path(root).absolute()
    plain(root, root)
    quiescent(root)
    plans = root / "runs/plans"
    protected = set()
    pending = 0
    for status in sorted(plans.glob("*/status.json")):
        plain(status, root)
        resolved = status.with_name("resolved_plan.json")
        jobs = read(status)["jobs"]
        definitions = {j["run_id"]: j for j in read(plain(resolved, root))["jobs"]}
        for name, job in jobs.items():
            if job["status"] != "completed":
                if definitions[name]["kind"] != "evaluation" or job["status"] != "pending":
                    raise ValueError("Recovery requires only pending evaluation jobs")
                pending += 1
        protected.update((status, resolved))
    if not pending or pending + 64 >= 1000:
        raise ValueError("Remaining jobs do not fit one fresh link generation")
    protected.update(root.glob("*authorization.json"))
    protected.update(root.glob("*resources.json"))
    protected.update((root / "runs").glob("*.json"))
    protected.update((root / "binding").rglob("*"))
    for plan in plans.iterdir():
        protected.update((plan / "artifacts").rglob("*.pt"))
        protected.update((plan / "replicas").rglob("*.pt"))
    protected = {p for p in protected if p.is_file()}
    manifests = {
        str(p.relative_to(root)): {"sha256": sha(plain(p, root)), "size": p.stat().st_size}
        for p in sorted(protected)
    }
    objects = []
    for path in sorted((root / "runs/input-objects").iterdir()):
        plain(path, root)
        if (
            not path.is_file()
            or not re.fullmatch("[0-9a-f]{64}", path.name)
            or sha(path) != path.name
        ):
            raise ValueError("Invalid content-addressed object")
        info = path.stat()
        objects.append(
            {
                "name": path.name,
                "size": info.st_size,
                "inode": info.st_ino,
                "links": info.st_nlink,
                "rotate": info.st_nlink + pending + 64 >= 1000,
            }
        )
    staging = []
    for path in sorted(plans.rglob("*.tmp")):
        plain(path, root)
        if not path.is_dir() or not re.fullmatch(r"attempt-001-input-agent\.tmp", path.name):
            raise ValueError("Unexpected incomplete path")
        job = read(path.parents[2] / "status.json")["jobs"][path.parent.name]
        if job["status"] != "pending" or job["attempts"]:
            raise ValueError("Incomplete snapshot already has a recorded attempt")
        files = {
            str(p.relative_to(path)): sha(plain(p, root)) for p in path.rglob("*") if p.is_file()
        }
        staging.append({"path": str(path.relative_to(root)), "files": files})
    return {
        "scope": "storage_only_no_scientific_change",
        "pending_evaluations": pending,
        "objects": objects,
        "protected": manifests,
        "incomplete_snapshots": staging,
    }


def recover(root, audit, apply=False):
    root, audit = Path(root).absolute(), Path(audit).absolute()
    report = inventory(root)
    if not apply:
        return report
    plain(audit, root)
    if audit.parent != root or audit.exists():
        raise ValueError("Use a fresh audit directory directly inside campaign root")
    audit.mkdir()
    (audit / "before.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    shutil.copy2(root / "supervisor.err.log", audit / "original-supervisor.err.log")
    for relative in report["protected"]:
        source = root / relative
        if source.suffix == ".json":
            destination = audit / "original-records" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    with (audit / "operations.jsonl").open("x", encoding="utf-8") as log:

        def record(data):
            log.write(json.dumps(data) + "\n")
            log.flush()
            os.fsync(log.fileno())

        store = root / "runs/input-objects"
        for obj in report["objects"]:
            if not obj["rotate"]:
                continue
            source = store / obj["name"]
            if source.stat().st_ino != obj["inode"] or sha(source) != obj["name"]:
                raise ValueError("Object changed after audit")
            replacement = audit / (obj["name"] + ".new")
            with source.open("rb") as old, replacement.open("xb") as new:
                shutil.copyfileobj(old, new)
                new.flush()
                os.fsync(new.fileno())
            replacement.chmod(stat.S_IREAD)
            if sha(replacement) != obj["name"]:
                raise ValueError("Replacement bytes differ")
            backup = audit / obj["name"]
            record({"operation": "rotate_begin", **obj})
            source.rename(backup)
            try:
                replacement.rename(source)
            except BaseException:
                backup.rename(source)
                record({"operation": "rotate_rolled_back", "name": obj["name"]})
                raise
            if sha(source) != sha(backup) or os.path.samefile(source, backup):
                raise ValueError("Independent replacement verification failed")
            record(
                {
                    "operation": "rotate_complete",
                    "name": obj["name"],
                    "old_inode": backup.stat().st_ino,
                    "new_inode": source.stat().st_ino,
                }
            )
        for item in report["incomplete_snapshots"]:
            source = root / item["path"]
            destination = source.with_name(source.name + ".failed-storage")
            if destination.exists():
                raise ValueError("Preserve existing failure evidence")
            record({"operation": "preserve_incomplete_begin", "path": item["path"]})
            source.rename(destination)
            actual = {
                str(p.relative_to(destination)): sha(p)
                for p in destination.rglob("*")
                if p.is_file()
            }
            if actual != item["files"]:
                raise ValueError("Incomplete evidence changed")
            record(
                {
                    "operation": "preserve_incomplete_complete",
                    "path": str(destination.relative_to(root)),
                }
            )
    after = inventory(root)
    if after["protected"] != report["protected"]:
        raise ValueError("Protected campaign evidence changed")
    if any(obj["rotate"] for obj in after["objects"]) or after["incomplete_snapshots"]:
        raise ValueError("Recovery incomplete; retain audit")
    (audit / "after.json").write_text(json.dumps(after, indent=2), encoding="utf-8")
    return {
        "scope": report["scope"],
        "rotated_objects": sum(o["rotate"] for o in report["objects"]),
        "protected_files_unchanged": len(report["protected"]),
        "pending_evaluations": report["pending_evaluations"],
        "audit": str(audit),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.apply and args.audit is None:
        parser.error("--apply requires --audit")
    print(
        json.dumps(
            recover(args.root, args.audit or args.root / "unused-audit", args.apply), indent=2
        )
    )


if __name__ == "__main__":
    main()
