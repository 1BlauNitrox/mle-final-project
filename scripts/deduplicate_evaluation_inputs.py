"""Storage-only recovery for stopped campaigns; share identical completed input copies.

No policy/checkpoint bytes, job states, failed attempts or budgets are changed.
Only completed evaluation attempt input checkpoint.pt files are eligible. Hard
links change inode/link counts and ctime; file content, mode and mtime remain.
The runner never writes an old completed attempt's input snapshot again.
"""

import argparse
import hashlib
import json
import os
import stat
import uuid
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def inside(root, relative):
    path = root / relative
    require(not Path(relative).is_absolute(), "Absolute evidence path")
    require(path.resolve().is_relative_to(root), "Evidence path escapes root")
    require(
        all(not p.is_symlink() for p in [path, *path.parents] if p != root.parent),
        "Symlink evidence path",
    )
    return path


def identity(path):
    value = path.stat()
    require(stat.S_ISREG(value.st_mode), "Checkpoint is not a regular file")
    return (
        value.st_size,
        stat.S_IMODE(value.st_mode),
        value.st_mtime_ns,
        value.st_uid,
        value.st_gid,
        value.st_dev,
    )


def inventory(root):
    """Validate every eligible candidate before proposing any replacement."""
    records, controls = [], {}
    resources = root / "resources.json"
    if resources.exists():
        require(
            not json.loads(resources.read_text())["active_root_pids"], "Recorded active workers"
        )
        controls[resources] = digest(resources)
    for status_path in sorted((root / "plans").glob("*/status.json")):
        require(not status_path.is_symlink(), "Symlink status")
        controls[status_path] = digest(status_path)
        data = json.loads(status_path.read_text(encoding="utf-8"))
        for job in data["jobs"].values():
            if job["kind"] != "evaluation" or job["status"] != "completed":
                continue
            require(
                job["artifact"]["selection"] == "immutable evaluation input",
                "Unexpected artifact selection",
            )
            expected = job["artifact"]["sha256"]
            for attempt in job["attempts"]:
                if attempt["status"] != "completed":
                    continue
                relative = Path(attempt["output"])
                require(
                    len(relative.parts) == 3
                    and relative.parts[0] == "jobs"
                    and relative.name.startswith("attempt-"),
                    "Unexpected attempt path",
                )
                snapshot = relative.with_name(relative.name + "-input-agent") / "checkpoint.pt"
                path = inside(root, (status_path.parent / snapshot).relative_to(root))
                require(path.is_file(), "Missing completed input checkpoint: " + str(path))
                require(digest(path) == expected, "Recorded input hash mismatch: " + str(path))
                records.append((path, expected, identity(path)))
    require(records, "No verified completed evaluation inputs found")
    require(len({r[0] for r in records}) == len(records), "Duplicate input path in job records")
    pairs, first = [], {}
    for path, sha, attributes in records:
        key = (sha, attributes)
        source = first.setdefault(key, path)
        if not os.path.samefile(source, path):
            pairs.append((source, path, sha, attributes))
    return pairs, controls, len(records)


def recover(root, *, audit=None):
    root = root.resolve(strict=True)
    lock = root / ".task3-campaign.lock"
    require(not lock.exists(), "Campaign/maintenance lock exists; do not remove an active lock")
    owned_lock = False
    try:
        if audit is not None:
            require(not audit.exists(), "Preserve previous audit")
            with lock.open("x", encoding="utf-8") as file:
                file.write(f"Storage-only recovery PID {os.getpid()}\n")
            owned_lock = True
        pairs, controls, count = inventory(root)
        unique_replaced = {}
        for _, path, _, attributes in pairs:
            info = path.stat()
            unique_replaced.setdefault(
                (info.st_dev, info.st_ino), (attributes[0], info.st_nlink, 0)
            )
            size, links, n = unique_replaced[(info.st_dev, info.st_ino)]
            unique_replaced[(info.st_dev, info.st_ino)] = (size, links, n + 1)
        savings = sum(size for size, links, n in unique_replaced.values() if links == n)
        result = {
            "scope": "storage_only_no_scientific_change",
            "verified_inputs": count,
            "replacement_paths": len(pairs),
            "estimated_reclaimed_bytes": savings,
            "apply": audit is not None,
        }
        if audit is None:
            return result
        with audit.open("x", encoding="utf-8") as log:

            def record(value):
                log.write(json.dumps(value) + "\n")
                log.flush()
                os.fsync(log.fileno())

            record(dict(result, root=str(root), controls={str(p): h for p, h in controls.items()}))
            for source, target, sha, attributes in pairs:
                require(not source.is_symlink() and not target.is_symlink(), "Changed path type")
                require(
                    identity(source) == identity(target) == attributes, "Changed file attributes"
                )
                require(digest(source) == digest(target) == sha, "Changed checkpoint bytes")
                temporary = target.with_name(target.name + ".dedup-" + uuid.uuid4().hex)
                record(
                    {
                        "phase": "before",
                        "source": str(source),
                        "target": str(target),
                        "sha256": sha,
                        "size_bytes": attributes[0],
                    }
                )
                try:
                    os.link(source, temporary)
                    os.replace(temporary, target)
                finally:
                    if temporary.exists():
                        temporary.unlink()
                require(
                    digest(target) == sha and identity(target) == attributes,
                    "Post-replacement verification failed",
                )
                record({"phase": "verified", "target": str(target), "sha256": sha})
            require(
                all(digest(p) == h for p, h in controls.items()), "Job/resource records changed"
            )
            record({"phase": "complete", **result})
        return result
    finally:
        if owned_lock:
            lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    require(args.apply == (args.audit is not None), "Use --apply with a new --audit path")
    print(json.dumps(recover(args.root, audit=args.audit), indent=2))


if __name__ == "__main__":
    main()
