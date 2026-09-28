"""Export every milestone episode that is complete but not yet bundled.

We run this on the training PC whenever new milestones may have landed. For each
episode where all jobs have retained a milestone, and no existing bundle already
holds all of them, it writes one bundle with the reference and that episode's
checkpoints, then verifies the zip before reporting it.

Training writes a milestone with a plain file copy, not an atomic rename, so a
file that was modified in the last MIN_AGE_SECONDS, or that does not load, is
treated as not ready yet. Read-only on the run root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MIN_AGE_SECONDS = 60
MILESTONE = re.compile(r"milestone-(\d+)\.pt$")


def jobs_of(root: Path) -> list[str]:
    return sorted(p.name for p in (root / "training-resume").iterdir() if p.is_dir())


def complete_episodes(root: Path, now: float, min_age: float = MIN_AGE_SECONDS) -> list[int]:
    """Episodes for which every job has a milestone old enough to be fully written."""
    jobs = jobs_of(root)
    ready: dict[int, set[str]] = {}
    for job in jobs:
        for path in (root / "training-resume" / job).glob("milestone-*.pt"):
            if now - path.stat().st_mtime < min_age:
                continue
            ready.setdefault(int(MILESTONE.search(path.name).group(1)), set()).add(job)
    return sorted(ep for ep, have in ready.items() if have == set(jobs))


def exported_episodes(bundles: Path, jobs: list[str]) -> set[int]:
    done = set()
    for zip_path in bundles.glob("*.zip"):
        try:
            with zipfile.ZipFile(zip_path) as zf:
                files = json.loads(zf.read("MANIFEST.json"))["files"]
        except (zipfile.BadZipFile, KeyError, json.JSONDecodeError):
            continue
        episodes = {int(m.group(1)) for name in files if (m := MILESTONE.search(name))}
        for ep in episodes:
            if all(f"training-resume/{job}/milestone-{ep:06d}.pt" in files for job in jobs):
                done.add(ep)
    return done


def loads(path: Path) -> bool:
    import torch

    try:
        torch.load(path, map_location="cpu", weights_only=True)
        return True
    except Exception:  # noqa: BLE001 - any failure means "not ready"
        return False


def verify(zip_path: Path, root: Path) -> list[str]:
    problems = []
    with zipfile.ZipFile(zip_path) as zf:
        if zf.testzip() is not None:
            problems.append("zip CRC check failed")
        files = json.loads(zf.read("MANIFEST.json"))["files"]
        for name, meta in files.items():
            inside = hashlib.sha256(zf.read(name)).hexdigest()
            if inside != meta["sha256"]:
                problems.append(f"{name}: zip content differs from manifest")
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != meta["sha256"]:
                problems.append(f"{name}: run-root file changed since export")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=Path("C:/task4-final-training"))
    parser.add_argument(
        "--bundles", type=Path, default=Path("D:/bomberman-archive/milestone-bundles")
    )
    args = parser.parse_args()

    root = args.root.resolve()
    args.bundles.mkdir(parents=True, exist_ok=True)
    jobs = jobs_of(root)
    complete = complete_episodes(root, time.time())
    pending = [ep for ep in complete if ep not in exported_episodes(args.bundles, jobs)]
    print(
        f"jobs: {len(jobs)} | complete episodes: {complete} | already bundled: "
        f"{sorted(set(complete) - set(pending))} | to export: {pending}"
    )

    written = []
    for ep in pending:
        files = [root / "training-resume" / job / f"milestone-{ep:06d}.pt" for job in jobs]
        unready = [f.parent.name for f in files if not loads(f)]
        if unready:
            print(f"episode {ep}: not ready yet, a checkpoint does not load for {unready}")
            continue
        out = args.bundles / f"final-training-ep{ep:06d}.zip"
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/export_milestones.py"),
                "--root",
                str(root),
                "--out",
                str(out),
                "--episodes",
                str(ep),
            ],
            check=True,
            capture_output=True,
        )
        problems = verify(out, root)
        if problems:
            out.unlink(missing_ok=True)
            print(f"episode {ep}: export failed verification, removed: {problems}")
            continue
        info = {
            "episode": ep,
            "path": str(out),
            "size_mib": round(out.stat().st_size / 1024**2, 2),
            "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "checkpoints": len(jobs) + 1,
        }
        written.append(info)
        print("EXPORTED " + json.dumps(info))

    print(f"NEW BUNDLES: {len(written)}" if written else "NO NEW MILESTONES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
