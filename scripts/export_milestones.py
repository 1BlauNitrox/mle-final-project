"""Copy retained milestones out of a running training root, for evaluation elsewhere.

Read-only on the run root. Writes one zip holding reference.pt and
training-resume/<job>/milestone-<episode>.pt in the same layout
evaluate_milestones.py expects, plus a manifest of SHA-256 digests so the copy
evaluated on another machine is provably the checkpoint that was trained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Zip file to write.")
    parser.add_argument("--episodes", type=int, nargs="*", help="Only these milestone episodes.")
    args = parser.parse_args()

    root = args.root.resolve()
    files = [root / "reference.pt"]
    for path in sorted(root.glob("training-resume/*/milestone-*.pt")):
        episode = int(re.search(r"milestone-(\d+)\.pt$", path.name).group(1))
        if not args.episodes or episode in args.episodes:
            files.append(path)
    if len(files) == 1:
        raise SystemExit("no milestones matched")

    binding = json.loads((root / "binding.json").read_text(encoding="utf-8"))
    manifest = {
        "created_readable": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_root": str(root),
        "profile": binding["profile"],
        "config_sha256": binding["config_sha256"],
        "use": "development-world monitoring only; the held-out suite is not evaluated from this bundle",
        "files": {},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            name = path.relative_to(root).as_posix()
            manifest["files"][name] = {"sha256": sha256(path), "size_bytes": path.stat().st_size}
            archive.write(path, name)
        archive.writestr("MANIFEST.json", json.dumps(manifest, indent=2) + "\n")

    print(f"wrote {args.out} ({args.out.stat().st_size / 1024**2:.1f} MiB, {len(files)} checkpoints)")
    for name in manifest["files"]:
        print("  ", name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
