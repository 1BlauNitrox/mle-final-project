"""Opaque freshness audit for the Issue #205 development and smoke seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OWN = "experiments/2026-09-19-hunting-agent-screen/"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / OWN / "config.json")
    parser.add_argument("--output", type=Path, default=ROOT / OWN / "seed-audit.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    candidates = set(config["design"]["world_seeds"]) | {config["design"]["smoke_seed"]}
    refs = subprocess.check_output(
        ["git", "for-each-ref", "--format=%(refname)", "refs/remotes/origin"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    refs = [ref for ref in refs if not ref.endswith("/HEAD")]
    objects: dict[str, str] = {}
    revisions = {}
    for ref in refs:
        revisions[ref] = subprocess.check_output(
            ["git", "rev-parse", ref], cwd=ROOT, text=True
        ).strip()
        listing = subprocess.check_output(
            [
                "git",
                "ls-tree",
                "-r",
                ref,
                "--",
                "training",
                "experiments",
                "docs",
                "scripts",
                "tests",
            ],
            cwd=ROOT,
            text=True,
        )
        for line in listing.splitlines():
            meta, name = line.split("\t", 1)
            if not name.startswith(OWN) and Path(name).suffix in {
                ".json",
                ".yaml",
                ".yml",
                ".py",
                ".md",
            }:
                objects.setdefault(meta.split()[2], name)
    process = subprocess.Popen(
        ["git", "cat-file", "--batch"], cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    collisions = set()
    try:
        for digest, name in objects.items():
            process.stdin.write((digest + "\n").encode())
            process.stdin.flush()
            header = process.stdout.readline().split()
            content = process.stdout.read(int(header[2])).decode("utf-8-sig", errors="replace")
            process.stdout.read(1)
            literals = {
                int(value.replace("_", ""))
                for value in re.findall(r"(?<![\w.])\d[\d_]*(?![\w.])", content)
            }
            if literals & candidates:
                collisions.add(name)
            for first, last in re.findall(r"(\d{5,})\s*(?:\.\.|through)\s*(\d{5,})", content):
                if any(int(first) <= seed <= int(last) for seed in candidates):
                    collisions.add(name)
    finally:
        process.stdin.close()
        process.wait()
    result = {
        "scope": "opaque collision-only audit; no held-out values exposed or games run",
        "candidate_count": len(candidates),
        "remote_revisions": revisions,
        "unique_text_blobs": len(objects),
        "blob_set_sha256": hashlib.sha256("\n".join(sorted(objects)).encode()).hexdigest(),
        "excluded_own_registration": OWN,
        "colliding_files": sorted(collisions),
        "passed": not collisions,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if collisions:
        raise RuntimeError(f"Registered seeds collide in: {sorted(collisions)}")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "remote_revisions"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
