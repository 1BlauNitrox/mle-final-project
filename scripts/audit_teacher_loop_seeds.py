"""Opaque collision audit for every Issue #221 seed and derived opponent offset."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from bisect import bisect_left
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/2026-09-20-teacher-loop"
OWN = "experiments/2026-09-20-teacher-loop/"


def inclusive(pair):
    first, last = pair
    return set(range(first, last + 1))


def candidates(config):
    values = {config["bootstrap_seed"], config["evaluation_agent_seed"], *config["learner_seeds"]}
    values |= set(range(config["smoke_seed"], config["smoke_seed"] + 3))
    for pair in config["train_ranges"]:
        values |= inclusive(pair)
    for stage in config["evaluation"].values():
        for suite in stage.values():
            values |= inclusive(suite["seeds"])
    values |= {
        seed + 168000000 + block
        for seed in config["learner_seeds"]
        for block in range(config["episodes"] // 5)
    }
    return values | {value + 1_000_000 for value in values}


def main():
    config = json.loads((EXPERIMENT / "config.json").read_text(encoding="utf-8-sig"))
    proposed = candidates(config)
    ordered = sorted(proposed)
    refs = subprocess.check_output(
        ["git", "for-each-ref", "--format=%(refname)", "refs/remotes/origin"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    refs = [ref for ref in refs if not ref.endswith("/HEAD")]
    objects = {}
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
                ".ps1",
            }:
                objects.setdefault(meta.split()[2], name)
    process = subprocess.Popen(
        ["git", "cat-file", "--batch"],
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
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
            if literals & proposed:
                collisions.add(name)
            for first, last in re.findall(r"(\d{8,})\s*(?:\.\.|through|,)\s*(\d{8,})", content):
                index = bisect_left(ordered, int(first))
                if index < len(ordered) and ordered[index] <= int(last):
                    collisions.add(name)
    finally:
        process.stdin.close()
        process.wait()
    result = {
        "scope": "opaque collision-only audit; no held-out values output or games run",
        "candidate_and_offset_count": len(proposed),
        "remote_revisions": revisions,
        "unique_text_blobs": len(objects),
        "blob_set_sha256": hashlib.sha256("\n".join(sorted(objects)).encode()).hexdigest(),
        "excluded_own_registration": OWN,
        "colliding_files": sorted(collisions),
        "passed": not collisions,
    }
    (EXPERIMENT / "seed-audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    if collisions:
        raise RuntimeError(f"Registered seeds collide in: {sorted(collisions)}")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "remote_revisions"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
