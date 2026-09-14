"""Audit fresh pilot seeds against tracked registrations without revealing held-out values."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path

from scripts.pilot_task3_learning_rate import ROOT, config, write


def populations():
    cfg = config()
    train = {s for seeds in cfg["training_world_seeds"] for s in seeds}
    train |= {s + 1000000 for s in list(train)}
    train |= set(cfg["replica_agent_seeds"])
    dev = {s for suite in cfg["evaluation_suites"].values() for s in suite["world_seeds"]}
    dev |= {s + 1000000 for s in list(dev)}
    smoke = set(cfg["smoke"]["world_seeds"]) | {cfg["smoke"]["agent_seed"]}
    if train & dev or train & smoke or dev & smoke:
        raise ValueError("New populations collide")
    return {"training_including_opponents": train, "development": dev, "smoke": smoke}


def audit():
    import re

    groups = populations()
    candidates = set().union(*groups.values())
    refs = subprocess.check_output(
        ["git", "for-each-ref", "--format=%(refname)", "refs/remotes/origin"], cwd=ROOT, text=True
    ).splitlines()
    refs = [r for r in refs if not r.endswith("/HEAD") and "/171-" not in r]
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
            if Path(name).suffix in {".json", ".yaml", ".yml", ".py", ".md"}:
                objects.setdefault(meta.split()[2], name)
    # Stream immutable blobs once. Values from old/final registrations are never output.
    process = subprocess.Popen(
        ["git", "cat-file", "--batch"], cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    collisions = set()
    try:
        for digest, name in objects.items():
            process.stdin.write((digest + "\n").encode())
            process.stdin.flush()
            header = process.stdout.readline().split()
            content = process.stdout.read(int(header[2]))
            process.stdout.read(1)
            text = content.decode("utf-8-sig", errors="replace")
            literals = {
                int(x.replace("_", "")) for x in re.findall(r"(?<![\w.])\d[\d_]*(?![\w.])", text)
            }
            if literals & candidates:
                collisions.add(name)
            # Cover literal range(...) and human-readable inclusive a..b registrations.
            for a, b in re.findall(r"(\d{5,})\s*(?:\.\.|through)\s*(\d{5,})", text):
                if any(int(a) <= c <= int(b) for c in candidates):
                    collisions.add(name)
            if name.endswith(".py"):
                try:
                    tree = ast.parse(text)
                except SyntaxError:
                    continue
                for node in ast.walk(tree):
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "range"
                        and 1 <= len(node.args) <= 3
                        and all(
                            isinstance(x, ast.Constant) and type(x.value) is int for x in node.args
                        )
                    ):
                        values = range(*(x.value for x in node.args))
                        if any(c in values for c in candidates):
                            collisions.add(name)
    finally:
        process.stdin.close()
        process.wait()
    result = {
        "scope": "opaque collision-only audit; no held-out values exposed or games run",
        "populations": {k: len(v) for k, v in groups.items()},
        "remote_revisions": revisions,
        "unique_text_blobs": len(objects),
        "blob_set_sha256": hashlib.sha256("\n".join(sorted(objects)).encode()).hexdigest(),
        "colliding_files": sorted(collisions),
        "passed": not collisions,
        "method": (
            "all integer literals, literal Python ranges and inclusive written ranges "
            "across remote tracked registrations; no dynamic code executed"
        ),
    }
    if collisions:
        raise ValueError("Seed audit collision in " + str(sorted(collisions)))
    return result


if __name__ == "__main__":
    output = ROOT / "experiments/2026-09-13-task3-learning-rate-retention/seed-audit.json"
    result = audit()
    write(output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "remote_revisions"}, indent=2))
