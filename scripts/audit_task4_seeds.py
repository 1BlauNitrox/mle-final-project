"""Audit fresh approach-pilot seeds against tracked registrations and each other.

The audit is opaque: it only reports whether any candidate seed already appears
in a tracked registration, never which held-out values exist. It also checks the
two new profiles against one another, because they are meant to run as
independent comparisons on two hosts at the same time.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

from scripts.pilot_task4_competition import PROFILES, ROOT, profile_dir, write


def profile_config(profile):
    path = ROOT / f"experiments/{profile_dir(profile)}/config.json"
    return json.loads(path.read_text(encoding="utf-8"))


def populations(cfg):
    train = {s for seeds in cfg["training_world_seeds"] for s in seeds}
    train |= {s + 1000000 for s in list(train)}
    train |= set(cfg["replica_agent_seeds"])
    dev = {s for suite in cfg["evaluation_suites"].values() for s in suite["world_seeds"]}
    dev |= {s + 1000000 for s in list(dev)}
    smoke = set(cfg["smoke"]["world_seeds"]) | {cfg["smoke"]["agent_seed"]}
    if train & dev or train & smoke or dev & smoke:
        raise ValueError("New populations collide")
    return {"training_including_opponents": train, "development": dev, "smoke": smoke}


def cross_profile():
    """Check every registered profile pair, not just the first two."""
    groups = {profile: populations(profile_config(profile)) for profile in PROFILES}
    merged = {profile: set().union(*value.values()) for profile, value in groups.items()}
    for index, first in enumerate(PROFILES):
        for second in PROFILES[index + 1 :]:
            if merged[first] & merged[second]:
                raise ValueError(f"Profiles {first} and {second} share seeds")
    return {
        "profiles": {p: {k: len(v) for k, v in g.items()} for p, g in groups.items()},
        "compared_pairs": len(PROFILES) * (len(PROFILES) - 1) // 2,
        "shared_seed_count": 0,
        "passed": True,
    }


def audit(profile):
    groups = populations(profile_config(profile))
    candidates = set().union(*groups.values())
    # Once a registration is pushed its own blobs are tracked, and every one of
    # its seeds then "collides" with itself. A file matching itself is not
    # evidence of reuse, so the profile's own registration directory is excluded
    # and the audit stays rerunnable after registration.
    own = f"experiments/2026-09-15-task4-{profile}/"
    # The report also has to converge. Preparation requires a committed report,
    # committing and pushing it advances the branch that carries it, and the
    # branch's own revision is part of what the report records - so each run
    # produced a different report than the one just committed and preparation
    # could never reach a clean tree. That branch's blobs are still scanned; only
    # its revision is left out of the record, which is the one field that cannot
    # be recorded without describing the act of recording it.
    head = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True
    ).strip()
    own_ref = f"refs/remotes/origin/{head}" if head != "HEAD" else None
    refs = subprocess.check_output(
        ["git", "for-each-ref", "--format=%(refname)", "refs/remotes/origin"], cwd=ROOT, text=True
    ).splitlines()
    refs = [r for r in refs if not r.endswith("/HEAD") and "task3-approach" not in r]
    objects, revisions = {}, {}
    for ref in refs:
        if ref != own_ref:
            revisions[ref] = subprocess.check_output(
                ["git", "rev-parse", ref], cwd=ROOT, text=True
            ).strip()
        listing = subprocess.check_output(
            ["git", "ls-tree", "-r", ref, "--", "training", "experiments", "docs", "scripts",
             "tests"],
            cwd=ROOT,
            text=True,
        )
        for line in listing.splitlines():
            meta, name = line.split("\t", 1)
            if name.startswith(own):
                continue
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
        "profile": profile,
        "scope": "opaque collision-only audit; no held-out values exposed or games run",
        "populations": {k: len(v) for k, v in groups.items()},
        "remote_revisions": revisions,
        "unique_text_blobs": len(objects),
        "blob_set_sha256": hashlib.sha256("\n".join(sorted(objects)).encode()).hexdigest(),
        "excluded_own_registration": own,
        "excluded_own_branch_revision": own_ref,
        "colliding_files": sorted(collisions),
        "cross_profile_separation": cross_profile(),
        "passed": not collisions,
        "method": (
            "all integer literals, literal Python ranges and inclusive written ranges "
            "across remote tracked registrations; no dynamic code executed"
        ),
    }
    if collisions:
        raise ValueError("Seed audit collision in " + str(sorted(collisions)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default=os.environ.get(
        "TASK4_PROFILE", PROFILES[0]
    ))
    args = parser.parse_args()
    result = audit(args.profile)
    write(ROOT / f"experiments/{profile_dir(args.profile)}/seed-audit.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "remote_revisions"}, indent=2))


if __name__ == "__main__":
    main()
