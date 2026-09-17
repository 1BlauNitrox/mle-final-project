"""Check the milestone monitoring outputs on a laptop, and say what is new.

  --snapshot before   record which checkpoints each suite already has games for
  --snapshot after    verify every suite is complete and clean, then list the
                      checkpoints that were evaluated since the "before" snapshot

"Complete" means every checkpoint under the root (the reference and every
milestone) has exactly one game on every world of every monitored suite, all
played by Bomb-omb on the right suite, and nothing else. It prints ALL CHECKS
PASS and exits 0 only in that case.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FINAL = REPO_ROOT / "experiments/2026-09-17-task4-final-training/config.json"
MONITORING = REPO_ROOT / "experiments/2026-09-17-final-training-monitoring/config.json"
SNAPSHOT = "monitoring-inventory-before.json"


def suites() -> dict[str, tuple[str, list[int]]]:
    final = json.loads(FINAL.read_text(encoding="utf-8"))["evaluation_suites"]
    classic = json.loads(MONITORING.read_text(encoding="utf-8"))["suite"]
    return {
        classic["name"]: (f"milestone-evaluation-{classic['name']}", classic["world_seeds"]),
        "coin-heaven": ("milestone-evaluation-coin-heaven", final["coin-heaven"]["world_seeds"]),
        "loot-crate": ("milestone-evaluation-loot-crate", final["loot-crate"]["world_seeds"]),
    }


def checkpoints(root: Path) -> list[str]:
    names = ["reference"] if (root / "reference.pt").is_file() else []
    for path in sorted((root / "training-resume").glob("*/milestone-*.pt")):
        episode = int(re.search(r"milestone-(\d+)\.pt$", path.name).group(1))
        names.append(f"{path.parent.name}@{episode}")
    return names


def games(folder: Path) -> list[dict]:
    log = folder / "games.jsonl"
    if not log.is_file():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def evaluated(root: Path) -> dict[str, list[str]]:
    return {name: sorted({g["artifact"] for g in games(root / folder)}) for name, (folder, _) in suites().items()}


def check(root: Path) -> tuple[bool, list[str]]:
    lines, ok = [], True
    expected_artifacts = checkpoints(root)
    for name, (folder, seeds) in suites().items():
        rows = games(root / folder)
        keys = [(g["artifact"], g["world_seed"]) for g in rows]
        expected = {(a, s) for a in expected_artifacts for s in seeds}
        missing, extra = expected - set(keys), set(keys) - expected
        duplicates = len(keys) - len(set(keys))
        wrong = sum(1 for g in rows if g.get("agent") != "Bomb-omb" or g.get("suite") != name)
        files = all((root / folder / f).is_file() for f in ("games.jsonl", "summary.csv", "paired.csv"))
        passed = files and not missing and not extra and not duplicates and not wrong
        ok = ok and passed
        lines.append(
            f"{'PASS' if passed else 'FAIL'} {name}: {len(rows)} games, expected {len(expected)} "
            f"({len(expected_artifacts)} checkpoints x {len(seeds)} worlds); missing {len(missing)}, "
            f"extra {len(extra)}, duplicates {duplicates}, wrong agent or suite {wrong}, files present {files}"
        )
        if missing:
            lines.append(f"     missing examples: {sorted(missing)[:3]}")
    return ok, lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--snapshot", choices=("before", "after"), required=True)
    args = parser.parse_args()
    root = args.root.resolve()

    if args.snapshot == "before":
        inventory = evaluated(root)
        (root / SNAPSHOT).write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
        present = checkpoints(root)
        for name, done in inventory.items():
            print(f"{name}: {len(done)} checkpoints already evaluated, "
                  f"{len(set(present) - set(done))} to evaluate")
        return 0

    ok, lines = check(root)
    print("\n".join(lines))
    before_path = root / SNAPSHOT
    before = json.loads(before_path.read_text(encoding="utf-8")) if before_path.is_file() else {}
    after = evaluated(root)
    new = {name: sorted(set(after[name]) - set(before.get(name, []))) for name in after}
    print("NEWLY EVALUATED: " + json.dumps(new))
    print("ALL CHECKS PASS" if ok else "VALIDATION FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
