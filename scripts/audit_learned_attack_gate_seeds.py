"""Audit separation of all learned attack-gate evaluation seed ranges."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/2026-09-21-learned-attack-gate/config.json"
OUTPUT = CONFIG.with_name("seed-audit.json")


def main() -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    groups = {}
    used = set()
    for stage, suites in cfg["evaluation"].items():
        for suite, setting in suites.items():
            seeds = set(range(setting["seeds"][0], setting["seeds"][1] + 1))
            overlap = used & seeds
            if overlap:
                raise ValueError(f"Seed collision in {stage}.{suite}: {min(overlap)}")
            used |= seeds
            groups[f"{stage}.{suite}"] = {
                "first": min(seeds),
                "last": max(seeds),
                "count": len(seeds),
            }
    special = {cfg["evaluation_agent_seed"], cfg["smoke_seed"]}
    if used & special or len(special) != 2:
        raise ValueError("Special seeds collide with evaluation ranges")
    OUTPUT.write_text(
        json.dumps({"passed": True, "groups": groups, "special": sorted(special)}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
