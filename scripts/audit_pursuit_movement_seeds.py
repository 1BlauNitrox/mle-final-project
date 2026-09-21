"""Audit separation of movement-only pursuit evaluation seeds."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/2026-09-21-pursuit-movement-only/config.json"
OUTPUT = CONFIG.with_name("seed-audit.json")


def main() -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    groups = {}
    used = set()
    for stage, suites in cfg["evaluation"].items():
        for suite, setting in suites.items():
            seeds = set(range(setting["seeds"][0], setting["seeds"][1] + 1))
            if used & seeds:
                raise ValueError(f"Evaluation seed collision in {stage}.{suite}")
            used |= seeds
            groups[f"{stage}.{suite}"] = {
                "first": min(seeds),
                "last": max(seeds),
                "count": len(seeds),
            }
    special = {cfg["evaluation_agent_seed"], cfg["smoke_seed"]}
    if len(special) != 2 or used & special:
        raise ValueError("Special seed collision")
    OUTPUT.write_text(
        json.dumps({"passed": True, "groups": groups, "special": sorted(special)}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
