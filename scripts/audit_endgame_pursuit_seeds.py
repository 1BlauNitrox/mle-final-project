"""Audit separation of learned endgame-pursuit seed ranges."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/2026-09-21-learned-endgame-pursuit/config.json"
OUTPUT = CONFIG.with_name("seed-audit.json")


def main() -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    groups = {
        "collection": set(
            range(cfg["collection"]["seed_range"][0], cfg["collection"]["seed_range"][1] + 1)
        )
    }
    for stage, suites in cfg["evaluation"].items():
        for suite, setting in suites.items():
            groups[f"{stage}.{suite}"] = set(range(setting["seeds"][0], setting["seeds"][1] + 1))
    names = list(groups)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if groups[left] & groups[right]:
                raise ValueError(f"Seed collision: {left} and {right}")
    special = {cfg["evaluation_agent_seed"], cfg["smoke_seed"], cfg["training"]["optimizer_seed"]}
    if len(special) != 3 or any(values & special for values in groups.values()):
        raise ValueError("Special seed collision")
    OUTPUT.write_text(
        json.dumps(
            {
                "passed": True,
                "groups": {
                    name: {"first": min(values), "last": max(values), "count": len(values)}
                    for name, values in groups.items()
                },
                "special": sorted(special),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
