"""Run the opaque collision audit for the BOMB-head fine-tuning attempt."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import audit_autonomous_antiloop_seeds as audit  # noqa: E402

audit.EXPERIMENT = audit.ROOT / "experiments/2026-09-21-bomb-head-finetune"
audit.OWN = "experiments/2026-09-21-bomb-head-finetune/"


def candidates(config):
    values = {
        config["bootstrap_seed"],
        config["evaluation_agent_seed"],
        config["smoke_seed"],
        config["training"]["optimizer_seed"],
    }
    first, last = config["collection"]["seed_range"]
    values |= set(range(first, last + 1))
    for stage in config["evaluation"].values():
        for suite in stage.values():
            values |= audit.inclusive(suite["seeds"])
    return values | {value + 1_000_000 for value in values}


audit.candidates = candidates

if __name__ == "__main__":
    audit.main()
