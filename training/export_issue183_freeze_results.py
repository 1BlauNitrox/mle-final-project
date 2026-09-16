"""Export compact, reproducible Issue #183 confirmation evidence."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOBS = (
    ROOT
    / "training_outputs/confirmation-issue183/"
    "issue153-compact-task2-control/jobs"
)
OUTPUT = (
    ROOT
    / "agent_code/DerKleineSprengstoffkapitalist/"
    "freeze-reference-results.csv"
)
JOB_NAME = re.compile(
    r"eval-r2-(classic|coin-heaven|loot-crate)-"
    r"(primary|repeat)-seed-(\d{3})"
)

rows = []
for episode_file in sorted(JOBS.glob("*/attempt-001/episodes.csv")):
    job_name = episode_file.parents[1].name
    match = JOB_NAME.fullmatch(job_name)
    if match is None:
        continue

    scenario, phase, _ = match.groups()
    attempt = episode_file.parent
    metadata = json.loads((attempt / "metadata.json").read_text())

    with episode_file.open(newline="", encoding="utf-8") as handle:
        episodes = list(csv.DictReader(handle))
    if len(episodes) != 1:
        raise ValueError(f"Expected one episode: {episode_file}")
    episode = episodes[0]

    available = int(episode["initially_available_coins"])
    collected = int(episode["coins_collected"])
    rows.append(
        {
            "scenario": scenario,
            "phase": phase,
            "world_seed": metadata["world_seed"],
            "agent_seed": metadata["agent_seed"],
            "coins_collected": collected,
            "initially_available_coins": available,
            "collection_fraction": collected / available if available else 0.0,
            "self_kills": episode["self_kills"],
            "action_bomb": episode["action_bomb"],
            "invalid_actions": episode["invalid_actions"],
            "termination_reason": episode["termination_reason"],
            "executed_action_sequence_sha256": episode[
                "executed_action_sequence_sha256"
            ],
        }
    )

if len(rows) != 600:
    raise ValueError(f"Expected 600 evaluation rows, found {len(rows)}")

fields = list(rows[0])
with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUTPUT}")