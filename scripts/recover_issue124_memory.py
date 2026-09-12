"""Record an owner-authorized recovery from the Issue 124 free-memory stop."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import psutil

REASON = "System free RAM fell below 1 GiB"


def resumed_state(state, available, now):
    """Permit only a recovered system-memory floor, never exhausted budgets."""
    if state.get("limit_reached") != REASON or state.get("active_root_pids") != []:
        raise ValueError("Recovery requires an inactive system-memory-floor stop")
    if available < 4 * 1024**3:
        raise ValueError("At least 4 GiB available RAM required")
    elapsed = now - datetime.fromisoformat(state["authorized_at"]).timestamp()
    limits = state["limits"]
    if (elapsed >= limits["wall_seconds"]
            or state["cpu_seconds_consumed"] >= limits["cpu_seconds"]
            or state["peak_memory_bytes"] > limits["memory_bytes"]):
        raise ValueError("Resource budget exhausted; cannot recover")
    return {**state, "limit_reached": None, "wall_seconds_elapsed": elapsed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--owner-instruction", required=True)
    args = parser.parse_args()
    root = args.campaign_root.resolve()
    if (root / "campaign.lock").exists():
        raise ValueError("Inspect existing campaign lock before recovery")
    state = json.loads((root / "resources.json").read_text())
    available = psutil.virtual_memory().available
    now = datetime.now(timezone.utc)
    updated = resumed_state(state, available, now.timestamp())
    amendment = json.loads((root / "budget-amendment.json").read_text())
    if state["limits"] != amendment["new_limits"]:
        raise ValueError("Limits differ from owner amendment")
    history = root / "runtime-recovery-history" / now.strftime("%Y%m%d-%H%M%S-memory")
    history.mkdir(parents=True, exist_ok=False)
    names = ["resources.json", "campaign-status.json", "budget-amendment.json"]
    for name in names:
        shutil.copy2(root / name, history / name)
    shutil.copy2(__file__, history / Path(__file__).name)
    paths = [history / name for name in names] + [history / Path(__file__).name]
    record = {
        "kind": "owner_authorized_system_memory_recovery",
        "instruction": args.owner_instruction,
        "recorded_at": now.isoformat(),
        "available_memory_bytes": available,
        "prior_breach": REASON,
        "accounting": "Original authorization, CPU, peak RAM and wall origin preserved",
        "snapshots": {p.relative_to(root).as_posix(): {
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "size_bytes": p.stat().st_size} for p in paths},
    }
    amendment.setdefault("operational_recoveries", []).append(record)
    amendment["preserved_records"].extend(p.relative_to(root).as_posix() for p in paths)
    # Commit the historical breach record before releasing the current segment.
    (root / "budget-amendment.json").write_text(json.dumps(amendment, indent=2) + "\n")
    temporary = root / "resources.recovery.tmp"
    temporary.write_text(json.dumps(updated, indent=2) + "\n")
    temporary.replace(root / "resources.json")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
