"""Verify retained Issue 162 observations without games, checkpoint loading or training."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import statistics
import zipfile
from pathlib import Path

from scripts.diagnose_issue162 import summarize

PROGRESS = {"COIN_COLLECTED", "CRATE_DESTROYED", "KILLED_OPPONENT"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stall_windows(rows, width=24):
    """Same registered period-1..4 predicate as instrumentation e960bcc."""
    windows = []
    for end in range(width, len(rows) + 1):
        part = rows[end - width : end]
        if any(r["hazards"] or PROGRESS.intersection(r["events"]) for r in part):
            continue
        if len({r["board_coins_sha256"] for r in part}) != 1:
            continue
        positions = [r["position"] for r in part]
        periods = [
            p for p in range(1, 5) if all(positions[i] == positions[i % p] for i in range(width))
        ]
        if periods:
            windows.append(
                {"start": part[0]["step"], "end": part[-1]["step"], "period": min(periods)}
            )
    return windows


def verify(record):
    manifest = json.loads((record / "evidence-manifest.json").read_text())
    archive = record / manifest["archive"]
    require(archive.stat().st_size == manifest["size_bytes"], "Archive size mismatch")
    require(
        hashlib.sha256(archive.read_bytes()).hexdigest() == manifest["sha256"],
        "Archive checksum mismatch",
    )
    saved = json.loads((record / "summary.json").read_text())
    config = json.loads((record / "config.json").read_text())
    with (record / "observations.csv").open(newline="") as stream:
        compact = list(csv.DictReader(stream))
    require(saved["status"] == "completed", "Incomplete diagnosis")
    require(len(saved["jobs"]) == config["max_episodes"] == 15, "Incomplete job matrix")
    expected = {
        f"{r['replica']}-{c['world_seed']}" for r in config["replicas"] for c in config["cases"]
    }
    seen, episodes, trajectories = set(), [], {}
    with zipfile.ZipFile(archive) as bundle:
        require(len(bundle.namelist()) == manifest["members"] == 32, "Wrong member count")
        require(json.loads(bundle.read("registration.json")) == config, "Registration mismatch")
        for job in saved["jobs"]:
            name = f"{job['replica']}-{job['world_seed']}"
            require(name in expected and name not in seen, "Unexpected or duplicate job")
            seen.add(name)
            payload = bundle.read(f"{name}/trajectory.json.gz")
            require(
                hashlib.sha256(payload).hexdigest() == job["trajectory_sha256"],
                "Trajectory checksum mismatch",
            )
            rows = json.loads(gzip.decompress(payload))
            native = json.loads(bundle.read(f"{name}/summary.json"))
            require(
                native["agent_seed"] == job["world_seed"] + config["agent_seed_offset"],
                "Agent seed mismatch",
            )
            require(
                native["world_seed"] == job["world_seed"] and native["opponent"] == job["opponent"],
                "Episode identity mismatch",
            )
            replica = next(r for r in config["replicas"] if r["replica"] == job["replica"])
            require(
                native["checkpoint_sha256"] == job["checkpoint_sha256"] == replica["sha256"],
                "Checkpoint identity mismatch",
            )
            require(
                native["checkpoint_size_bytes"] == int(replica["size_bytes"]),
                "Checkpoint size mismatch",
            )
            require(
                [r["step"] for r in rows] == list(range(1, len(rows) + 1)),
                "Missing or duplicate steps",
            )
            metrics = summarize(rows)
            require(metrics == job["metrics"], "Recomputed metrics differ")
            stalls = stall_windows(rows)
            require(
                stalls == job["stall_windows"] == native["stall_windows"],
                "Recomputed stall windows differ",
            )
            small = [r for r in compact if r["run"] == name]
            require(len(small) == len(rows), "Compact step count mismatch")
            for row, item in zip(rows, small, strict=True):
                f, state = row["features"], row["state"]
                positions = [o[3] for o in state["others"]]
                distance = min(
                    (
                        abs(row["position"][0] - p[0]) + abs(row["position"][1] - p[1])
                        for p in positions
                    ),
                    default=None,
                )
                available = bool(f[31] and state["self"][2] and row["legal_mask"][5])
                require(
                    int(item["step"]) == row["step"]
                    and [int(item["x"]), int(item["y"])] == row["position"]
                    and [int(item["next_x"]), int(item["next_y"])] == row["next_position"]
                    and item["board_coins_sha256"] == row["board_coins_sha256"]
                    and item["hazards"] == str(row["hazards"])
                    and json.loads(item["events"]) == row["events"],
                    "Compact state/event mismatch",
                )
                require(
                    item["opponent_distance"] == ("" if distance is None else str(distance)),
                    "Compact opponent distance mismatch",
                )
                require(
                    int(item["attack"]) == bool(f[30])
                    and int(item["safe_attack"]) == bool(f[31])
                    and item["available_safe_attack"] == str(available)
                    and int(item["crate_count"]) == f[19]
                    and float(item["wasteful_penalty"])
                    == row["reward_components"].get("WASTEFUL_BOMB_PLACED", 0),
                    "Compact attack/reward mismatch",
                )
                if available:
                    gap = (
                        max(
                            q
                            for q, legal in zip(row["q_values"], row["legal_mask"], strict=True)
                            if legal
                        )
                        - row["q_values"][5]
                    )
                    require(float(item["bomb_q_gap"]) == gap, "Compact Q gap mismatch")
                else:
                    require(item["bomb_q_gap"] == "", "Unexpected Q gap")
            trajectories[name] = rows
            episodes.append(
                {
                    "run": name,
                    "opponent": job["opponent"],
                    **{k: v for k, v in metrics.items() if not isinstance(v, list)},
                    "stall_windows": len(stalls),
                    "eliminations": sum(r["events"].count("KILLED_OPPONENT") for r in rows),
                }
            )
    require(
        seen == expected and len(compact) == sum(e["steps"] for e in episodes),
        "Incomplete retained matrix",
    )
    examples = json.loads((record / "attack-example.json").read_text())
    require(
        examples == [trajectories["r3-1501101"][i - 1] for i in (173, 177)],
        "Public-state examples differ",
    )
    distances = [d for j in saved["jobs"] for d in j["metrics"]["opponent_manhattan_distances"]]
    counts = {k: sum(e[k] for e in episodes) for k in episodes[0] if k not in {"run", "opponent"}}
    return {
        "scope": config["scope"],
        "verified": True,
        "new_games": 0,
        "episodes": episodes,
        "totals": counts,
        "episodes_with_stalls": sum(e["stall_windows"] > 0 for e in episodes),
        "opponent_distance": {
            "minimum": min(distances),
            "median": statistics.median(distances),
            "maximum": max(distances),
        },
        "resource_accounting": "Aggregate CPU/RSS invalid; no reconstructed totals claimed",
        "checkpoint_selection": None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--record", type=Path, default=Path("experiments/2026-09-12-task3-hunting-diagnosis")
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.record)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)
