"""Bounded read-only development trajectories; never a performance/latency benchmark."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import json
import os
import subprocess
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import psutil
import torch

from training.run_task3_campaign import require, sha256
from training.seeded_framework import wrap_process_event

ROOT = Path(__file__).resolve().parents[1]
PROGRESS = {"COIN_COLLECTED", "CRATE_DESTROYED", "KILLED_OPPONENT"}


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def stall_windows(rows, width=24):
    """Conservative no-progress cycle: fixed board/coins, globally bomb-free, period <=4."""
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


def diagnose(*, agent, checkpoint, world_seed, agent_seed, opponent, output):
    """Record an immutable policy in an unchanged world; instrumentation stays training-only."""
    from agents import AgentRunner
    from environment import BombeRLeWorld, WorldArgs

    require(agent in {"DagobertDuckDQNTask2", "DagobertDuckDQNTask3"}, "Unsupported agent")
    require(opponent in {None, "peaceful_agent"}, "Unsupported diagnostic opponent")
    require(
        world_seed in {1460001, 1460002} and agent_seed == world_seed + 1000000,
        "Unregistered diagnostic seed",
    )
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = checkpoint.resolve()
    before = sha256(checkpoint)
    modules = {
        name: importlib.import_module(f"agent_code.{agent}.{name}")
        for name in ("callbacks", "features", "train", "config", "legality")
    }
    args = WorldArgs(
        True,
        30,
        False,
        0,
        False,
        None,
        False,
        True,
        str(output),
        str(output / "framework_stats.json"),
        "issue146-diagnostic",
        world_seed,
        False,
        "classic",
    )
    slots = {opponent: 1} if opponent else {}
    rows = []
    started = time.monotonic()
    process = psutil.Process()
    with (
        patch.dict(os.environ, {"BOMBERMAN_AGENT_SEED": str(agent_seed)}),
        patch.object(modules["callbacks"], "_evaluation_checkpoint_path", lambda: checkpoint),
        patch.object(
            AgentRunner,
            "process_event",
            wrap_process_event(AgentRunner.process_event, agent_seed, slots),
        ),
    ):
        world = BombeRLeWorld(args, [(agent, False)] + ([(opponent, False)] if opponent else []))
        learner = world.agents[0]
        policy = learner.backend.runner.fake_self
        world.new_round()
        while world.running:
            require(time.monotonic() - started < 120, "Diagnostic episode exceeded 120 seconds")
            require(process.memory_info().rss < 4 * 1024**3, "Diagnostic memory limit")
            alive = not learner.dead
            world.do_step()
            if not alive:
                continue
            state = learner.last_game_state
            after = world.get_state_for_agent(learner)
            features = modules["features"].state_to_features(
                state, include_continuation_features=policy.config.escape_continuation_features
            )
            normalized = modules["features"].normalize_features(features)
            with torch.no_grad():
                q = policy.policy_network(
                    torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)
                )[0].tolist()
            action = world.replay["actions"][learner.name][-1]
            events = list(learner.events)
            movement = (
                modules["train"]._coin_movement_event(state, after, action)
                if after is not None and world.running
                else None
            )
            bomb = modules["train"]._bomb_usefulness_event(state, events) if world.running else None
            shaped = events + [v for v in (movement, bomb) if v]
            signature = json.dumps(
                plain([state["field"], sorted(state["coins"])]), separators=(",", ":")
            )
            rows.append(
                plain(
                    {
                        "step": state["step"],
                        "position": state["self"][3],
                        "state": state,
                        "next_position": [learner.x, learner.y],
                        "action": action,
                        "events": events,
                        "features": features,
                        "normalized_features": normalized,
                        "q_values": q,
                        "legal_mask": modules["legality"].framework_legal_action_mask(state),
                        "reward_components": {
                            e: shaped.count(e) * modules["config"].REWARDS.get(e, 0)
                            for e in set(shaped)
                        },
                        "hazards": bool(state["bombs"] or np.any(state["explosion_map"])),
                        "board_coins_sha256": hashlib.sha256(signature.encode()).hexdigest(),
                    }
                )
            )
        world.end()
    require(sha256(checkpoint) == before, "Diagnostic altered checkpoint")
    with gzip.open(output / "trajectory.json.gz", "wt", encoding="utf-8") as file:
        json.dump(rows, file, separators=(",", ":"))
    windows = stall_windows(rows)
    feature_positions = {}
    for row in rows:
        feature_positions.setdefault(tuple(row["features"]), set()).add(tuple(row["position"]))
    summary = {
        "scope": "development_diagnosis_not_efficacy_or_latency",
        "agent": agent,
        "checkpoint_sha256": before,
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "source": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "world_seed": world_seed,
        "agent_seed": agent_seed,
        "opponent": opponent,
        "steps": len(rows),
        "stall_windows": windows,
        "aliased_feature_vectors": sum(len(p) > 1 for p in feature_positions.values()),
        "wall_seconds": time.monotonic() - started,
        "trajectory_sha256": sha256(output / "trajectory.json.gz"),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--world-seed", required=True, type=int)
    parser.add_argument("--agent-seed", required=True, type=int)
    parser.add_argument("--opponent", choices=["peaceful_agent"])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = diagnose(**vars(args))
    print(json.dumps({k: result[k] for k in ("steps", "aliased_feature_vectors", "wall_seconds")}))


if __name__ == "__main__":
    main()
