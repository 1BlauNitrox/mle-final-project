"""Replay the two phase-C invalid movements without changing the policy or framework."""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.probe_task3_exploration import INPUT, sha, worker  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    if sha(checkpoint) != INPUT:
        raise ValueError("Wrong immutable input")
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(source))
    os.chdir(source)
    from agent_code.DagobertDuckDQNTask3.config import ACTIONS
    from agent_code.DagobertDuckDQNTask3.legality import framework_legal_action_mask
    from environment import BombeRLeWorld

    original = BombeRLeWorld.perform_agent_action
    started, cpu = time.monotonic(), time.process_time()
    rows = []
    seed = None

    def record(world, agent, action):
        if time.monotonic() - started > 120 or time.process_time() - cpu > 120:
            raise RuntimeError("Two-replay diagnostic allocation exhausted")
        if agent.name != "DagobertDuckDQNTask3":
            return original(world, agent, action)
        state = agent.last_game_state
        moves = {"UP": (0, -1), "RIGHT": (1, 0), "DOWN": (0, 1), "LEFT": (-1, 0)}
        delta = moves.get(action, (0, 0))
        destination = (int(agent.x + delta[0]), int(agent.y + delta[1]))
        before = {a.name: [int(a.x), int(a.y)] for a in world.active_agents}
        legal = bool(framework_legal_action_mask(state)[ACTIONS.index(action)])
        original(world, agent, action)
        if "INVALID_ACTION" in agent.events:
            rows.append(
                {
                    "world_seed": seed,
                    "step": world.step,
                    "action": action,
                    "snapshot_legal": legal,
                    "destination": destination,
                    "snapshot_self": list(map(int, state["self"][3])),
                    "snapshot_others": {a[0]: list(map(int, a[3])) for a in state["others"]},
                    "positions_before_execution": before,
                    "occupied_by_other_at_execution": any(
                        name != agent.name and tuple(pos) == destination
                        for name, pos in before.items()
                    ),
                }
            )

    with patch.object(BombeRLeWorld, "perform_agent_action", record):
        for seed in (1682118, 1682119):
            worker(source, checkpoint, seed, 0.0, output / str(seed), phase="C")
    result = {
        "scope": "technical_replay_of_both_observed_phase_C_invalid_events",
        "input_sha256": INPUT,
        "events": rows,
        "cpu_seconds": time.process_time() - cpu,
        "wall_seconds": time.monotonic() - started,
        "checkpoints_unchanged": sha(checkpoint) == INPUT,
    }
    (output / "collision-trace.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
