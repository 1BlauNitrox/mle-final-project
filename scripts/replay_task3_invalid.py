"""Replay the fixed #171 development failure; observation-only, no policy edits."""

import argparse
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

from scripts.pilot_task3_frozen import play, sha, write


def replay(root, checkpoint, output):
    original_hash = sha(checkpoint)
    sys.path.insert(0, str(root / "source"))
    os.chdir(root / "source")
    from agent_code.DagobertDuckDQNTask3.legality import framework_legal_action_mask
    from environment import GenericWorld

    original = GenericWorld.perform_agent_action
    invalids = []

    def observed(world, agent, action):
        before = [(a.name, a.x, a.y) for a in world.active_agents]
        observation = agent.last_game_state
        original(world, agent, action)
        if agent.name == "DagobertDuckDQNTask3" and "INVALID_ACTION" in agent.events:
            invalids.append(
                {
                    "step": world.step,
                    "action": action,
                    "position": [agent.x, agent.y],
                    "execution_positions": before,
                    "observed_others": observation["others"],
                    "observed_legal_mask": framework_legal_action_mask(observation).tolist(),
                }
            )

    with patch.object(GenericWorld, "perform_agent_action", observed):
        row = play(
            root, checkpoint, 5931405, 6931405, "classic", ["peaceful_agent"], False, 0.0, output
        )
    if sha(checkpoint) != original_hash:
        raise ValueError("Replay changed checkpoint")
    write(
        output / "diagnosis.json",
        {
            "scope": "technical replay of retained failure",
            "checkpoint_sha256": original_hash,
            "world_seed": 5931405,
            "agent_seed": 6931405,
            "invalid_actions": json.loads(json.dumps(invalids, default=lambda v: v.item())),
            "observation": row,
        },
    )
    print(invalids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay(args.root.resolve(), args.checkpoint.resolve(), args.output.resolve())
