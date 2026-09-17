"""Explain invalid actions and pacing by instrumenting the unmodified framework.

  play    Run real games through main.main() and record, for every step the
          agent takes: its action, whether the framework rejected it, why, how
          close the nearest opponent was, and whether any opponent was alive.
  report  Summarize one or more recordings: invalid-action causes, exposure,
          deaths shortly after an invalid action, reversal ("pacing") rates with
          and without opponents alive, and how often rounds hit the step limit.

Nothing is patched in behaviour. Two framework methods are wrapped only to
observe: poll_and_run_agents (to capture every position before anyone acts) and
perform_agent_action (to see each action's outcome in execution order).

Every agent observes the board before any agent acts, and actions then execute
one at a time in a shuffled order. So a move to a tile that was free when the
agent looked can still be rejected, if an opponent acting earlier moved there.
"""

from __future__ import annotations

import argparse
import collections
import importlib
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "experiments/2026-09-17-task4-final-training/config.json"
ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")
DELTA = {"UP": (0, -1), "RIGHT": (1, 0), "DOWN": (0, 1), "LEFT": (-1, 0)}
REVERSE = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}


def classify_invalid(*, action, allowed, start, occupant_start, occupant_acted_first,
                     bomb_on_target, arena_blocked):
    """Name the reason the framework rejected one action."""
    if not allowed:
        return "mask_disallowed_it"
    if action not in DELTA:
        return "non_move"
    if occupant_start is not None and occupant_acted_first and occupant_start != target_of(start, action):
        return "contested_tile_opponent_moved_in_first"
    if bomb_on_target:
        return "bomb_on_target"
    if arena_blocked:
        return "arena_blocked"
    return "unexplained"


def target_of(position, action):
    return position[0] + DELTA[action][0], position[1] + DELTA[action][1]


def reversals(actions):
    """Per step, whether that move exactly undoes the previous move."""
    return [
        index > 0 and actions[index] in REVERSE and REVERSE[actions[index]] == actions[index - 1]
        for index in range(len(actions))
    ]


def play(args) -> int:
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if args.suite not in cfg["evaluation_suites"] or "holdout" in args.suite:
        raise SystemExit(f"refusing suite {args.suite!r}: development suites only")
    suite = cfg["evaluation_suites"][args.suite]
    seeds = suite["world_seeds"][: args.worlds]

    agent_dir = REPO_ROOT / "agent_code" / args.agent
    staged = f"diag-{uuid.uuid4().hex[:8]}.pt"
    shutil.copyfile(args.checkpoint, agent_dir / staged)
    os.environ["BOMBERMAN_EVALUATION_CHECKPOINT"] = staged
    os.chdir(REPO_ROOT)
    sys.path.insert(0, str(REPO_ROOT))

    import events as e
    import main
    from environment import BombeRLeWorld, GenericWorld

    legality = importlib.import_module(f"agent_code.{args.agent}.legality")
    steps: list[dict] = []
    current = {"seed": None}
    original_poll = BombeRLeWorld.poll_and_run_agents
    original_perform = GenericWorld.perform_agent_action

    def poll(self):
        self._diag = {"start": {a.name: (int(a.x), int(a.y)) for a in self.active_agents}, "order": []}
        return original_poll(self)

    def perform(self, agent, action):
        diag = self._diag
        diag["order"].append(agent.name)
        before = len(agent.events)
        original_perform(self, agent, action)
        if not agent.name.startswith(args.agent):
            return
        invalid = e.INVALID_ACTION in agent.events[before:]
        observed = agent.last_game_state
        start = diag["start"][agent.name]
        others = observed.get("others", [])
        reason = None
        if invalid:
            mask = legality.framework_legal_action_mask(observed)
            target = target_of(start, action) if action in DELTA else None
            occupant = next(
                (a for a in self.active_agents if a is not agent and (a.x, a.y) == target), None
            ) if target else None
            reason = classify_invalid(
                action=action,
                allowed=bool(mask[ACTIONS.index(action)]) if action in ACTIONS else False,
                start=start,
                occupant_start=diag["start"].get(occupant.name) if occupant else None,
                occupant_acted_first=bool(occupant) and occupant.name in diag["order"][:-1],
                bomb_on_target=bool(target) and any((b.x, b.y) == target for b in self.bombs),
                arena_blocked=bool(target) and self.arena[target] != 0,
            )
        steps.append({
            "seed": current["seed"],
            "step": int(self.step),
            "action": action,
            "move": action in DELTA,
            "invalid": invalid,
            "reason": reason,
            "nearest_opponent": min(
                (abs(int(o[3][0]) - start[0]) + abs(int(o[3][1]) - start[1]) for o in others),
                default=None,
            ),
            "opponents_alive": len(others),
        })

    BombeRLeWorld.poll_and_run_agents = poll
    GenericWorld.perform_agent_action = perform
    agents = [args.agent, *suite["opponents"]]
    try:
        for seed in seeds:
            current["seed"] = seed
            main.main(["play", "--agents", *agents, "--scenario", suite["scenario"],
                       "--n-rounds", "1", "--seed", str(seed), "--no-gui"])
    finally:
        (agent_dir / staged).unlink(missing_ok=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    record = {"agent": args.agent, "checkpoint": str(args.checkpoint), "suite": args.suite,
              "seeds": seeds, "steps": steps}
    args.out.write_text(json.dumps(record) + "\n", encoding="utf-8")
    print(f"{len(seeds)} worlds, {len(steps)} steps -> {args.out}")
    return 0


def report(args) -> int:
    rows = []
    for path in args.recordings:
        rec = json.loads(Path(path).read_text(encoding="utf-8"))
        by_world = collections.defaultdict(list)
        for step in rec["steps"]:
            by_world[step["seed"]].append(step)
        steps = rec["steps"]
        games = len(by_world)
        invalid = [s for s in steps if s["invalid"]]
        moves = sum(s["move"] for s in steps)
        pace = {True: [0, 0], False: [0, 0]}
        fatal = 0
        for world in by_world.values():
            flags = reversals([s["action"] for s in world])
            last = world[-1]["step"]
            for flag, s in zip(flags, world):
                alone = s["opponents_alive"] == 0
                pace[alone][0] += 1
                pace[alone][1] += flag
                if s["invalid"] and last < 400 and last - s["step"] <= 5:
                    fatal += 1
        rows.append({
            "recording": Path(path).stem,
            "suite": rec["suite"],
            "games": games,
            "invalid_per_game": round(len(invalid) / games, 3),
            "invalid_reasons": dict(collections.Counter(s["reason"] for s in invalid)),
            "invalid_per_1000_moves": round(1000 * len(invalid) / max(moves, 1), 2),
            "moves_per_game": round(moves / games, 1),
            "share_steps_opponent_within_2": round(
                sum(1 for s in steps if s["nearest_opponent"] is not None and s["nearest_opponent"] <= 2)
                / max(len(steps), 1), 4),
            "invalid_followed_by_death_within_5": fatal,
            "reversal_share_opponents_alive": round(pace[False][1] / max(pace[False][0], 1), 4),
            "reversal_share_alone": round(pace[True][1] / pace[True][0], 4) if pace[True][0] else None,
            "mean_round_steps": round(mean(len(w) for w in by_world.values()), 1),
            "rounds_hitting_400": sum(len(w) >= 400 for w in by_world.values()),
        })
    for row in rows:
        print(json.dumps(row))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("play")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--agent", default="Bomb-omb")
    p.add_argument("--suite", default="classic-rule-based")
    p.add_argument("--worlds", type=int, default=20)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("report")
    r.add_argument("recordings", nargs="+")
    args = parser.parse_args()
    return play(args) if args.mode == "play" else report(args)


if __name__ == "__main__":
    raise SystemExit(main())
