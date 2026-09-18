"""Does the agent stall once the crates are gone?

Watching it play, Julius noticed it paces between two tiles far more often late in
a round, when the board has been cleared of crates, and rarely while crates are
still standing. That is a testable claim, and it matters: a stalled agent is
collecting nothing and killing nobody for the rest of the round.

  play    run real games and record, for every step, where the agent stood, how
          many crates were left, how many coins were visible and whether any
          opponent was alive
  report  give the stall rate conditioned on crates remaining

A step counts as stalled when the agent's last WINDOW positions cover at most
MAX_DISTINCT tiles, the same predicate the loop guard uses. Nothing is patched in
behaviour: one framework method is wrapped only to observe.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from collections import defaultdict, deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "experiments/2026-09-17-task4-final-training/config.json"
WINDOW = 8
MAX_DISTINCT = 3


def stalled(positions) -> bool:
    """The loop guard's predicate: WINDOW positions covering at most MAX_DISTINCT tiles."""
    recent = list(positions)[-WINDOW:]
    return len(recent) == WINDOW and len(set(recent)) <= MAX_DISTINCT


def play(args) -> int:
    sys.path.insert(0, str(REPO_ROOT))
    import main  # noqa: E402
    from environment import BombeRLeWorld, GenericWorld  # noqa: E402

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    suite = cfg["evaluation_suites"][args.suite]
    seeds = list(suite["world_seeds"])[: args.worlds]

    agent_dir = REPO_ROOT / "agent_code" / args.agent
    staged = agent_dir / f"stall-{uuid.uuid4().hex[:8]}.pt"
    shutil.copyfile(args.checkpoint, staged)
    import os
    os.environ["BOMBERMAN_EVALUATION_CHECKPOINT"] = staged.name

    steps: list[dict] = []
    current = {"seed": None}
    original_poll = BombeRLeWorld.poll_and_run_agents

    def poll(self):
        for agent in self.active_agents:
            if not agent.name.startswith(args.agent):
                continue
            crates = int((self.arena == 1).sum())
            steps.append({
                "seed": current["seed"],
                "step": int(self.step),
                "position": (int(agent.x), int(agent.y)),
                "crates_left": crates,
                "coins_visible": sum(1 for coin in self.coins if coin.collectable),
                "opponents_alive": len(self.active_agents) - 1,
            })
        return original_poll(self)

    BombeRLeWorld.poll_and_run_agents = poll
    agents = [args.agent, *suite["opponents"]]
    try:
        for seed in seeds:
            current["seed"] = seed
            main.main([
                "play", "--agents", *agents, "--scenario", suite["scenario"],
                "--n-rounds", "1", "--seed", str(seed), "--no-gui",
            ])
    finally:
        BombeRLeWorld.poll_and_run_agents = original_poll
        staged.unlink(missing_ok=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(steps) + "\n", encoding="utf-8")
    print(f"recorded {len(steps)} steps over {len(seeds)} games -> {args.out}")
    return 0


def report(args) -> int:
    steps = []
    for path in args.recordings:
        steps.extend(json.loads(Path(path).read_text(encoding="utf-8")))

    by_game = defaultdict(list)
    for step in steps:
        by_game[step["seed"]].append(step)

    buckets = defaultdict(lambda: {"steps": 0, "stalled": 0})
    for game in by_game.values():
        game.sort(key=lambda s: s["step"])
        positions = deque(maxlen=WINDOW)
        for step in game:
            positions.append(tuple(step["position"]))
            bucket = "no crates left" if step["crates_left"] == 0 else "crates remaining"
            buckets[bucket]["steps"] += 1
            buckets[bucket]["stalled"] += int(stalled(positions))

    print(f"{len(by_game)} games, {len(steps)} agent steps\n")
    print(f"{'board state':<20}{'steps':>10}{'stalled':>10}{'rate':>9}")
    for name in ("crates remaining", "no crates left"):
        entry = buckets[name]
        rate = entry["stalled"] / entry["steps"] if entry["steps"] else float("nan")
        print(f"{name:<20}{entry['steps']:>10}{entry['stalled']:>10}{rate:>9.3f}")

    with_crates = buckets["crates remaining"]
    without = buckets["no crates left"]
    if with_crates["steps"] and without["steps"]:
        a = with_crates["stalled"] / with_crates["steps"]
        b = without["stalled"] / without["steps"]
        print(f"\nstalling is {b / a:.1f}x more likely once the crates are gone"
              if a else "\nno stalling at all while crates remain")

    # What is on the board when it stalls?
    stalls = defaultdict(int)
    for game in by_game.values():
        positions = deque(maxlen=WINDOW)
        for step in game:
            positions.append(tuple(step["position"]))
            if not stalled(positions):
                continue
            key = (step["crates_left"] == 0, step["coins_visible"] == 0, step["opponents_alive"] > 0)
            stalls[key] += 1
    print("\nwhat the board looked like during stalled steps")
    print(f"{'crates gone':<13}{'no coins visible':<19}{'opponent alive':<16}{'steps':>8}")
    for (no_crates, no_coins, opponent), count in sorted(stalls.items(), key=lambda kv: -kv[1]):
        print(f"{str(no_crates):<13}{str(no_coins):<19}{str(opponent):<16}{count:>8}")
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
