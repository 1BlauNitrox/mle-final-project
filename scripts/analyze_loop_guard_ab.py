"""Decide whether the loop guard ships, by the rule we registered before playing.

We compare the guarded and unguarded agent on the same checkpoints and the same
worlds. The decision comes only from the registered classic world set and the
registered rule. The coin-heaven comparison is reported next to it but never
changes it.

The script refuses to decide when a game is missing, when a game was played by
the wrong agent or on the wrong world set, or when the agent code on disk no
longer matches the hashes in the registration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
METRICS = ("score", "self_kills", "survived", "coins", "kills", "invalid")


def load_games(directory: Path) -> list[dict]:
    path = directory / "games.jsonl"
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def matrix(games: list[dict], artifacts: list[str], seeds: list[int], metric: str) -> np.ndarray:
    index = {(g["artifact"], g["world_seed"]): g[metric] for g in games}
    missing = [(a, s) for a in artifacts for s in seeds if (a, s) not in index]
    if missing:
        raise ValueError(f"{len(missing)} games missing, for example {missing[:3]}")
    return np.array([[index[(a, s)] for s in seeds] for a in artifacts], dtype=float)


def hierarchical_interval(diff: np.ndarray, resamples: int, seed: int, percent: float) -> dict:
    rng = np.random.default_rng(seed)
    checkpoints, worlds = diff.shape
    estimates = np.empty(resamples)
    for i in range(resamples):
        rows = rng.integers(0, checkpoints, checkpoints)
        cols = rng.integers(0, worlds, worlds)
        estimates[i] = diff[np.ix_(rows, cols)].mean()
    tail = (100 - percent) / 2
    return {
        "mean_difference": float(diff.mean()),
        "low": float(np.percentile(estimates, tail)),
        "high": float(np.percentile(estimates, 100 - tail)),
    }


def decide(score: dict, self_kills: dict, rule: dict) -> tuple[bool, list[str]]:
    t = rule["thresholds"]
    checks = [
        (
            score["low"] > t["score_lower_bound_above"],
            f"score lower bound {score['low']:+.3f} is above {t['score_lower_bound_above']:+.2f}",
        ),
        (
            score["mean_difference"] >= t["score_point_estimate_at_least"],
            f"score point estimate {score['mean_difference']:+.3f} is at least "
            f"{t['score_point_estimate_at_least']:+.2f}",
        ),
        (
            self_kills["high"] <= t["self_kills_upper_bound_at_most"],
            f"self-kill upper bound {self_kills['high']:+.3f} is at most "
            f"{t['self_kills_upper_bound_at_most']:+.2f}",
        ),
    ]
    return all(ok for ok, _ in checks), [("PASS " if ok else "FAIL ") + text for ok, text in checks]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--unguarded", type=Path, required=True)
    parser.add_argument("--guarded", type=Path, required=True)
    parser.add_argument("--coin-heaven-unguarded", type=Path)
    parser.add_argument("--coin-heaven-guarded", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    reg = json.loads(args.registration.read_text(encoding="utf-8"))
    rule = reg["decision_rule"]
    interval = rule["interval"]
    episode = reg["checkpoints"]["episode"]
    artifacts = [f"{job}@{episode}" for job in reg["checkpoints"]["judged"]]
    seeds = reg["suite"]["world_seeds"]
    agents = reg["compared_agents"]

    code = {
        "unguarded_callbacks": sha256(REPO_ROOT / "agent_code/Bomb-omb/callbacks.py")
        == agents["unguarded"]["callbacks_sha256"],
        "guarded_callbacks": sha256(REPO_ROOT / "agent_code/Bomb-omb-loopguard/callbacks.py")
        == agents["guarded"]["callbacks_sha256"],
        "guarded_loop_guard": sha256(REPO_ROOT / "agent_code/Bomb-omb-loopguard/loop_guard.py")
        == agents["guarded"]["loop_guard_sha256"],
    }
    if not all(code.values()):
        raise SystemExit(f"agent code differs from the registration: {code}")

    unguarded, guarded = load_games(args.unguarded), load_games(args.guarded)
    for games, expected in (
        (unguarded, agents["unguarded"]["agent"]),
        (guarded, agents["guarded"]["agent"]),
    ):
        wrong = {(g.get("agent"), g.get("suite")) for g in games} - {
            (expected, reg["suite"]["name"])
        }
        if wrong:
            raise SystemExit(f"games from the wrong agent or world set: {sorted(map(str, wrong))}")

    results = {}
    for metric in METRICS:
        diff = matrix(guarded, artifacts, seeds, metric) - matrix(
            unguarded, artifacts, seeds, metric
        )
        results[metric] = hierarchical_interval(
            diff, interval["resamples"], interval["seed"], interval["percent"]
        )
    ships, checks = decide(results["score"], results["self_kills"], rule)

    per_checkpoint = {}
    for artifact in artifacts:
        rows = {
            m: (
                matrix(unguarded, [artifact], seeds, m).mean(),
                matrix(guarded, [artifact], seeds, m).mean(),
            )
            for m in ("score", "self_kills", "coins")
        }
        per_checkpoint[artifact] = {
            m: {"unguarded": round(u, 3), "guarded": round(g, 3)} for m, (u, g) in rows.items()
        }

    secondary = None
    if args.coin_heaven_unguarded and args.coin_heaven_guarded:
        cu, cg = load_games(args.coin_heaven_unguarded), load_games(args.coin_heaven_guarded)
        worlds = sorted({g["world_seed"] for g in cu})
        secondary = {
            artifact: {
                "coins_unguarded": round(matrix(cu, [artifact], worlds, "coins").mean(), 2),
                "coins_guarded": round(matrix(cg, [artifact], worlds, "coins").mean(), 2),
                "self_kills_unguarded": round(
                    matrix(cu, [artifact], worlds, "self_kills").mean(), 3
                ),
                "self_kills_guarded": round(matrix(cg, [artifact], worlds, "self_kills").mean(), 3),
            }
            for artifact in artifacts
        }

    report = {
        "registration": reg["name"],
        "decision": "SHIP the guard" if ships else "DO NOT SHIP the guard in this form",
        "checks": checks,
        "guarded_minus_unguarded": {
            m: {k: round(v, 4) for k, v in r.items()} for m, r in results.items()
        },
        "per_checkpoint": per_checkpoint,
        "coin_heaven_descriptive": secondary,
        "games_per_agent": len(artifacts) * len(seeds),
        "agent_code_matches_registration": code,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")

    print(f"DECISION: {report['decision']}")
    for line in checks:
        print("  " + line)
    print("guarded minus unguarded (95% hierarchical interval):")
    for metric, r in results.items():
        print(f"  {metric:<11}{r['mean_difference']:+8.3f}  [{r['low']:+.3f}, {r['high']:+.3f}]")
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
