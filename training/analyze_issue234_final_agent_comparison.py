"""Issue #234: analyse the head-to-head benchmark of the tabular agent and Bomb-omb.

The experimental unit is the world. For every world the per-game values are averaged over
its games (four seat rotations in the head-to-head suite), and 95% percentile bootstrap
intervals are computed by resampling worlds.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/2026-09-27-final-agent-comparison"
DQN = "Bomb-omb"
TABULAR = "DerKleineKonkurrenzvernichter"
METRICS = (
    "score",
    "coins",
    "kills",
    "self_kills",
    "survived",
    "strict_first",
    "invalid_actions",
    "placement",
)


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def value(agent: dict, metric: str) -> float:
    return float(agent[metric])


def rule_based_mean(game: dict, metric: str) -> float:
    values = [
        value(a, metric) for name, a in game["agents"].items() if name.startswith("rule_based")
    ]
    return float(np.mean(values))


def bootstrap_interval(per_world: np.ndarray, resamples: int = 10_000, seed: int = 234) -> dict:
    """Mean and 95% percentile bootstrap interval over worlds."""
    per_world = np.asarray(per_world, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_world), size=(resamples, len(per_world)))
    means = per_world[idx].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "mean": float(per_world.mean()),
        "low": float(low),
        "high": float(high),
        "worlds": int(len(per_world)),
    }


def per_world(rows: list[dict], suite: str, fn) -> np.ndarray:
    groups: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row["suite"] == suite:
            result = fn(row)
            if result is not None:
                groups[row["world_seed"]].append(result)
    return np.array([np.mean(groups[w]) for w in sorted(groups)])


def head_to_head(rows: list[dict], resamples: int, seed: int) -> dict:
    out = {"difference": {}, DQN: {}, TABULAR: {}, "rule_based_agent": {}}
    for metric in METRICS:
        out["difference"][metric] = bootstrap_interval(
            per_world(
                rows,
                "head_to_head",
                lambda r, m=metric: value(r["agents"][DQN], m) - value(r["agents"][TABULAR], m),
            ),
            resamples,
            seed,
        )
        for name in (DQN, TABULAR):
            out[name][metric] = bootstrap_interval(
                per_world(
                    rows, "head_to_head", lambda r, m=metric, n=name: value(r["agents"][n], m)
                ),
                resamples,
                seed,
            )
        out["rule_based_agent"][metric] = bootstrap_interval(
            per_world(rows, "head_to_head", lambda r, m=metric: rule_based_mean(r, m)),
            resamples,
            seed,
        )
    return out


def versus_rule_based(rows: list[dict], resamples: int, seed: int) -> dict:
    """Each agent alone against three rule-based agents on the same worlds."""
    games = defaultdict(dict)
    for row in rows:
        if row["suite"] != "versus_rule_based":
            continue
        learned = next(a for a in row["agents"] if not a.startswith("rule_based"))
        games[row["world_seed"]][learned] = row
    worlds = sorted(w for w, g in games.items() if DQN in g and TABULAR in g)
    out = {"difference": {}, "gap_to_rule_based": {}, DQN: {}, TABULAR: {}}
    for metric in METRICS:
        diff = [
            value(games[w][DQN]["agents"][DQN], metric)
            - value(games[w][TABULAR]["agents"][TABULAR], metric)
            for w in worlds
        ]
        out["difference"][metric] = bootstrap_interval(np.array(diff), resamples, seed)
        for name in (DQN, TABULAR):
            own = [value(games[w][name]["agents"][name], metric) for w in worlds]
            out[name][metric] = bootstrap_interval(np.array(own), resamples, seed)
    for name in (DQN, TABULAR):
        gap = [
            value(games[w][name]["agents"][name], "score")
            - rule_based_mean(games[w][name], "score")
            for w in worlds
        ]
        out["gap_to_rule_based"][name] = bootstrap_interval(np.array(gap), resamples, seed)
    return out


def decision_times(rows: list[dict]) -> dict:
    out = {}
    for name in (DQN, TABULAR):
        p95 = [r["agents"][name]["decision_time_p95_ms"] for r in rows if name in r["agents"]]
        mx = [r["agents"][name]["decision_time_max_ms"] for r in rows if name in r["agents"]]
        out[name] = {"median_of_p95_ms": float(np.median(p95)), "max_ms": float(np.max(mx))}
    return out


def analyse(rows: list[dict], config: dict) -> dict:
    resamples = config["analysis"]["bootstrap_resamples"]
    seed = config["analysis"]["bootstrap_seed"]
    h2h = head_to_head(rows, resamples, seed)
    primary = h2h["difference"]["score"]
    return {
        "games": {
            s: sum(1 for r in rows if r["suite"] == s)
            for s in ("head_to_head", "versus_rule_based")
        },
        "primary": {
            "metric": "head_to_head score difference Bomb-omb minus tabular",
            **primary,
            "success": primary["low"] > 0,
        },
        "head_to_head": h2h,
        "versus_rule_based": versus_rule_based(rows, resamples, seed),
        "decision_times": decision_times(rows),
    }


def plot(analysis: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    groups = [
        ("Head-to-head\n(same games)", analysis["head_to_head"]),
        ("Alone vs three\nrule_based_agents", analysis["versus_rule_based"]),
    ]
    labels = {
        DQN: "Bomb-omb (DQN)",
        TABULAR: "Tabular (r5)",
        "rule_based_agent": "rule_based_agent",
    }
    colors = {DQN: "#2b6cb0", TABULAR: "#dd8452", "rule_based_agent": "#8c8c8c"}
    for ax, metric, title in zip(
        axes,
        ("score", "kills", "survived"),
        ("Score per game", "Kills per game", "Survival rate"),
        strict=True,
    ):
        x = 0
        ticks, names = [], []
        for group_name, data in groups:
            members = [DQN, TABULAR] + (["rule_based_agent"] if "rule_based_agent" in data else [])
            for offset, name in enumerate(members):
                est = data[name][metric]
                ax.bar(
                    x + offset * 0.27,
                    est["mean"],
                    width=0.25,
                    color=colors[name],
                    label=labels[name] if x == 0 else None,
                )
                ax.errorbar(
                    x + offset * 0.27,
                    est["mean"],
                    yerr=[[est["mean"] - est["low"]], [est["high"] - est["mean"]]],
                    color="black",
                    capsize=3,
                    linewidth=1,
                )
            ticks.append(x + 0.27)
            names.append(group_name)
            x += 1.2
        ax.set_xticks(ticks, names, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    config = json.loads((EXPERIMENT / "config.json").read_text(encoding="utf-8"))
    rows = load_rows(EXPERIMENT / "results" / "per-game.jsonl")
    analysis = analyse(rows, config)
    (EXPERIMENT / "results" / "analysis.json").write_text(
        json.dumps(analysis, indent=2) + "\n", encoding="utf-8"
    )
    (EXPERIMENT / "figures").mkdir(exist_ok=True)
    plot(analysis, EXPERIMENT / "figures" / "comparison.png")
    print(json.dumps(analysis["primary"], indent=2))


if __name__ == "__main__":
    main()
