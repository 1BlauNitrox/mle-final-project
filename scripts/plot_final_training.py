"""Figures that document the final training run.

trajectory.png     what the run learned, in its own environment and on the
                   tournament suite, with the two benchmarks that matter
where-we-stand.png  the same agent against rule_based_agent, paired per world

The remaining PNGs expose the main panels as simple standalone plots for the
report and presentation.

Colours come from the validated categorical palette: slot 1 blue, slot 2 orange,
slot 3 aqua. Every series is direct-labelled as well as coloured, so identity is
never colour alone.
"""

from __future__ import annotations

import gzip
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "experiments/2026-09-17-task4-final-training/figures"
# The training logs stay on the run root; the evaluation evidence the figures
# need is committed next to them, so panels B to D redraw without the run.
RUN = Path(os.environ.get("TASK4_RUN_ROOT", r"C:\task4-final-training")) / "training-resume"
AGENT = "DagobertDuckDQNTask3"
BLOCK = 250
GREEDY_EPISODES_PER_BLOCK = 200

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRID = {"color": "#d8d7d2", "lw": 0.8}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": "#c9c8c3",
        "axes.labelcolor": MUTED,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


def training_blocks():
    per_job = {}
    for job in sorted(RUN.iterdir()):
        blocks = defaultdict(lambda: defaultdict(list))
        with gzip.open(job / "episodes.json.gz", "rt", encoding="utf-8") as handle:
            for row in json.load(handle):
                if row["behavior_epsilon"] != 0.0:
                    continue
                agent = row["native"]["agents"][AGENT]
                block = (row["completed_episodes"] - 1) // BLOCK
                for metric in ("score", "coins", "kills", "self_kills", "survived"):
                    blocks[block][metric].append(float(agent[metric]))
        per_job[job.name] = blocks
    return per_job


def complete_training_blocks(per_job):
    shared = set.intersection(*(set(blocks) for blocks in per_job.values()))
    return [
        block
        for block in sorted(shared)
        if all(
            len(blocks[block]["score"]) == GREEDY_EPISODES_PER_BLOCK for blocks in per_job.values()
        )
    ]


def monitoring():
    games = [
        json.loads(line)
        for line in (OUT / "monitoring-classic-100.jsonl")
        .read_text(encoding="utf-8-sig")
        .splitlines()
        if line.strip()
    ]
    by_episode = defaultdict(lambda: defaultdict(list))
    for row in games:
        key = 0 if row["artifact"] == "reference" else row["episode"]
        for metric in ("score", "coins", "kills", "self_kills", "survived", "invalid"):
            by_episode[key][metric].append(float(row[metric]))
    per_world = defaultdict(lambda: defaultdict(dict))
    for row in games:
        key = 0 if row["artifact"] == "reference" else row["episode"]
        per_world[key][row["world_seed"]].setdefault("rows", []).append(row)
    return by_episode, games


def rule_based():
    rows = [
        json.loads(line)
        for line in (OUT / "rule-based-seat.jsonl").read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    return {row["world_seed"]: row for row in rows}


def ci(values, rng, draws=4000):
    values = np.asarray(values, dtype=float)
    boots = np.array(
        [values[rng.integers(0, len(values), len(values))].mean() for _ in range(draws)]
    )
    return values.mean(), *np.percentile(boots, [2.5, 97.5])


def figure_trajectory(per_job, monitor, rule):
    episodes = [e for e in sorted(monitor) if e]
    rng = np.random.default_rng(20260918)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Final training run: six jobs, 24,000 episodes registered, stopped at ~9,600",
        fontsize=13,
        y=0.98,
        ha="center",
    )

    # A: the training environment
    ax = axes[0][0]
    complete = complete_training_blocks(per_job)
    x = [(b + 0.5) * BLOCK for b in complete]
    for name, blocks in per_job.items():
        series = [np.mean(blocks[b]["score"]) for b in complete]
        ax.plot(x, series, lw=1.0, alpha=0.30, color=BLUE if name.startswith("control") else ORANGE)
    pooled = [
        np.mean([np.mean(blocks[b]["score"]) for blocks in per_job.values()]) for b in complete
    ]
    ax.plot(x, pooled, lw=2.0, color=INK)
    ax.annotate(
        "pooled",
        (x[-1], pooled[-1]),
        xytext=(-4, 10),
        textcoords="offset points",
        color=INK,
        fontsize=9,
        ha="right",
    )
    ax.annotate("control, lr 2e-4", (x[len(x) // 2], 3.6), color=BLUE, fontsize=9)
    ax.annotate("halved, lr 1e-4", (x[len(x) // 2], 3.25), color=ORANGE, fontsize=9)
    ax.axvline(7000, color=MUTED, lw=1.0, ls=":")
    ax.annotate(
        "flat from here",
        (7000, 2.85),
        xytext=(6, 0),
        textcoords="offset points",
        color=MUTED,
        fontsize=9,
    )
    ax.set_title("A. Score in the training environment (soft line-up)", loc="left")
    ax.set_xlabel("training episode")
    ax.set_ylabel("score per episode")
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)

    # B: the tournament suite
    ax = axes[0][1]
    stats = [ci(monitor[e]["score"], rng) for e in episodes]
    means = [s[0] for s in stats]
    yerr = [
        [m - s[1] for m, s in zip(means, stats, strict=False)],
        [s[2] - m for m, s in zip(means, stats, strict=False)],
    ]
    ax.errorbar(
        episodes,
        means,
        yerr=yerr,
        marker="o",
        ms=7,
        lw=2.0,
        color=BLUE,
        capsize=4,
        ecolor=BLUE,
        elinewidth=1.4,
    )
    reference = np.mean(monitor[0]["score"])
    heuristic = np.mean([r["score"] for r in rule.values()])
    for value, label, colour, dy in (
        (reference, "untrained reference", MUTED, 6),
        (heuristic, "rule_based_agent", AQUA, -14),
    ):
        ax.axhline(value, color=colour, lw=1.6, ls="--")
        ax.annotate(
            f"{label}  {value:.2f}",
            (episodes[0], value),
            xytext=(0, dy),
            textcoords="offset points",
            color=colour,
            fontsize=9,
        )
    ax.set_ylim(reference - 0.10, heuristic + 0.10)
    ax.annotate(
        f"{means[-1]:.2f}",
        (episodes[-1], means[-1]),
        xytext=(8, -2),
        textcoords="offset points",
        color=BLUE,
        fontsize=9,
        fontweight="bold",
    )
    ax.set_title("B. Score on 100 fresh tournament worlds (95% intervals)", loc="left")
    ax.set_xlabel("milestone episode")
    ax.set_ylabel("score per game")
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)

    # C: where the score comes from
    ax = axes[1][0]
    labels, coins, killpoints = [], [], []
    labels.append("untrained")
    coins.append(np.mean(monitor[0]["coins"]))
    killpoints.append(5 * np.mean(monitor[0]["kills"]))
    for episode in episodes:
        labels.append(f"ep{episode // 1000}k")
        coins.append(np.mean(monitor[episode]["coins"]))
        killpoints.append(5 * np.mean(monitor[episode]["kills"]))
    labels.append("rule_based")
    coins.append(np.mean([r["coins"] for r in rule.values()]))
    killpoints.append(5 * np.mean([r["kills"] for r in rule.values()]))
    y = np.arange(len(labels))
    ax.barh(y, coins, height=0.62, color=BLUE, label="coins")
    ax.barh(
        y, killpoints, height=0.62, left=[c + 0.03 for c in coins], color=ORANGE, label="kills x 5"
    )
    for index, (c, k) in enumerate(zip(coins, killpoints, strict=False)):
        ax.annotate(f"{c + k:.2f}", (c + k + 0.12, index), va="center", fontsize=9, color=INK)
        ax.annotate(
            f"{c:.2f}", (c / 2, index), va="center", ha="center", fontsize=8.5, color="white"
        )
        ax.annotate(
            f"{k:.2f}",
            (c + 0.03 + k / 2, index),
            va="center",
            ha="center",
            fontsize=8.5,
            color="white",
        )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(np.array(coins) + np.array(killpoints)) + 0.55)
    ax.set_title("C. Score is coins plus five times kills", loc="left")
    ax.set_xlabel("score per game")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.01), ncols=2, fontsize=9)
    ax.grid(axis="x", **GRID)
    ax.set_axisbelow(True)

    # D: safety and survival
    ax = axes[1][1]
    for metric, colour, label in (
        ("self_kills", ORANGE, "self-kills"),
        ("survived", BLUE, "survived"),
    ):
        values = [np.mean(monitor[e][metric]) for e in episodes]
        ax.plot(episodes, values, marker="o", ms=7, lw=2.0, color=colour)
        ax.annotate(
            label,
            (episodes[-1], values[-1]),
            xytext=(8, -3),
            textcoords="offset points",
            color=colour,
            fontsize=9,
        )
        start = np.mean(monitor[0][metric])
        ax.plot([episodes[0] - 600], [start], marker="s", ms=7, color=colour, alpha=0.45)
        ax.annotate(
            f"untrained {start:.2f}",
            (episodes[0] - 600, start),
            xytext=(0, 9),
            textcoords="offset points",
            color=colour,
            fontsize=8.5,
            ha="left",
        )
    ax.set_title("D. Safety improved early, then stopped (rate per game)", loc="left")
    ax.set_xlabel("milestone episode")
    ax.set_ylabel("rate")
    ax.set_ylim(0, 0.8)
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)

    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.text(
        0.5,
        0.005,
        "Training-environment panel pools greedy episodes in blocks of 250. "
        "Tournament panels: 100 worlds, three rule_based opponents, "
        "one game per world per checkpoint.",
        ha="center",
        color=MUTED,
        fontsize=8.5,
    )
    path = OUT / "trajectory.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def figure_where_we_stand(games, rule):
    rng = np.random.default_rng(20260918)
    latest = {}
    for row in games:
        if row["artifact"] != "reference" and row["episode"] == 8000:
            latest.setdefault(row["world_seed"], []).append(row)
    worlds = sorted(w for w in latest if w in rule)

    # "higher" says which direction is in our favour, so colour can mean
    # better-or-worse rather than the raw sign of the difference.
    rows = [
        ("score", "score", True),
        ("coins", "coins", True),
        ("kills", "kills", True),
        ("self_kills", "self-kills (lower is better)", False),
        ("survived", "survived", True),
    ]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [3, 1]})
    fig.suptitle(
        "Episode-8,000 milestones minus rule_based_agent, paired on the same 100 worlds",
        fontsize=12,
        y=0.99,
    )

    labels, means, lows, highs, favourable = [], [], [], [], []
    for metric, label, higher in rows:
        diffs = [np.mean([r[metric] for r in latest[w]]) - rule[w][metric] for w in worlds]
        mean, low, high = ci(diffs, rng)
        labels.append(label)
        means.append(mean)
        lows.append(low)
        highs.append(high)
        favourable.append(mean >= 0 if higher else mean <= 0)

    y = np.arange(len(labels))
    ax.axvline(0, color=MUTED, lw=1.4)
    for index, (mean, low, high, good) in enumerate(
        zip(means, lows, highs, favourable, strict=False)
    ):
        clears = not (low <= 0 <= high)
        colour = BLUE if good else ORANGE
        ax.plot(
            [low, high], [index, index], lw=2.0, color=colour, alpha=0.85, solid_capstyle="round"
        )
        ax.plot(
            [mean],
            [index],
            marker="o",
            ms=9,
            color=colour,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
        )
        ax.annotate(
            f"{mean:+.3f}  [{low:+.2f}, {high:+.2f}]" + ("  *" if clears else ""),
            (max(high, 0.02), index),
            xytext=(10, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK,
        )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(-0.75, 1.35)
    ax.set_xlabel("difference per game   (blue = in our favour, orange = against us)")
    ax.set_title(
        "We win on coins, we lose on kills, and kills are worth five times more", loc="left"
    )
    ax.grid(axis="x", **GRID)
    ax.set_axisbelow(True)

    diffs = [np.mean([r["invalid"] for r in latest[w]]) - rule[w]["invalid"] for w in worlds]
    mean, low, high = ci(diffs, rng)
    ax2.axvline(0, color=MUTED, lw=1.4)
    ax2.plot([low, high], [0, 0], lw=2.0, color=BLUE, solid_capstyle="round")
    ax2.plot([mean], [0], marker="o", ms=9, color=BLUE, markeredgecolor=SURFACE, markeredgewidth=2)
    ax2.annotate(
        f"{mean:+.2f}",
        (mean, 0),
        xytext=(0, 12),
        textcoords="offset points",
        ha="center",
        fontsize=9,
        color=INK,
    )
    ax2.set_yticks([0], ["invalid actions (lower is better)"])
    ax2.set_xlabel("difference per game")
    ax2.set_title("And we are far cleaner", loc="left")
    ax2.grid(axis="x", **GRID)
    ax2.set_axisbelow(True)

    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.text(
        0.5,
        0.005,
        "Four rule_based_agents played the same 100 worlds; the one in our agent's "
        "starting corner is the comparison. Intervals are 95% bootstrap over worlds; "
        "* marks an interval that clears zero.",
        ha="center",
        color=MUTED,
        fontsize=8.5,
    )
    path = OUT / "where-we-stand.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def figure_training_score(per_job):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    complete = complete_training_blocks(per_job)
    x = [(b + 0.5) * BLOCK for b in complete]
    for name, blocks in per_job.items():
        values = [np.mean(blocks[b]["score"]) for b in complete]
        colour = BLUE if name.startswith("control") else ORANGE
        ax.plot(x, values, lw=1.0, alpha=0.32, color=colour)
    pooled = [
        np.mean([np.mean(blocks[b]["score"]) for blocks in per_job.values() if blocks[b]["score"]])
        for b in complete
    ]
    ax.plot(x, pooled, lw=2.2, color=INK, label="pooled")
    ax.axvline(7000, color=MUTED, lw=1.0, ls=":")
    ax.annotate(
        "curve flattens",
        (7000, pooled[min(len(pooled) - 1, 27)]),
        xytext=(7, 8),
        textcoords="offset points",
        color=MUTED,
    )
    ax.set(
        title="Score during final training",
        xlabel="training episode",
        ylabel="score per greedy episode",
    )
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)
    ax.legend()
    fig.tight_layout()
    path = OUT / "training-score.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def figure_tournament_score(monitor, rule):
    episodes = [e for e in sorted(monitor) if e]
    rng = np.random.default_rng(20260918)
    stats = [ci(monitor[e]["score"], rng) for e in episodes]
    means = [s[0] for s in stats]
    yerr = [
        [mean - stat[1] for mean, stat in zip(means, stats, strict=False)],
        [stat[2] - mean for mean, stat in zip(means, stats, strict=False)],
    ]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.errorbar(
        episodes,
        means,
        yerr=yerr,
        marker="o",
        ms=6,
        lw=2,
        color=BLUE,
        capsize=4,
        label="trained milestones",
    )
    reference = np.mean(monitor[0]["score"])
    heuristic = np.mean([row["score"] for row in rule.values()])
    ax.axhline(
        reference, color=MUTED, lw=1.5, ls="--", label=f"untrained reference ({reference:.2f})"
    )
    ax.axhline(heuristic, color=AQUA, lw=1.5, ls="--", label=f"rule_based_agent ({heuristic:.2f})")
    ax.set(
        title="Score on 100 fixed tournament worlds",
        xlabel="milestone episode",
        ylabel="score per game",
    )
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)
    ax.legend()
    fig.tight_layout()
    path = OUT / "tournament-score.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def figure_score_components(monitor, rule):
    episodes = [e for e in sorted(monitor) if e]
    labels = ["untrained", *[f"ep{e // 1000}k" for e in episodes], "rule_based"]
    coins = [np.mean(monitor[0]["coins"])]
    kills = [5 * np.mean(monitor[0]["kills"])]
    coins.extend(np.mean(monitor[e]["coins"]) for e in episodes)
    kills.extend(5 * np.mean(monitor[e]["kills"]) for e in episodes)
    coins.append(np.mean([row["coins"] for row in rule.values()]))
    kills.append(5 * np.mean([row["kills"] for row in rule.values()]))

    fig, ax = plt.subplots(figsize=(8, 5.2))
    y = np.arange(len(labels))
    ax.barh(y, coins, height=0.62, color=BLUE, label="coins")
    ax.barh(y, kills, height=0.62, left=coins, color=ORANGE, label="kills x 5")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set(title="Where the score comes from", xlabel="score per game")
    ax.grid(axis="x", **GRID)
    ax.set_axisbelow(True)
    ax.legend()
    fig.tight_layout()
    path = OUT / "score-components.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def figure_safety_survival(monitor):
    episodes = [e for e in sorted(monitor) if e]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for metric, colour, label in (
        ("self_kills", ORANGE, "self-kills"),
        ("survived", BLUE, "survived"),
    ):
        values = [np.mean(monitor[e][metric]) for e in episodes]
        ax.plot(episodes, values, marker="o", ms=6, lw=2, color=colour, label=label)
        start = np.mean(monitor[0][metric])
        ax.scatter([episodes[0] - 600], [start], marker="s", s=45, color=colour, alpha=0.55)
    ax.set(
        title="Safety improved early and then flattened",
        xlabel="milestone episode",
        ylabel="rate per game",
        ylim=(0, 0.8),
    )
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)
    ax.legend()
    fig.tight_layout()
    path = OUT / "safety-survival.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def figure_rule_based_comparison(games, rule):
    rng = np.random.default_rng(20260918)
    latest = defaultdict(list)
    for row in games:
        if row["artifact"] != "reference" and row["episode"] == 8000:
            latest[row["world_seed"]].append(row)
    worlds = sorted(world for world in latest if world in rule)
    metrics = [
        ("score", "score", True),
        ("coins", "coins", True),
        ("kills", "kills", True),
        ("self_kills", "self-kills", False),
        ("survived", "survived", True),
    ]
    labels, estimates = [], []
    for metric, label, higher_is_better in metrics:
        diffs = [
            np.mean([row[metric] for row in latest[world]]) - rule[world][metric]
            for world in worlds
        ]
        mean, low, high = ci(diffs, rng)
        labels.append(label)
        estimates.append((mean, low, high, mean >= 0 if higher_is_better else mean <= 0))

    fig, ax = plt.subplots(figsize=(8, 4.8))
    y = np.arange(len(labels))
    ax.axvline(0, color=MUTED, lw=1.3)
    for index, (mean, low, high, favourable) in enumerate(estimates):
        colour = BLUE if favourable else ORANGE
        ax.plot([low, high], [index, index], lw=2, color=colour)
        ax.plot(mean, index, marker="o", ms=8, color=colour)
        ax.annotate(
            f"{mean:+.3f}",
            (high, index),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set(
        title="Episode 8,000 minus rule_based_agent",
        xlabel="paired difference per game (95% bootstrap interval)",
    )
    ax.grid(axis="x", **GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = OUT / "rule-based-comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    per_job = training_blocks()
    monitor, games = monitoring()
    rule = rule_based()
    figure_trajectory(per_job, monitor, rule)
    figure_where_we_stand(games, rule)
    figure_training_score(per_job)
    figure_tournament_score(monitor, rule)
    figure_score_components(monitor, rule)
    figure_safety_survival(monitor)
    figure_rule_based_comparison(games, rule)


if __name__ == "__main__":
    main()
