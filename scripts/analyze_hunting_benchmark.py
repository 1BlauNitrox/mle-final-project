"""Analyze Issue #205 without treating rotated games as independent worlds."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = ("score", "kills", "coins", "survived", "self_kills", "invalid")
ANALYSIS_METRICS = (*METRICS, "first_place_share")
LEARNED = {"A": "benchmark_fallback", "B": "RUEHL_BASED_AGENT"}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def round_agents(path: Path) -> dict:
    rounds = list(load_json(path)["by_round"].values())
    require(len(rounds) == 1, f"Expected one round: {path}")
    return rounds[0]["agents"]


def game_observations(path: Path, lineup: str, seed: int, rotation: int) -> list[dict]:
    agents = round_agents(path)
    top_score = max(a["score"] for a in agents.values())
    leaders = sum(a["score"] == top_score for a in agents.values())
    rows = []
    for name, data in agents.items():
        identity = name if not name.startswith("rule_based_agent") else "rule_based_agent"
        rows.append(
            {
                "lineup": lineup,
                "world_seed": seed,
                "rotation": rotation,
                "agent_name": name,
                "identity": identity,
                "score": data["score"],
                "kills": data["kills"],
                "coins": data["coins"],
                "survived": int(data["survived"]),
                "self_kills": data["self_kills"],
                "invalid": data["invalid"],
                "attempted_actions": data["attempted_actions"],
                "unique_win": int(data["score"] == top_score and leaders == 1),
                "score_tie": int(data["score"] == top_score and leaders > 1),
                "first_place_share": 1.0 / leaders if data["score"] == top_score else 0.0,
                "survival_steps": data["survival_steps"],
                "decision_time_median_ms": data["decision_time_median_ms"],
                "decision_time_p95_ms": data["decision_time_p95_ms"],
                "decision_time_max_ms": data["decision_time_max_ms"],
            }
        )
    return rows


def interval(world_values: np.ndarray, *, resamples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples)
    for index in range(resamples):
        draws[index] = np.mean(rng.choice(world_values, size=len(world_values), replace=True))
    return {
        "estimate": float(np.mean(world_values)),
        "ci95": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
    }


def clustered_difference(rows: list[dict], metric: str, config: dict) -> dict:
    by_cell = defaultdict(list)
    for row in rows:
        if row["lineup"] in LEARNED and row["identity"] == LEARNED[row["lineup"]]:
            by_cell[(row["world_seed"], row["lineup"])].append(float(row[metric]))
    differences = []
    for world in config["design"]["world_seeds"]:
        a, b = by_cell[(world, "A")], by_cell[(world, "B")]
        require(len(a) == len(b) == 4, f"Incomplete rotations for world {world}")
        differences.append(float(np.mean(b) - np.mean(a)))
    uncertainty = config["uncertainty"]
    return interval(
        np.asarray(differences), resamples=uncertainty["resamples"], seed=uncertainty["seed"]
    )


def descriptive(rows: list[dict], identity: str, lineup: str) -> dict:
    selected = [row for row in rows if row["lineup"] == lineup and row["identity"] == identity]
    return {
        metric: float(np.mean([row[metric] for row in selected]))
        for metric in (*METRICS, "unique_win", "score_tie", "first_place_share")
    }


def relative_to_rules(rows: list[dict], lineup: str, identity: str, config: dict) -> dict:
    games = defaultdict(list)
    for row in rows:
        if row["lineup"] == lineup:
            games[(row["world_seed"], row["rotation"])].append(row)
    output = {}
    uncertainty = config["uncertainty"]
    for metric in ANALYSIS_METRICS:
        by_world = defaultdict(list)
        for (world, _rotation), game in games.items():
            learned = next(row for row in game if row["identity"] == identity)
            rules = [row for row in game if row["identity"] == "rule_based_agent"]
            by_world[world].append(float(learned[metric] - np.mean([row[metric] for row in rules])))
        world_values = []
        for world in config["design"]["world_seeds"]:
            require(len(by_world[world]) == 4, f"Incomplete rotations for world {world}")
            world_values.append(float(np.mean(by_world[world])))
        output[metric] = interval(
            np.asarray(world_values),
            resamples=uncertainty["resamples"],
            seed=uncertainty["seed"],
        )
    return output


def timing_report(input_root: Path, config: dict) -> dict:
    limits = config["runtime"]["timing_limits_ms"]
    cells = {
        "fallback": ("A", "benchmark_fallback"),
        "external": ("B", "RUEHL_BASED_AGENT"),
    }
    report = {}
    timing_seed = config["design"]["serial_timing_seed"]
    for label, (lineup, identity) in cells.items():
        path = input_root / "timing" / f"{lineup}-s{timing_seed}-r0.json"
        require(path.is_file(), f"Missing registered timing game: {path}")
        agent = round_agents(path)[identity]
        p95 = float(agent["decision_time_p95_ms"])
        maximum = float(agent["decision_time_max_ms"])
        report[label] = {
            "decision_time_p95_ms": p95,
            "decision_time_maximum_ms": maximum,
            "p95_limit_ms": limits["decision_time_p95"],
            "maximum_limit_ms": limits["decision_time_maximum"],
            "passes": (
                p95 <= limits["decision_time_p95"] and maximum <= limits["decision_time_maximum"]
            ),
        }
    return report


def external_training_admitted(verdict: str, timing: dict) -> bool:
    return verdict == "appears stronger" and timing["external"]["passes"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_json(args.config)
    rows = []
    for seed in config["design"]["world_seeds"]:
        for rotation in range(4):
            for lineup in ("A", "B", "C"):
                path = args.input / "games" / f"{lineup}-s{seed}-r{rotation}.json"
                require(path.is_file(), f"Missing registered game: {path}")
                rows.extend(game_observations(path, lineup, seed, rotation))
    require(len(rows) == 960, "Expected 240 games and 960 agent observations")

    contrasts = {metric: clustered_difference(rows, metric, config) for metric in ANALYSIS_METRICS}
    score, kills, survival = contrasts["score"], contrasts["kills"], contrasts["survived"]
    if (
        score["estimate"] > 0
        and kills["estimate"] > 0
        and score["ci95"][0] > -0.15
        and survival["estimate"] >= -0.05
    ):
        verdict = "appears stronger"
    elif (
        score["estimate"] < 0
        and kills["estimate"] < 0
        and score["ci95"][1] < 0.15
        and survival["estimate"] <= 0.05
    ):
        verdict = "appears weaker"
    else:
        verdict = "inconclusive"

    timing = timing_report(args.input, config)
    result = {
        "protocol_issue": 205,
        "game_count": 240,
        "observation_count": len(rows),
        "verdict": verdict,
        "contrast_direction": "external minus fallback",
        "clustered_contrasts": contrasts,
        "lineup_A_fallback": descriptive(rows, "benchmark_fallback", "A"),
        "lineup_B_external": descriptive(rows, "RUEHL_BASED_AGENT", "B"),
        "fallback_minus_rule_mean": relative_to_rules(rows, "A", "benchmark_fallback", config),
        "external_minus_rule_mean": relative_to_rules(rows, "B", "RUEHL_BASED_AGENT", config),
        "lineup_C_fallback": descriptive(rows, "benchmark_fallback", "C"),
        "lineup_C_external": descriptive(rows, "RUEHL_BASED_AGENT", "C"),
        "serial_timing": timing,
        "external_training_admitted": external_training_admitted(verdict, timing),
        "observations": rows,
        "limitations": [
            "screening benchmark, not proof of tournament superiority",
            "rule_based_agent retains OS-entropy and Python-shuffle randomness",
            "confidence intervals resample 20 world clusters, retaining all rotations",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"VERDICT: external agent {verdict}")
    for metric in ANALYSIS_METRICS:
        item = contrasts[metric]
        print(f"{metric}: {item['estimate']:.4f} [{item['ci95'][0]:.4f}, {item['ci95'][1]:.4f}]")
    for identity in ("fallback", "external"):
        item = timing[identity]
        print(
            f"{identity} timing: p95={item['decision_time_p95_ms']:.3f} ms, "
            f"max={item['decision_time_maximum_ms']:.3f} ms, passes={item['passes']}"
        )
    print(f"EXTERNAL TRAINING ADMITTED: {result['external_training_admitted']}")


if __name__ == "__main__":
    main()
