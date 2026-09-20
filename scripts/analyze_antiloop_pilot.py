"""Registered two-replica exploratory screen; never promote a submission."""

import argparse
from pathlib import Path

import numpy as np

from scripts.analyze_hunting_curriculum import (
    diagnostic_summary,
    difference,
    latency_gate,
    replay_summary,
    rows,
)
from scripts.analyze_hunting_curriculum import (
    safety_gate as retention_gate,
)
from scripts.curriculum_io import read, require, write


def safety_gate(candidate, reference, limits):
    report = retention_gate(candidate, reference, limits)
    invalid = float(difference(candidate, reference, "classic", "invalid").mean())
    report["effects"]["classic.invalid"] = invalid
    report["gates"]["invalid"] = invalid <= limits["invalid_increase"] + 1e-12
    for suite in ("coins", "crates"):
        report["gates"][suite + "_valid"] = (
            sum(r["native"]["invalid"] for r in rows(candidate, suite)) == 0
        )
    report["passed"] = all(report["gates"].values())
    return report


def interval(matrix, cfg):
    values = np.asarray(matrix)
    require(values.ndim == 2 and values.shape[0] == 2, "Two paired replicas required")
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    draws = []
    for _ in range(cfg["bootstrap_resamples"]):
        replicas = rng.integers(0, 2, 2)
        worlds = rng.integers(0, values.shape[1], values.shape[1])
        draws.append(values[np.ix_(replicas, worlds)].mean())
    return {
        "difference": float(values.mean()),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "replica_differences": values.mean(axis=1).tolist(),
    }


def loop_rate(observations):
    eligible = sum(row["loop_windows"]["eligible"] for row in observations)
    looping = sum(row["loop_windows"]["looping"] for row in observations)
    return {
        "eligible": eligible,
        "looping": looping,
        "rate": looping / eligible if eligible else None,
    }


def analyze(root):
    cfg = read(root / "config.json")
    report = {
        "complete": False,
        "eligible": False,
        "submission_promotion": False,
        "pairs": {},
        "training": {},
        "pilot": {},
        "contrasts": {},
        "gates": {},
    }
    data = {}
    for replica in (1, 2):
        directory = root / "pairs" / f"r{replica}"
        path = directory / "decision.json"
        decision = read(path) if path.exists() else {"status": "incomplete"}
        report["pairs"][str(replica)] = decision
        report["pilot"][str(replica)] = {p.stem: read(p) for p in directory.glob("gate-*.json")}
        for arm in cfg["arms"]:
            key = f"r{replica}-{arm}"
            report["training"][key] = {}
            for path in (directory / arm / "snapshots").glob("*/state.json"):
                state = read(path)
                report["training"][key][path.parent.name] = {
                    "episodes": state["episodes"],
                    "updates": state["updates"],
                    "diagnostics": diagnostic_summary(state["rows"]),
                    "replay": replay_summary(state["tracker"]),
                    "loop_penalties": sum(row["loop_penalties"] for row in state["rows"]),
                }
            data[replica, arm] = directory / arm / "snapshots/final/final-evaluation"
        data[replica, "reference"] = directory / "reference/final"
    if any(d["status"] != "training_complete" for d in report["pairs"].values()):
        return report
    for directory in data.values():
        if any(not (directory / f"{suite}.json").exists() for suite in cfg["evaluation"]["final"]):
            return report
    for directory in data.values():
        for suite, setting in cfg["evaluation"]["final"].items():
            observations = rows(directory, suite)
            expected_seeds = list(range(setting["seeds"][0], setting["seeds"][1] + 1))
            require(
                [r["world_seed"] for r in observations] == expected_seeds,
                "Evaluation does not match registered seeds/count",
            )
            require(
                all(
                    r["slot"] == i % (len(setting["opponents"]) + 1)
                    and r["opponents"] == setting["opponents"]
                    and r["epsilon"] == 0
                    and r["scenario"] == setting["scenario"]
                    for i, r in enumerate(observations)
                ),
                "Evaluation conditions changed",
            )
    report["complete"] = True
    limits = cfg["screen"]
    report["diagnostics"] = {
        f"r{replica}-{arm}": {
            suite: diagnostic_summary(rows(directory, suite))
            for suite in cfg["evaluation"]["final"]
            if suite != "latency"
        }
        for (replica, arm), directory in data.items()
    }
    for suite, setting in cfg["evaluation"]["final"].items():
        if suite == "latency":
            for replica in (1, 2):
                result = latency_gate(data[replica, "memory"], limits)
                report["contrasts"][f"latency-r{replica}"] = result
                report["gates"][f"latency-r{replica}"] = result["passed"]
            continue
        for baseline in ("control", "reference"):
            pairs = [(data[r, "memory"], data[r, baseline]) for r in (1, 2)]
            thresholds = {
                "score": (0, "min"),
                "kills": (0, "min"),
                "survived": (-limits["survival_loss"], "min"),
                "self_kills": (limits["self_kills_increase"], "max"),
            }
            if not setting["opponents"]:
                thresholds = {
                    "collection_fraction": (-limits["collection_loss"], "min"),
                    "self_kills": (limits["self_kills_increase"], "max"),
                }
            for metric, (threshold, mode) in thresholds.items():
                result = interval([difference(a, b, suite, metric) for a, b in pairs], cfg)
                key = f"{suite}.{baseline}.{metric}"
                report["contrasts"][key] = result
                report["gates"][key] = (
                    result["difference"] >= threshold - 1e-12
                    if mode == "min"
                    else result["difference"] <= threshold + 1e-12
                )
            if setting["opponents"]:
                effect = [float(difference(a, b, suite, "invalid").mean()) for a, b in pairs]
                valid = limits["invalid_actions"]
                key = f"{suite}.{baseline}.invalid"
                report["contrasts"][key] = effect
                report["gates"][key] = (
                    np.mean(effect) <= valid["pooled_increase_per_game"] + 1e-12
                    and max(effect) <= valid["replica_increase_per_game"] + 1e-12
                )
            else:
                report["gates"][f"{suite}.zero_invalid"] = all(
                    sum(r["native"]["invalid"] for r in rows(a, suite)) == 0 for a, _ in pairs
                )
            if suite == "classic":
                candidate = loop_rate([row for a, _ in pairs for row in rows(a, suite)])
                base = loop_rate([row for _, b in pairs for row in rows(b, suite)])
                key = f"loops.{baseline}"
                report["contrasts"][key] = {"candidate": candidate, "baseline": base}
                report["gates"][key] = bool(
                    candidate["eligible"]
                    and base["eligible"]
                    and base["rate"] > 0
                    and candidate["rate"] <= limits["loop_ratio"] * base["rate"]
                )
    report["gates"] = {key: bool(value) for key, value in report["gates"].items()}
    report["eligible"] = all(report["gates"].values())
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write(args.output, analyze(args.root))
