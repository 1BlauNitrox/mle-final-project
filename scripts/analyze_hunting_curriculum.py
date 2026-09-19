"""Recompute registered paired gates from portable issue217 observations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from scripts.curriculum_io import read, require, sha, write


def metric(row, key):
    native = row["native"]
    if key == "collection_fraction":
        require(native["initially_available_coins"] > 0, "Undefined collection fraction")
        return native["coins"] / native["initially_available_coins"]
    return float(native[key])


def rows(directory, suite):
    return read(directory / f"{suite}.json")["rows"]


def difference(candidate, reference, suite, key):
    a, b = rows(candidate, suite), rows(reference, suite)
    require(
        [(r["world_seed"], r["slot"], r["opponents"]) for r in a]
        == [(r["world_seed"], r["slot"], r["opponents"]) for r in b],
        "Unpaired evaluation",
    )
    return np.array([metric(x, key) - metric(y, key) for x, y in zip(a, b, strict=True)])


def safety_gate(candidate, reference, limits):
    effects = {
        f"{suite}.{key}": float(difference(candidate, reference, suite, key).mean())
        for suite, key in [
            ("classic", "score"),
            ("classic", "survived"),
            ("classic", "self_kills"),
            ("coins", "collection_fraction"),
            ("crates", "collection_fraction"),
            ("crates", "self_kills"),
        ]
    }
    gates = {
        "score": effects["classic.score"] >= -limits["score_loss"],
        "survival": effects["classic.survived"] >= -limits["survival_loss"],
        "self_kills": effects["classic.self_kills"] <= limits["self_kills_increase"],
        "coin_retention": effects["coins.collection_fraction"] >= -limits["collection_loss"],
        "crate_retention": effects["crates.collection_fraction"] >= -limits["collection_loss"],
        "crate_self_kills": effects["crates.self_kills"] <= limits["self_kills_increase"],
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "effects": effects,
        "interpretation": "Exploratory point estimates; no significance claim",
    }


def interval(matrix, seed, samples):
    values = np.asarray(matrix)
    require(values.ndim == 2 and values.shape[0] == 3, "Three paired replicas required")
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(samples):
        replicas = rng.integers(0, 3, 3)
        worlds = rng.integers(0, values.shape[1], values.shape[1])
        draws.append(values[np.ix_(replicas, worlds)].mean())
    return {
        "difference": float(values.mean()),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "replica_differences": values.mean(axis=1).tolist(),
    }


def latency_gate(directory, limits):
    times = [t for row in rows(directory, "latency") for t in row["native"]["decision_times_ms"]]
    require(bool(times), "No latency observations")
    p95, maximum = float(np.quantile(times, 0.95)), max(times)
    return {
        "p95_ms": p95,
        "max_ms": maximum,
        "passed": p95 < limits["latency_p95_ms"] and maximum < limits["latency_max_ms"],
    }


def analyze(roots):
    cfg = read(roots[0] / "config.json")
    require(
        all(sha(r / "config.json") == sha(roots[0] / "config.json") for r in roots),
        "Different protocols",
    )
    pairs = {}
    for root in roots:
        for pair in (root / "pairs").glob("r*"):
            require(pair.name not in pairs, "Duplicate replica")
            pairs[pair.name] = pair
    require(set(pairs) == {"r1", "r2", "r3"}, "All three replicas required")
    decisions = {name: read(path / "decision.json") for name, path in pairs.items()}
    if any(d["status"] != "training_complete" for d in decisions.values()):
        return {
            "complete": False,
            "eligible": False,
            "submission_promotion": False,
            "decisions": decisions,
            "reason": "At least one registered pair stopped",
        }
    data = {}
    for name, path in pairs.items():
        for arm in cfg["arms"]:
            snapshot = path / arm / "snapshots/final"
            state = read(snapshot / "state.json")
            require(state["updates"] == cfg["updates"], "Incomplete update endpoint")
            for suite, setting in cfg["evaluation"]["final"].items():
                evidence = read(snapshot / "evaluation" / f"{suite}.json")
                require(
                    evidence["checkpoint_sha256"] == sha(snapshot / "checkpoint.pt"),
                    "Wrong checkpoint",
                )
                observed = evidence["rows"]
                require(
                    [r["world_seed"] for r in observed]
                    == list(range(*(setting["seeds"][0], setting["seeds"][1] + 1))),
                    "Wrong evaluation seeds",
                )
                require(
                    all(
                        r["epsilon"] == 0
                        and r["opponents"] == setting["opponents"]
                        and r["scenario"] == setting["scenario"]
                        and r["slot"] == i % (len(setting["opponents"]) + 1)
                        for i, r in enumerate(observed)
                    ),
                    "Wrong evaluation conditions",
                )
            data[name, arm] = snapshot / "evaluation"
        data[name, "reference"] = path / "reference/final"
        for suite in cfg["evaluation"]["final"]:
            require(
                read(data[name, "reference"] / f"{suite}.json")["checkpoint_sha256"]
                == cfg["reference_sha256"],
                "Wrong reference",
            )
    contrasts = {}
    gates = {}
    limits = cfg["screen"]
    for baseline in ("control", "reference"):
        contrasts[baseline] = {}
        for suite in cfg["evaluation"]["final"]:
            if suite == "latency":
                continue
            contrasts[baseline][suite] = {}
            for key in (
                "score",
                "kills",
                "survived",
                "self_kills",
                "coins",
                "invalid",
                "collection_fraction",
            ):
                values = [
                    difference(data[f"r{r}", "curriculum"], data[f"r{r}", baseline], suite, key)
                    for r in range(1, 4)
                ]
                contrasts[baseline][suite][key] = interval(
                    values, cfg["bootstrap_seed"], cfg["bootstrap_resamples"]
                )
        effects = contrasts[baseline]
        gates[baseline] = {
            "score": effects["classic"]["score"]["difference"] >= limits["score_gain"],
            "kills": effects["classic"]["kills"]["difference"] >= limits["kills_gain"],
            "survival": effects["classic"]["survived"]["difference"] >= -limits["survival_loss"],
            "self_kills": effects["classic"]["self_kills"]["difference"]
            <= limits["self_kills_increase"],
            "coins": effects["coins"]["collection_fraction"]["difference"]
            >= -limits["collection_loss"],
            "crates": effects["crates"]["collection_fraction"]["difference"]
            >= -limits["collection_loss"],
            "crate_self_kills": effects["crates"]["self_kills"]["difference"]
            <= limits["self_kills_increase"],
            "replicas": sum(x > 0 for x in effects["classic"]["kills"]["replica_differences"]) >= 2,
        }
    latency = {name: latency_gate(data[name, "curriculum"], limits) for name in pairs}
    eligible = all(all(g.values()) for g in gates.values()) and all(
        x["passed"] for x in latency.values()
    )
    return {
        "complete": True,
        "eligible": eligible,
        "submission_promotion": False,
        "candidate_for_confirmation": "curriculum-r1" if eligible else None,
        "gates": gates,
        "contrasts": contrasts,
        "latency": latency,
        "interpretation": "Exploratory screen; fresh confirmation and compatibility required",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write(args.output, analyze(args.roots))


if __name__ == "__main__":
    main()
