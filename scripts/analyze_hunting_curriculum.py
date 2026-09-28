"""Recompute registered paired gates from portable issue217 observations."""

from __future__ import annotations

import argparse
from collections import Counter
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


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def diagnostic_summary(observations):
    """Keep denominators and missing attribution explicit, including failed pilots."""
    count = len(observations)
    eligible = sum(r["loop_windows"]["eligible"] for r in observations)
    looping = sum(r["loop_windows"]["looping"] for r in observations)
    decisions = sum(len(r["native"]["decision_times_ms"]) for r in observations)
    attack = [r["attack_exposure"] for r in observations]
    used = [r for r in observations if r["attack_exposure"]["safe_attack_bombs"] > 0]
    attributed = [r for r in observations if "bomb_credits" in r]
    bomb_records = [b for r in attributed for b in r["bomb_credits"]]
    safe_bombs = [b for b in bomb_records if b["safe_attack"]]
    coverage = len(attributed) == count and count > 0
    for row in attributed:
        require(
            sum(b["kill_credits"] for b in row["bomb_credits"]) == row["native"]["kills"],
            "Bomb attribution disagrees with native kills",
        )
    reachable = sum(r["initial_exposure"]["reachable_attack_position"] for r in observations)
    reached = sum(a["safe_attack_steps"] > 0 for a in attack)
    safe_steps = sum(a["safe_attack_steps"] for a in attack)
    invalid = sum(r["native"]["invalid"] for r in observations)
    return {
        "episodes": count,
        "decisions": decisions,
        "survival_rate": ratio(sum(r["native"]["survived"] for r in observations), count),
        "loops": {
            "eligible_windows": eligible,
            "looping_windows": looping,
            "window_rate": ratio(looping, eligible),
            "episodes_with_eligible_windows": sum(
                r["loop_windows"]["eligible"] > 0 for r in observations
            ),
            "episodes_with_loops": sum(r["loop_windows"]["looping"] > 0 for r in observations),
            "note": "Overlapping windows are descriptive, not independent samples",
        },
        "attack": {
            "reachable_at_start_episodes": reachable,
            "reachable_at_start_fraction": ratio(reachable, count),
            "reached_safe_attack_episodes": reached,
            "reached_safe_attack_fraction": ratio(reached, count),
            "safe_attack_steps": safe_steps,
            "safe_attack_steps_per_decision": ratio(safe_steps, decisions),
            "threatening_steps": sum(a["threatening_steps"] for a in attack),
            "safe_attack_bombs": sum(a["safe_attack_bombs"] for a in attack),
            "episodes_using_safe_attack": len(used),
            "episode_kills_when_safe_attack_used": sum(r["native"]["kills"] for r in used),
            "native_kills": sum(r["native"]["kills"] for r in observations),
            "attribution_coverage_episodes": len(attributed),
            "safe_bomb_kill_credits": sum(b["kill_credits"] for b in safe_bombs)
            if coverage
            else None,
            "safe_bomb_self_kill_credits": sum(b["self_kill_credits"] for b in safe_bombs)
            if coverage
            else None,
            "safe_bombs_with_kills": sum(b["kill_credits"] > 0 for b in safe_bombs)
            if coverage
            else None,
            "note": "Reachability is initial-state geometry; episode-kill association "
            "is separate from native credits attributable to a tagged bomb",
        },
        "invalid": {
            "count": invalid,
            "per_game": ratio(invalid, count),
            "per_100_decisions": ratio(100 * invalid, decisions),
        },
    }


def replay_summary(tracker):
    counters = {
        "generated": tracker["generated"],
        "sampled": tracker["sampled"],
        "resident": dict(Counter(tracker["origins"])),
    }
    require(
        sum(counters["generated"].values()) == tracker["transitions"],
        "Replay generated-count mismatch",
    )
    result = {}
    for source, counts in counters.items():
        require(set(counts) <= {"parent", "classic", "hunting"}, "Unknown replay origin")
        require(
            all(isinstance(n, int) and n >= 0 for n in counts.values()),
            "Invalid replay origin counts",
        )
        total = sum(counts.values())
        result[source] = {
            "count": total,
            "counts": dict(counts),
            "fractions": {
                k: ratio(counts.get(k, 0), total) for k in ("parent", "classic", "hunting")
            },
        }
    return result


def training_diagnostics(pairs):
    result = {}
    for replica, directory in pairs.items():
        result[replica] = {}
        for arm in ("control", "curriculum"):
            snapshots = {}
            for path in sorted((directory / arm / "snapshots").glob("*/state.json")):
                state = read(path)
                require(state["episodes"] == len(state["rows"]), "Training episode-count mismatch")
                snapshots[path.parent.name] = {
                    "episodes": state["episodes"],
                    "updates": state["updates"],
                    "all": diagnostic_summary(state["rows"]),
                    "by_kind": {
                        kind: diagnostic_summary([r for r in state["rows"] if r["kind"] == kind])
                        for kind in ("classic", "hunting")
                    },
                    "replay": replay_summary(state["tracker"]),
                }
            result[replica][arm] = snapshots
    return result


def invalid_gates(data, cfg):
    """Behavioral regression tolerance, separate from runtime-only latency worlds."""
    limits = cfg["screen"]["invalid_actions"]
    report = {}
    for suite, setting in cfg["evaluation"]["final"].items():
        if suite == "latency":
            continue
        report[suite] = {}
        if not setting["opponents"]:
            counts = {
                f"r{r}": sum(
                    row["native"]["invalid"] for row in rows(data[f"r{r}", "curriculum"], suite)
                )
                for r in range(1, 4)
            }
            report[suite]["solo"] = {
                "counts": counts,
                "passed": all(n <= limits["solo_maximum_total"] for n in counts.values()),
            }
        else:
            for baseline in ("control", "reference"):
                diffs = [
                    float(
                        difference(
                            data[f"r{r}", "curriculum"], data[f"r{r}", baseline], suite, "invalid"
                        ).mean()
                    )
                    for r in range(1, 4)
                ]
                report[suite][baseline] = {
                    "pooled_increase": float(np.mean(diffs)),
                    "replica_increases": diffs,
                    "passed": np.mean(diffs) <= limits["pooled_increase_per_game"]
                    and all(d <= limits["replica_increase_per_game"] for d in diffs),
                }
    return {
        "passed": all(bool(c["passed"]) for s in report.values() for c in s.values()),
        "suites": report,
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
    require(set(pairs) <= {"r1", "r2", "r3"}, "Unexpected replica")
    decisions = {
        name: read(path / "decision.json")
        if (path / "decision.json").exists()
        else {"status": "incomplete"}
        for name, path in pairs.items()
    }
    training = training_diagnostics(pairs)
    if set(pairs) != {"r1", "r2", "r3"} or any(
        d["status"] != "training_complete" for d in decisions.values()
    ):
        return {
            "complete": False,
            "eligible": False,
            "submission_promotion": False,
            "decisions": decisions,
            "training_diagnostics": training,
            "reason": "All three complete pairs required; missing/stopped evidence retained",
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
    invalid = invalid_gates(data, cfg)
    eligible = (
        all(all(g.values()) for g in gates.values())
        and all(x["passed"] for x in latency.values())
        and invalid["passed"]
    )
    diagnostics = {
        name: {
            arm: {
                suite: diagnostic_summary(rows(data[name, arm], suite))
                for suite in cfg["evaluation"]["final"]
            }
            for arm in ("control", "curriculum", "reference")
        }
        for name in pairs
    }
    return {
        "complete": True,
        "eligible": eligible,
        "submission_promotion": False,
        "candidate_for_confirmation": "curriculum-r1" if eligible else None,
        "gates": gates,
        "contrasts": contrasts,
        "latency": latency,
        "invalid_actions": invalid,
        "training_diagnostics": training,
        "evaluation_diagnostics": diagnostics,
        "diagnostic_note": "Reference worlds repeat across replicas; do not treat them "
        "as independent reference samples. Snapshot training summaries are cumulative.",
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
