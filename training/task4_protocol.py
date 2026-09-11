"""Issue 126 protocol expansion and proposed statistical decision; execution stays blocked."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from training.analyze_task3_campaign import crossed_interval, metric, summarize
from training.run_issue107_campaign import _seed_values_from_path
from training.run_plan import load_plan
from training.run_task3_campaign import require, sha256
from training.task3_mask_campaign import paired_interval

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/2026-09-12-task4-opponent-distribution/config.yaml"
ARMS = ("strong", "mixture", "reference")


def validate():
    config = yaml.safe_load(CONFIG.read_text())
    plans = {arm: load_plan(ROOT / path) for arm, path in config["plans"].items()}
    signatures = []
    for arm, plan in plans.items():
        budget = [
            sum(j.rounds for j in plan.jobs if j.kind == k) for k in ("training", "evaluation")
        ]
        require(
            budget == ([0, 720] if arm == "reference" else [25000, 3600]), "Wrong Task 4 budget"
        )
        signatures.append(
            [
                (j.stage_or_suite, j.scenario, j.opponents, j.world_seed, j.agent_seed)
                for j in plan.jobs
                if j.kind == "evaluation" and j.replica == plan.replicas[0].replica_id
            ]
        )
        require(all(r.parent_artifact is None for r in plan.replicas), "Unregistered parent")
    require(all(s == signatures[0] for s in signatures), "Unmatched Task 4 evaluation")
    seeds = {s for plan in plans.values() for j in plan.jobs for s in (j.world_seed, j.agent_seed)}
    protected = {s for a, b in config["protected_seed_ranges"] for s in range(a, b + 1)}
    require(not seeds & protected, "Protected Task 4 seed")
    excluded = {CONFIG, *(ROOT / path for path in config["plans"].values())}
    for path in [
        *(ROOT / "training/run_plans").glob("*.yaml"),
        *(ROOT / "experiments").glob("*/config.yaml"),
    ]:
        if path not in excluded:
            require(
                not (seeds | protected) & _seed_values_from_path(path), f"Seed collision: {path}"
            )
    return (
        config,
        plans,
        {
            "issue": 126,
            "training_episodes": 50000,
            "evaluation_episodes": 7920,
            "parent_bound": False,
            "compute_authorized": False,
            "launch_blockers": [
                "passing reviewed #137 selected parent unavailable",
                "owner ratification of numeric protocol and budget",
                "bound raw-evidence/launch adapter and workload timing required",
            ],
            "protocol_sha256": sha256(CONFIG),
        },
    )


def decide_primary(rows, config):
    """Proposed statistics on verified primary rows; not a substitute for raw provenance checks."""
    suites = (
        "strong-three",
        "strong-one",
        "mixed-three",
        "learned-one",
        "classic-peaceful",
        "classic-collector",
        "classic-retention",
        "coin-heaven-retention",
        "loot-crate-retention",
    )
    require({r["suite"] for r in rows} == set(suites), "Missing/extra Task 4 suites")
    require({r["arm"] for r in rows} == set(ARMS), "Wrong Task 4 arms")
    limits = config["gates"]
    summaries, contrasts, gates = {}, {}, {}
    groups_by_suite = {}
    for suite in suites:
        groups = {}
        for arm in ARMS:
            selected = [r for r in rows if r["suite"] == suite and r["arm"] == arm]
            models = sorted({r["replica"] for r in selected})
            require(len(models) == (1 if arm == "reference" else 5), "Missing Task 4 replica")
            groups[arm] = [
                sorted(
                    [r for r in selected if r["replica"] == m],
                    key=lambda r: (r["world_seed"], r["agent_seed"]),
                )
                for m in models
            ]
            require(all(len(g) == 40 for g in groups[arm]), "Missing Task 4 pair")
        def keys(group):
            return [(r["world_seed"], r["agent_seed"]) for r in group]
        expected = keys(groups["reference"][0])
        require(
            len(set(expected)) == 40
            and all(keys(g) == expected for arm in ARMS for g in groups[arm]),
            "Unpaired Task 4 worlds",
        )
        groups_by_suite[suite] = groups
        summaries[suite] = {
            arm: {
                "aggregate": summarize([r for g in groups[arm] for r in g]),
                "per_model": [summarize(g) for g in groups[arm]],
            }
            for arm in ARMS
        }
        for arm in ("strong", "mixture"):
            values = summaries[suite][arm]
            gates[f"{arm}/{suite}/invalid"] = all(
                g["invalid_action_rate"] < 0.01 for g in [values["aggregate"], *values["per_model"]]
            )
            gates[f"{arm}/{suite}/self_kill"] = values["aggregate"]["self_kill"] <= 0.10

            def contrast(name, scale=1.0, groups=groups, arm=arm, suite=suite):
                result = crossed_interval(
                    [[metric(r, name) for r in g] for g in groups[arm]],
                    [metric(r, name) for r in groups["reference"][0]],
                    seed=126,
                    samples=10000,
                    reference_scale=scale,
                )
                contrasts[f"{arm}/{suite}/{name}"] = result
                return result

            gates[f"{arm}/{suite}/survival"] = contrast("survival")["lower"] >= -0.05
            gates[f"{arm}/{suite}/self_kill_retention"] = contrast("self_kill")["upper"] <= 0.02
            if suite.endswith("retention"):
                margin = 0.02 if suite == "coin-heaven-retention" else 0.03
                gates[f"{arm}/{suite}/collection"] = contrast("collection")["lower"] > -margin
                if suite != "coin-heaven-retention":
                    gates[f"{arm}/{suite}/crates"] = contrast("crates_destroyed", 0.9)["lower"] >= 0
            if suite in ("classic-peaceful", "classic-collector"):
                gates[f"{arm}/{suite}/hunting"] = contrast("elimination")["lower"] >= -0.05
            if suite == "strong-three":
                gates[f"{arm}/strong_absolute"] = (
                    values["aggregate"]["first_place"] >= 0.35
                    and values["aggregate"]["score_margin"] > 0
                )
                gates[f"{arm}/parent_score"] = contrast("score")["lower"] > 0
    primary = groups_by_suite["strong-three"]
    effect = paired_interval(
        [[r["score"] for r in g] for g in primary["mixture"]],
        [[r["score"] for r in g] for g in primary["strong"]],
        seed=126,
    )
    gates["mixture_effect"] = (
        effect["mean"] >= limits["primary_score_improvement_min"] and effect["lower"] > 0
    )
    gates["runtime"] = all(
        np.isfinite(r["decision_time_p95_ms"])
        and r["decision_time_p95_ms"] < 50
        and np.isfinite(r["decision_time_max_ms"])
        and r["decision_time_max_ms"] < 100
        for r in rows
    )
    eligible = (
        gates["mixture_effect"]
        and gates["runtime"]
        and all(v for k, v in gates.items() if k.startswith("mixture/"))
    )
    ordering = sorted(
        range(5), key=lambda i: (summaries["strong-three"]["mixture"]["per_model"][i]["score"], i)
    )
    models = sorted({r["replica"] for r in rows if r["arm"] == "mixture"})
    return {
        "statistics_only": True,
        "raw_integrity_required": True,
        "task2_complete": False,
        "gates": gates,
        "contrasts": contrasts,
        "summaries": summaries,
        "primary_effect": effect,
        "proposed_selected_replica": models[ordering[2]] if eligible else None,
        "automatic_training_authorized": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(validate()[2], indent=2))


if __name__ == "__main__":
    main()
