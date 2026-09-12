"""Validate paired Task 3 evidence and apply the prospectively registered gates."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from training.aggregate import read_episodes_csv
from training.metrics import _parse_round_number, normalize_episode_rows
from training.run_task3_campaign import (
    CONFIG,
    portable_plan,
    read_json,
    relative_file,
    require,
    sha256,
    validate_protocol,
)

LATENCY = ("decision_time_median_ms", "decision_time_p95_ms", "decision_time_max_ms")
SUITES = ("classic-peaceful", "classic-retention", "coin-heaven-retention", "loot-crate-retention")


def index_rounds(raw):
    """Identify numeric and timestamped framework round keys independently of JSON order."""
    indexed = {_parse_round_number(key): value for key, value in raw["by_round"].items()}
    require(len(indexed) == len(raw["by_round"]), "Duplicate normalized round keys")
    return indexed


def crossed_interval(candidate, reference, *, seed=109, samples=10000, reference_scale=1.0):
    """Pair all models with one reference; do not count its copies as replicas."""
    candidate, reference = np.asarray(candidate, dtype=float), np.asarray(reference, dtype=float)
    require(candidate.ndim == 2 and reference.shape == (candidate.shape[1],), "Unpaired arrays")
    require(np.isfinite(candidate).all() and np.isfinite(reference).all(), "Nonfinite observations")
    rng = np.random.default_rng(seed)
    model_indices = rng.integers(candidate.shape[0], size=(samples, candidate.shape[0]))
    pair_indices = rng.integers(candidate.shape[1], size=(samples, candidate.shape[1]))
    differences = candidate - reference_scale * reference[None, :]
    means = differences[model_indices[:, :, None], pair_indices[:, None, :]].mean(axis=(1, 2))
    low, high = np.quantile(means, [0.025, 0.975])
    return {"mean": float(differences.mean()), "lower": float(low), "upper": float(high)}


def metric(row, name):
    if name == "collection":
        require(row["initially_available_coins"] > 0, "Missing collection denominator")
        return row["coins_collected"] / row["initially_available_coins"]
    if name == "elimination":
        return int(row["opponents_eliminated"] > 0)
    if name == "self_kill":
        return int(row["self_kills"] > 0)
    if name == "survival":
        return int(row["survived"])
    return row[name]


def summarize(rows):
    """Complete observed-agent diagnostics, including absolute earlier-task results."""
    times = [t for row in rows for t in row["decision_times_ms"]]
    attempted = sum(row["attempted_actions"] for row in rows)
    names = [
        "collection",
        "elimination",
        "self_kill",
        "survival",
        "score",
        "coins_collected",
        "crates_destroyed",
        "bombs_dropped",
        "survival_steps",
        "episode_steps",
        "opponents_eliminated",
    ]
    if rows[0]["opponent_count"]:
        names += ["score_margin", "first_place", "tied_first"]
    return {
        "episodes": len(rows),
        **{name: float(np.mean([metric(row, name) for row in rows])) for name in names},
        "invalid_action_rate": sum(r["invalid_actions"] for r in rows) / attempted
        if attempted
        else None,
        "action_counts": {
            name: sum(r[name] for r in rows) for name in rows[0] if name.startswith("action_")
        },
        "decision_median_ms": float(np.median(times)),
        "decision_p95_ms": float(np.percentile(times, 95)),
        "decision_max_ms": float(max(times)),
    }


def decide(rows, config):
    """Apply all conjunctive gates; no post-hoc fallback can produce a pass."""
    models = sorted({r["replica"] for r in rows if r["arm"] == "candidate"})
    require(len(models) == 5, "Require all five candidate replicas")
    primary = [r for r in rows if r["suite"].endswith("-primary")]
    gates, contrasts, summaries = {}, {}, {}
    limits = config["gates"]
    for suite in SUITES:
        groups = {
            model: sorted(
                [
                    r
                    for r in primary
                    if r["arm"] == "candidate"
                    and r["replica"] == model
                    and r["suite"] == suite + "-primary"
                ],
                key=lambda r: (r["world_seed"], r["agent_seed"]),
            )
            for model in models
        }
        reference = sorted(
            [r for r in primary if r["arm"] == "reference" and r["suite"] == suite + "-primary"],
            key=lambda r: (r["world_seed"], r["agent_seed"]),
        )

        def keys(group):
            return [(r["world_seed"], r["agent_seed"]) for r in group]

        require(
            len(reference) == config["evaluation_pairs_per_suite"]
            and len(set(keys(reference))) == len(reference),
            "Missing/duplicate reference pairs",
        )
        require(
            all(keys(group) == keys(reference) for group in groups.values()),
            "Unpaired candidate seeds",
        )
        candidate_rows = [r for group in groups.values() for r in group]
        summaries[suite] = {
            "candidate": summarize(candidate_rows),
            "reference": summarize(reference),
            "per_model": {m: summarize(g) for m, g in groups.items()},
        }

        def contrast(name, scale=1.0, groups=groups, reference=reference, suite=suite):
            result = crossed_interval(
                [[metric(r, name) for r in groups[m]] for m in models],
                [metric(r, name) for r in reference],
                seed=config["bootstrap"]["seed"],
                samples=config["bootstrap"]["samples"],
                reference_scale=scale,
            )
            contrasts[f"{suite}/{name}"] = result
            return result

        if suite == "classic-peaceful":
            elimination = contrast("elimination")
            gates["elimination_absolute"] = (
                summaries[suite]["candidate"]["elimination"] >= limits["elimination_min"]
            )
            gates["elimination_improvement"] = (
                elimination["mean"] >= limits["elimination_improvement_min"]
                and elimination["lower"] > 0
            )
            gates["first_place_absolute"] = (
                summaries[suite]["candidate"]["first_place"] >= limits["first_place_min"]
            )
            gates["first_place_improvement"] = contrast("first_place")["lower"] > 0
            gates["score_margin_absolute"] = summaries[suite]["candidate"]["score_margin"] > 0
            gates["score_margin_improvement"] = contrast("score_margin")["lower"] > 0
            gates["self_kill_absolute"] = (
                summaries[suite]["candidate"]["self_kill"] <= limits["self_kill_max"]
            )
        else:
            margin = (
                limits["task1_collection_margin"]
                if suite == "coin-heaven-retention"
                else limits["task2_collection_margin"]
            )
            gates[suite + "/collection"] = contrast("collection")["lower"] > -margin
            gates[suite + "/survival"] = contrast("survival")["lower"] >= -limits["survival_margin"]
            gates[suite + "/self_kill"] = (
                contrast("self_kill")["upper"] <= limits["self_kill_increase_margin"]
            )
            if suite != "coin-heaven-retention":
                # C - 0.9 R avoids a division by zero for zero-crate baselines.
                gates[suite + "/crates"] = (
                    contrast("crates_destroyed", 1 - limits["crates_relative_margin"])["lower"] >= 0
                )
        gates[suite + "/invalid_actions"] = all(
            group["invalid_action_rate"] is not None
            and group["invalid_action_rate"] < limits["invalid_action_max_exclusive"]
            for group in [summaries[suite]["candidate"], *summaries[suite]["per_model"].values()]
        )
    gates["runtime"] = all(
        r[LATENCY[1]] is not None
        and r[LATENCY[2]] is not None
        and r[LATENCY[1]] < limits["decision_p95_ms_max_exclusive"]
        and r[LATENCY[2]] < limits["decision_max_ms_max_exclusive"]
        for r in rows
    )
    passed = all(gates.values())
    ordering = sorted(
        models, key=lambda m: (summaries["classic-peaceful"]["per_model"][m]["elimination"], m)
    )
    return {
        "status": "exploratory_pass" if passed else "exploratory_mixed_or_negative",
        "task2_complete": False,
        "gates": gates,
        "contrasts": contrasts,
        "summaries": summaries,
        "selected_replica": ordering[len(ordering) // 2] if passed else None,
        "next_decision": "review_coincollector_protocol"
        if passed
        else "stop_no_automatic_continuation",
    }


def load_evidence(root, binding_directory):
    config, plans, _report = validate_protocol(binding_directory)
    auth = read_json(root / "authorization.json")
    identity = auth["identity"]
    require(identity["limits"] == config["resources"], "Authorized resource ceiling mismatch")
    for name, digest in identity["opponent_sources"].items():
        from training.run_task3_campaign import ROOT

        require(
            sha256(ROOT / "agent_code" / name / "callbacks.py") == digest, "Opponent source changed"
        )
    require(identity["protocol_sha256"] == sha256(CONFIG), "Protocol authorization mismatch")
    require(
        identity["binding_sha256"] == sha256(binding_directory / "binding.json"), "Binding changed"
    )
    require(
        identity["plans"] == {n: portable_plan(p) for n, p in plans.items()},
        "Authorized matrix/source mismatch",
    )
    resources = read_json(root / "resources.json")
    require(
        not resources.get("limit_reached") and not resources.get("active_root_pids"),
        "Campaign unfinished or resource breach",
    )
    require(
        resources["authorized_at"].replace("Z", "+00:00")
        == auth["authorized_at"].replace("Z", "+00:00"),
        "Resource authorization mismatch",
    )
    limits = config["resources"]
    expected_limits = {
        "cpu_seconds": limits["cpu_hours"] * 3600,
        "wall_seconds": limits["wall_hours"] * 3600,
        "memory_bytes": limits["memory_gib"] * 1024**3,
    }
    require(resources["limits"] == expected_limits, "Resource limits mismatch")
    require(
        resources["cpu_seconds_consumed"] <= expected_limits["cpu_seconds"]
        and resources["wall_seconds_elapsed"] <= expected_limits["wall_seconds"]
        and resources["peak_memory_bytes"] < expected_limits["memory_bytes"],
        "Exceeded registered resources",
    )
    campaign = {
        "issue": 109,
        "opponent_seed_policy": "task3_per_slot_v1",
        "authorization_sha256": sha256(root / "authorization.json"),
        "reviewed_commit": identity["reviewed_commit"],
    }
    evidence, manifest, training = [], {}, []
    for arm, plan in plans.items():
        directory = root / "plans" / plan.plan_id
        resolved = read_json(directory / "resolved_plan.json")
        status = read_json(directory / "status.json")
        require(
            portable_plan(resolved) == portable_plan(plan),
            "Executed plan differs from registered plan",
        )
        require(
            status["status"] == "completed"
            and set(status["jobs"]) == {j.run_id for j in plan.jobs},
            "Incomplete job matrix",
        )
        model_hashes = {}
        for job in plan.jobs:
            record = status["jobs"][job.run_id]
            require(
                record["status"] == "completed" and record["attempts"][-1]["status"] == "completed",
                "Incomplete job",
            )
            run = relative_file(directory, record["attempts"][-1]["output"])
            meta = read_json(run / "metadata.json")
            require(
                meta.get("opponent_seed_policy") == "task3_per_slot_v1",
                "Missing reproducible opponent RNG control",
            )
            require(
                meta["status"] == "completed"
                and meta["return_code"] == 0
                and not meta["git_dirty"]
                and meta["git_commit"] == identity["reviewed_commit"],
                "Invalid execution provenance",
            )
            for name in ("world_seed", "agent_seed", "scenario", "rounds"):
                require(meta[name] == getattr(job, name), f"Job metadata mismatch: {name}")
            require(
                meta["opponents"] == list(job.opponents) and meta["mode"] == job.kind,
                "Opponent/mode mismatch",
            )
            run_plan = meta["run_plan"]
            require(
                run_plan["campaign"] == campaign
                and run_plan["job_id"] == job.run_id
                and run_plan["logical_agent"] == plan.agent
                and run_plan["processes"] == 1
                and run_plan["artifact_writable"] == (job.kind == "training"),
                "Job authorization mismatch",
            )
            for name in ("action_masking", "escape_continuations", "replay_treatment"):
                require(run_plan[name] == getattr(plan, name), "Persisted mode mismatch")
            artifact = relative_file(directory, record["artifact"]["path"])
            require(sha256(artifact) == record["artifact"]["sha256"], "Artifact bytes changed")
            if job.kind == "evaluation":
                model_hashes.setdefault(job.replica, set()).add(record["artifact"]["sha256"])
            raw = read_json(run / "framework_stats.json")
            normalized = normalize_episode_rows(raw, job.kind)
            csv_rows = read_episodes_csv(run / "episodes.csv")
            observed = meta["observed_agent"]
            require(
                {r["agent"] for r in normalized} == {r["agent"] for r in csv_rows},
                "Participant mismatch",
            )
            normalized = [r for r in normalized if r["agent"] == observed]
            selected = [r for r in csv_rows if r["agent"] == observed]
            require(
                len(selected) == len(normalized) == job.rounds, "Missing observed-agent episodes"
            )
            raw_rounds = index_rounds(raw)
            for row, source in zip(selected, normalized, strict=True):
                require(
                    all(row.get(k) == v for k, v in source.items()), "CSV/raw observation mismatch"
                )
                require(row["opponents_eliminated"] is not None, "Missing attributable kills")
                require(row["opponent_count"] == len(job.opponents), "Opponent count mismatch")
                times = raw_rounds[row["round"]]["agents"][observed]["decision_times_ms"]
                require(
                    times and all(np.isfinite(t) and t >= 0 for t in times),
                    "Missing/nonfinite latency evidence",
                )
                measured = [
                    float(np.median(times)),
                    float(np.percentile(times, 95)),
                    float(max(times)),
                ]
                require(
                    all(
                        np.isclose(row[k], v, rtol=1e-12, atol=1e-12)
                        for k, v in zip(LATENCY, measured, strict=True)
                    ),
                    "Latency summary differs from raw decisions",
                )
                retained = {
                    **row,
                    "arm": arm,
                    "replica": job.replica,
                    "suite": job.stage_or_suite,
                    "world_seed": job.world_seed,
                    "agent_seed": job.agent_seed,
                    "artifact_sha256": record["artifact"]["sha256"],
                }
                if job.kind == "evaluation":
                    evidence.append({**retained, "decision_times_ms": times})
                else:
                    training.append(retained)
            for name in ("metadata.json", "framework_stats.json", "episodes.csv"):
                path = run / name
                manifest[path.relative_to(root).as_posix()] = {
                    "sha256": sha256(path),
                    "size_bytes": path.stat().st_size,
                }
        require(
            all(len(hashes) == 1 for hashes in model_hashes.values()),
            "Evaluation used different model bytes",
        )
        for replica in plan.replicas:
            training_jobs = [
                j for j in plan.jobs if j.kind == "training" and j.replica == replica.replica_id
            ]
            expected = (
                status["jobs"][training_jobs[-1].run_id]["artifact"]["sha256"]
                if training_jobs
                else replica.parent_artifact_sha256
            )
            require(
                model_hashes[replica.replica_id] == {expected},
                "Evaluation did not use the mechanical final checkpoint",
            )
    primary = {
        (
            r["arm"],
            r["replica"],
            r["suite"].removesuffix("-primary"),
            r["world_seed"],
            r["agent_seed"],
        ): r
        for r in evidence
        if r["suite"].endswith("-primary")
    }
    repeats = [r for r in evidence if r["suite"].endswith("-repeat")]
    expected_pairs = (
        sum(j.rounds for p in plans.values() for j in p.jobs if j.kind == "evaluation") // 2
    )
    require(len(primary) == len(repeats) == expected_pairs, "Incomplete primary/repeat matrix")
    for repeat in repeats:
        key = (
            repeat["arm"],
            repeat["replica"],
            repeat["suite"].removesuffix("-repeat"),
            repeat["world_seed"],
            repeat["agent_seed"],
        )
        first = primary[key]
        require(first["executed_action_sequence_sha256"], "Missing action-sequence evidence")
        require(
            all(
                first[k] == repeat[k]
                for k in first
                if k not in (*LATENCY, "decision_times_ms", "suite")
            ),
            "Deterministic repeat mismatch",
        )
    return config, evidence, training, manifest, auth, resources


def analyze(root, binding_directory, output):
    config, rows, training, manifest, auth, resources = load_evidence(root, binding_directory)
    result = decide(rows, config)
    result.update({"authorization": auth, "resources": resources})
    require(not output.exists(), "Analysis output already exists; preserve the previous evidence")
    output.mkdir(parents=True)
    with gzip.open(output / "observations.json.gz", "wt", encoding="utf-8") as file:
        json.dump({"evaluation": rows, "training": training}, file)
    result["observations"] = {
        "path": "observations.json.gz",
        "sha256": sha256(output / "observations.json.gz"),
        "size_bytes": (output / "observations.json.gz").stat().st_size,
    }
    if result["selected_replica"] is not None:
        result["selected_artifact_sha256"] = next(
            r["artifact_sha256"]
            for r in rows
            if r["arm"] == "candidate" and r["replica"] == result["selected_replica"]
        )
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output / "source-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return result


def verify_compact(directory):
    """Recompute every decision and summary from the retained lossless observations."""
    result = read_json(directory / "result.json")
    config = validate_protocol()[0]
    require(
        result["authorization"]["identity"]["protocol_sha256"] == sha256(CONFIG),
        "Compact evidence uses another protocol",
    )
    record = result["observations"]
    path = relative_file(directory, record["path"])
    require(
        sha256(path) == record["sha256"] and path.stat().st_size == record["size_bytes"],
        "Compact observations hash/size mismatch",
    )
    with gzip.open(path, "rt", encoding="utf-8") as file:
        observations = json.load(file)
    if result.get("selected_replica") is not None:
        selected_hashes = {
            row["artifact_sha256"]
            for row in observations["evaluation"]
            if row["arm"] == "candidate" and row["replica"] == result["selected_replica"]
        }
        require(
            len(selected_hashes) == 1 and result.get("selected_artifact_sha256") in selected_hashes,
            "Selected artifact does not match retained replica observations",
        )
    computed = decide(observations["evaluation"], config)
    require(all(result[k] == value for k, value in computed.items()), "Compact result mismatch")
    return {"verified": True, "status": result["status"], "task2_complete": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path)
    parser.add_argument("--binding-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-evidence", type=Path)
    args = parser.parse_args()
    if args.verify_evidence:
        print(json.dumps(verify_compact(args.verify_evidence.resolve()), indent=2))
        return
    if not all((args.campaign_root, args.binding_dir, args.output)):
        parser.error("Analysis requires --campaign-root, --binding-dir and --output")
    result = analyze(
        args.campaign_root.resolve(), args.binding_dir.resolve(), args.output.resolve()
    )
    print(
        json.dumps(
            {k: result[k] for k in ("status", "selected_replica", "task2_complete")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
