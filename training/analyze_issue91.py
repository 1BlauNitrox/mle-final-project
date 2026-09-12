"""Verify and analyze the prospective Issue 91 paired gamma comparison."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import fmean

import numpy as np
import torch

from agent_code.DagobertDuckDQNTask2.features import normalize_features, state_to_features
from agent_code.DagobertDuckDQNTask2.persistence import load_evaluation_checkpoint
from training import analyze_issue107_task2_factorial as stats
from training.aggregate import read_episodes_csv
from training.run_issue91 import LIMITS, read, sha, validate, write


def decision(comparison, guards, deterministic, latency):
    efficacy = comparison["mean_difference"] >= 0.10 and comparison["ci95_lower"] > 0
    safe = all(value["ci95_lower"] > -0.05 for value in guards.values())
    return (
        "adopt_gamma_0_97"
        if efficacy and safe and deterministic and latency
        else "retain_gamma_0_90"
    )


def q_probe(artifact):
    """Value-scale diagnostics on fixed public synthetic states, never training data."""
    loaded = load_evaluation_checkpoint(artifact)
    field = np.zeros((17, 17), dtype=int)
    field[[0, -1], :] = -1
    field[:, [0, -1]] = -1
    field[2:16:2, 2:16:2] = -1
    vectors = []
    for crates, bombs in ((False, False), (True, False), (True, True)):
        board = field.copy()
        if crates:
            board[3, 1] = 1
            board[1, 3] = 1
        state = {
            "field": board,
            "self": ("probe", 0, True, (1, 1)),
            "coins": [(5, 1)],
            "others": [],
            "bombs": [((1, 1), 3)] if bombs else [],
            "explosion_map": np.zeros_like(board),
            "round": 1,
            "step": 1,
        }
        vectors.append(
            normalize_features(
                state_to_features(
                    state, include_continuation_features=loaded.config.escape_continuation_features
                )
            )
        )
    with torch.no_grad():
        values = loaded.network(torch.from_numpy(np.stack(vectors))).numpy()
    if not np.isfinite(values).all():
        raise ValueError("Non-finite learned Q-values")
    return {
        "gamma": loaded.config.discount_factor,
        "mean_abs_q": float(np.abs(values).mean()),
        "max_abs_q": float(np.abs(values).max()),
        "q_values": values.tolist(),
    }


def analyze(root):
    root = Path(root).resolve()
    plans = validate(root)
    auth = read(root / "authorization.json")
    resources = read(root / "resources.json")
    if (
        resources["limits"] != vars(LIMITS)
        or resources["limit_reached"]
        or resources["active_root_pids"]
    ):
        raise ValueError("Incomplete or breached resource record")
    for key, cap in (
        ("cpu_seconds_consumed", LIMITS.cpu_seconds),
        ("wall_seconds_elapsed", LIMITS.wall_seconds),
        ("peak_memory_bytes", LIMITS.memory_bytes),
    ):
        if not 0 <= resources[key] <= cap:
            raise ValueError("Resource budget exceeded")
    files, rows, failures, final, diagnostics = {}, [], [], {}, []

    def record(path):
        files[path.relative_to(root).as_posix()] = {
            "sha256": sha(path),
            "size_bytes": path.stat().st_size,
        }

    for path in (root / "authorization.json", root / "protocol.json", root / "resources.json"):
        record(path)
    deterministic = True
    for cell, plan in plans.items():
        directory = root / "run-plans" / plan.plan_id
        status = read(directory / "status.json")
        resolved = read(directory / "resolved_plan.json")
        if resolved != plan.to_dict() or set(status["jobs"]) != {j.run_id for j in plan.jobs}:
            raise ValueError("Registered run matrix or fingerprints changed")
        record(directory / "status.json")
        record(directory / "resolved_plan.json")
        for job in plan.jobs:
            item = status["jobs"][job.run_id]
            if item["status"] != "completed" or not item["attempts"]:
                raise ValueError(f"Incomplete job: {cell}/{job.run_id}")
            failures.extend(
                {"cell": cell, "job": job.run_id, **a}
                for a in item["attempts"]
                if a["status"] != "completed"
            )
            attempt = directory / item["attempts"][-1]["output"]
            metadata = read(attempt / "metadata.json")
            stats._validate_job_campaign(metadata, auth, job.run_id)
            for name in ("scenario", "world_seed", "agent_seed", "rounds"):
                if metadata[name] != getattr(job, name):
                    raise ValueError("Job conditions changed")
            if metadata["mode"] != job.kind or metadata["opponents"] != []:
                raise ValueError("Job mode or opponents changed")
            if metadata["run_plan"]["fingerprints"] != plan.fingerprints:
                raise ValueError("Job provenance fingerprint changed")
            episodes = read_episodes_csv(attempt / "episodes.csv")
            if len(episodes) != job.rounds:
                raise ValueError("Incomplete episode count")
            record(attempt / "metadata.json")
            record(attempt / "episodes.csv")
            if job.kind == "training":
                artifact = directory / item["artifact"]["path"]
                if sha(artifact) != item["artifact"]["sha256"]:
                    raise ValueError("Training artifact changed")
                probe = q_probe(artifact)
                if probe["gamma"] != (0.9 if cell == "A" else 0.97):
                    raise ValueError("Training silently changed gamma")
                for metric in ("mean_loss", "mean_abs_td_error"):
                    values = [r[metric] for r in episodes if r.get(metric) is not None]
                    if not values or not all(math.isfinite(v) for v in values):
                        raise ValueError("Missing or non-finite training diagnostics")
                    probe[metric] = fmean(values)
                diagnostics.append(
                    {"cell": cell, "replica": job.replica, "stage": job.stage_or_suite, **probe}
                )
                record(artifact)
                if job.stage_or_suite == "classic":
                    loaded = load_evaluation_checkpoint(artifact)
                    if loaded.completed_episodes != 10000:
                        raise ValueError("Not an exact-budget final checkpoint")
                    final[(cell, job.replica)] = item["artifact"]
        for replica in plan.replicas:
            for suite, scenario in stats.PRIMARY_SUITES.items():
                if not any(j.stage_or_suite == suite for j in plan.jobs):
                    continue
                primary, primary_hash = stats._suite_rows(
                    directory, status, resolved, replica.replica_id, suite, "issue91", auth
                )
                repeat, repeat_hash = stats._suite_rows(
                    directory,
                    status,
                    resolved,
                    replica.replica_id,
                    suite.replace("primary", "repeat"),
                    "issue91",
                    auth,
                )
                expected = (
                    final[(cell, replica.replica_id)]["sha256"]
                    if cell in ("A", "B")
                    else (replica.parent_artifact_sha256)
                )
                if primary_hash != {expected} or repeat_hash != {expected}:
                    raise ValueError("Evaluation did not use the exact final artifact")
                for first, second in zip(primary, repeat, strict=True):
                    if any(
                        first.get(k) is None or second.get(k) is None
                        for k in stats.DETERMINISTIC_COLUMNS
                    ):
                        raise ValueError("Missing deterministic evidence")
                    deterministic &= all(first[k] == second[k] for k in stats.DETERMINISTIC_COLUMNS)
                    if first["initially_available_coins"] <= 0:
                        raise ValueError("Missing total-board coin denominator")
                    rows.append(
                        {
                            **first,
                            "cell": cell,
                            "model": replica.replica_id,
                            "scenario": scenario,
                            "collection_fraction": first["coins_collected"]
                            / first["initially_available_coins"],
                        }
                    )
    if len(rows) != 1360 or len(final) != 10:
        raise ValueError("Incomplete scientific matrix")
    summaries = stats._summarize(rows)
    values = stats._metric_values(rows)
    primary = stats._contrast(values, "B", "A", "classic", "collection", 9101, 1.0)
    guards = {}
    for scenario in ("classic", "coin-heaven", "loot-crate"):
        for metric in ("collection", "survival", "self_kill"):
            name = scenario + "_" + metric
            guards[name] = stats._contrast(
                values,
                "B",
                "A",
                scenario,
                metric,
                stats._stable_seed("issue91_" + name),
                -1.0 if metric == "self_kill" else 1.0,
            )
    contrasts = {}
    for cell in ("A", "B"):
        contrasts[f"{cell.lower()}_classic_minus_untrained"] = stats._contrast(
            values, cell, "untrained", "classic", "collection", 9110 + ord(cell), 1.0
        )
        contrasts[f"{cell.lower()}_coin_heaven_minus_task1"] = stats._contrast(
            values, cell, "frozen_task1", "coin-heaven", "collection", 9210 + ord(cell), 1.0
        )
    gates = {
        cell: stats._absolute_gates(cell, rows, summaries, contrasts, deterministic)
        for cell in ("A", "B")
    }
    latency = all(stats._latency_ok(row) for row in rows)
    verdict = decision(primary, guards, deterministic, latency)
    selected = "B" if verdict == "adopt_gamma_0_97" else "A"
    replica = stats._select_representative(selected, summaries)
    for item in summaries:
        item["collected_per_revealed_coin_proxy"] = (
            item["mean_coins"] * item["episodes"] / item["coins_found"]
            if item["coins_found"]
            else None
        )
    result = {
        "issue": 91,
        "analysis_valid": deterministic and latency,
        "decision": verdict if deterministic and latency else "invalid_evaluation",
        "primary": primary,
        "guards": guards,
        "absolute_gates": gates,
        "deterministic": deterministic,
        "latency": latency,
        "selection": {
            "cell": selected,
            "replica": replica,
            "artifact": final[(selected, replica)],
            "task2_complete": gates[selected]["overall_passed"],
        }
        if deterministic and latency
        else None,
        "failed_attempts": failures,
        "summaries": summaries,
        "training_diagnostics": diagnostics,
    }
    destination = root / "analysis"
    destination.mkdir(exist_ok=True)
    stats._write_csv(destination / "episodes.csv", rows)
    stats._write_csv(destination / "summary.csv", summaries)
    write(destination / "result.json", result)
    write(destination / "evidence-files.json", files)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    args = parser.parse_args()
    print(analyze(args.campaign_root)["decision"])


if __name__ == "__main__":
    main()
