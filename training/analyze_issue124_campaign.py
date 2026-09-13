"""Verify Issue #124 observations and apply its prospective factorial decision."""

from __future__ import annotations

import argparse
from pathlib import Path

from training import analyze_issue107_task2_factorial as stats
from training.aggregate import read_episodes_csv
from training.run_issue124_campaign import (
    CONFIG,
    LIMITS,
    git,
    read_json,
    sha256,
    validate_protocol,
    write_json,
)


def contained(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Evidence path escapes plan directory")
    return path


def comparisons(rows):
    values = stats._metric_values(rows)
    result = {}
    for index, (name, treatment, baseline, scenario) in enumerate(
        (
            ("rehearsal_b_minus_a", "B", "A", "coin-heaven"),
            ("rehearsal_d_minus_c", "D", "C", "coin-heaven"),
            ("mask_c_minus_a", "C", "A", "classic"),
            ("mask_d_minus_b", "D", "B", "classic"),
        )
    ):
        result[name] = stats._contrast(
            values, treatment, baseline, scenario, "collection", 12401 + index, 1.0
        )
    for treatment, baseline in (("B", "A"), ("C", "A"), ("D", "B"), ("D", "C"), ("D", "A")):
        guards = {}
        for scenario in stats.PRIMARY_SUITES.values():
            for metric in ("collection", "survival"):
                name = f"124_{treatment}_{baseline}_{scenario}_{metric}"
                guards[f"{scenario}_{metric}"] = stats._contrast(
                    values, treatment, baseline, scenario, metric, stats._stable_seed(name), 1.0
                )
        result[f"guards_{treatment.lower()}_minus_{baseline.lower()}"] = guards
    for cell in "ABCD":
        for suffix, baseline, scenario in (
            ("classic_minus_untrained", "untrained", "classic"),
            ("coin_heaven_minus_task1", "frozen_task1", "coin-heaven"),
        ):
            name = f"{cell.lower()}_{suffix}"
            result[name] = stats._contrast(
                values,
                cell,
                baseline,
                scenario,
                "collection",
                stats._stable_seed("124_" + name),
                1.0,
            )
    result["interaction_d_minus_b_minus_c_plus_a"] = stats._interaction(values, 12405)
    return result


def eligibility(contrasts):
    return {
        "A": True,
        "B": stats._efficacy(contrasts["rehearsal_b_minus_a"], 0.10)
        and stats._guards(contrasts["guards_b_minus_a"]),
        "C": stats._efficacy(contrasts["mask_c_minus_a"], 0.10)
        and stats._guards(contrasts["guards_c_minus_a"]),
        "D": stats._efficacy(contrasts["rehearsal_d_minus_c"], 0.10)
        and stats._efficacy(contrasts["mask_d_minus_b"], 0.10)
        and all(stats._guards(contrasts[f"guards_d_minus_{cell}"]) for cell in "abc"),
    }


def verify_resources(resource):
    if (
        resource.get("limits") != vars(LIMITS)
        or resource.get("limit_reached") is not None
        or resource.get("active_root_pids") != []
    ):
        raise ValueError("Resource record incomplete, changed or breached")
    for name, limit in (
        ("cpu_seconds_consumed", LIMITS.cpu_seconds),
        ("wall_seconds_elapsed", LIMITS.wall_seconds),
        ("peak_memory_bytes", LIMITS.memory_bytes),
    ):
        value = resource.get(name)
        if not isinstance(value, (int, float)) or not 0 <= value <= limit:
            raise ValueError(f"Invalid resource usage: {name}")


def analyze(root):
    root = Path(root).resolve()
    plans = validate_protocol()
    authorization = read_json(root / "authorization.json")
    if authorization["reviewed_commit"] != git("rev-parse", "HEAD") or authorization[
        "config_sha256"
    ] != sha256(CONFIG):
        raise ValueError("Analyze from the exact registered execution revision")
    verify_resources(read_json(root / "resources.json"))
    rows, failed, files, final_artifacts = [], [], {}, {}
    observations = {}
    deterministic, latency = True, True

    def record(path):
        relative = path.relative_to(root).as_posix()
        if relative not in files:
            files[relative] = {"sha256": sha256(path), "size_bytes": path.stat().st_size}

    record(root / "authorization.json")
    # resources.json remains live during analysis; record its final value below.
    for cell, plan in plans.items():
        directory = root / "run-plans" / plan.plan_id
        resolved = read_json(directory / "resolved_plan.json")
        status = read_json(directory / "status.json")
        if stats._without_parent_locations(resolved) != stats._without_parent_locations(
            plan.to_dict()
        ):
            raise ValueError(f"Registered plan mismatch: {cell}")
        if status["status"] != "completed" or set(status["jobs"]) != {
            job.run_id for job in plan.jobs
        }:
            raise ValueError(f"Incomplete job matrix: {cell}")
        record(directory / "resolved_plan.json")
        record(directory / "status.json")
        hashes = {}
        for job in plan.jobs:
            job_status = status["jobs"][job.run_id]
            attempts = job_status.get("attempts", [])
            if job_status["status"] != "completed" or not attempts:
                raise ValueError(f"Job incomplete: {job.run_id}")
            for attempt in attempts:
                if attempt.get("metadata"):
                    record(contained(directory, attempt["metadata"]))
                if attempt.get("status") != "completed":
                    failed.append({"cell": cell, "job": job.run_id, "attempt": attempt})
            attempt = attempts[-1]
            run = contained(directory, attempt["output"])
            metadata = read_json(run / "metadata.json")
            record(run / "metadata.json")
            if (
                metadata.get("status") != "completed"
                or metadata.get("return_code") != 0
                or metadata.get("git_dirty") is not False
            ):
                raise ValueError(f"Job completion/source invalid: {job.run_id}")
            stats._validate_job_campaign(metadata, authorization, job.run_id)
            for field in ("world_seed", "agent_seed", "scenario", "rounds"):
                if metadata.get(field) != getattr(job, field):
                    raise ValueError(f"Job metadata mismatch: {job.run_id}/{field}")
            details = metadata["run_plan"]
            if metadata.get("mode") != job.kind or metadata.get("opponents") != list(job.opponents):
                raise ValueError("Job conditions changed")
            for field in (
                "action_masking",
                "escape_continuations",
                "replay_treatment",
                "fingerprints",
                "reward_variant",
                "useful_bomb_reward",
            ):
                if details.get(field) != getattr(plan, field):
                    raise ValueError(f"Job treatment mismatch: {field}")
            if details.get("processes") != 1 or details.get("artifact_writable") != (
                job.kind == "training"
            ):
                raise ValueError("Job execution contract mismatch")
            episodes = read_episodes_csv(run / "episodes.csv")
            record(run / "episodes.csv")
            if len(episodes) != job.rounds:
                raise ValueError("Episode budget mismatch")
            artifact = job_status["artifact"]
            artifact_path = contained(directory, artifact["path"])
            record(artifact_path)
            if files[artifact_path.relative_to(root).as_posix()]["sha256"] != artifact["sha256"]:
                raise ValueError("Artifact checksum mismatch")
            if job.kind == "training" and job.stage_or_suite == "block-20":
                final_artifacts[(cell, job.replica)] = {
                    **artifact,
                    "path": artifact_path.relative_to(root).as_posix(),
                    "size_bytes": artifact_path.stat().st_size,
                }
            if job.kind == "evaluation":
                hashes.setdefault(job.replica, set()).add(artifact["sha256"])
                row = episodes[0]
                if any(row.get(key) is None for key in stats.DETERMINISTIC_COLUMNS):
                    raise ValueError("Missing deterministic evidence")
                available = row.get("initially_available_coins")
                if not isinstance(available, int) or available <= 0:
                    raise ValueError("Missing total-board coin count")
                observations[(cell, job.replica, job.stage_or_suite, job.world_seed)] = row
                latency &= stats._latency_ok(row)
                if job.stage_or_suite.endswith("-primary"):
                    rows.append(
                        {
                            **row,
                            "cell": cell,
                            "model": job.replica if cell in "ABCD" else cell,
                            "scenario": job.scenario,
                            "world_seed": job.world_seed,
                            "agent_seed": job.agent_seed,
                            "collection_fraction": row["coins_collected"] / available,
                        }
                    )
        if any(len(values) != 1 for values in hashes.values()):
            raise ValueError("Evaluation artifact changed")
        for replica in plan.replicas:
            if cell not in "ABCD" and hashes[replica.replica_id] != {
                replica.parent_artifact_sha256
            }:
                raise ValueError("Baseline evaluation artifact changed from its pinned parent")
            if cell in "ABCD" and hashes[replica.replica_id] != {
                final_artifacts[(cell, replica.replica_id)]["sha256"]
            }:
                raise ValueError("Evaluation did not use the final training artifact")
    if len(final_artifacts) != 20 or len(rows) != 2560:
        raise ValueError("Incomplete retained campaign")
    for (cell, replica, suite, seed), primary in observations.items():
        if suite.endswith("-primary"):
            repeat = observations[(cell, replica, suite.replace("-primary", "-repeat"), seed)]
            deterministic &= all(primary[key] == repeat[key] for key in stats.DETERMINISTIC_COLUMNS)
    summaries = stats._summarize(rows)
    contrasts = comparisons(rows)
    gates = {
        cell: stats._absolute_gates(cell, rows, summaries, contrasts, deterministic)
        for cell in "ABCD"
    }
    eligible = eligibility(contrasts)
    selected = stats._select_cell(eligible, gates, summaries)
    replica = stats._select_representative(selected, summaries)
    selection = {
        "cell": selected,
        "replica": replica,
        "artifact": final_artifacts[(selected, replica)],
        "task2_complete": gates[selected]["overall_passed"],
        "task3_status": "eligible_predecessor"
        if gates[selected]["overall_passed"]
        else "exploratory_predecessor_task2_not_complete",
    }
    result = {
        "issue": 124,
        "schema_version": 1,
        "primary_evaluation_episodes": len(rows),
        "repeat_evaluation_episodes": len(rows),
        "summaries": summaries,
        "comparisons": contrasts,
        "absolute_gates": gates,
        "eligibility": eligible,
        "criteria": {
            "deterministic": deterministic,
            "latency": latency,
            "complete_verified_evidence": True,
        },
        "analysis_valid": deterministic and latency,
        "selection": selection if deterministic and latency else None,
        "failed_attempts": failed,
    }
    verify_resources(read_json(root / "resources.json"))
    destination = root / "analysis"
    destination.mkdir(exist_ok=True)
    stats._write_csv(destination / "summary.csv", summaries)
    write_json(destination / "result.json", result)
    record(root / "resources.json")
    write_json(destination / "evidence-files.json", files)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    result = analyze(parser.parse_args().campaign_root)
    print({"analysis_valid": result["analysis_valid"], "selection": result["selection"]})


if __name__ == "__main__":
    main()
