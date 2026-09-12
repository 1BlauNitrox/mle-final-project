"""Verify and analyze the prospectively amended 19-replica Issue 124 evaluation."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import zipfile
from pathlib import Path

import yaml

from training import analyze_issue107_task2_factorial as stats
from training.aggregate import _parse_csv_row
from training.run_issue124_campaign import PLANS

EXECUTION = "1ce18c8736b6a50773b60b95ec0f11dd4c901028"
CELLS = ("A", "B", "C", "D", "untrained", "frozen_task1")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def import_archive(path, training_root):
    """Retain lossless required observations/metadata; omit redundant agent copies."""
    evidence = {"records": {}, "hashes": {}, "artifact_verification": {}}
    with zipfile.ZipFile(path) as archive:
        require(len(archive.namelist()) == len(set(archive.namelist())), "Duplicate archive paths")

        def get(name):
            data = archive.read(name)
            evidence["hashes"][name] = {
                "sha256": hashlib.sha256(data).hexdigest(),
                "size_bytes": len(data),
            }
            result = json.loads(data) if name.endswith(".json") else data.decode("utf-8-sig")
            evidence["records"][name] = result
            return result

        for name in (
            "status.json",
            "authorization.json",
            "resources.json",
            "runtime-extension.json",
            "evaluation-amendment.json",
            "laptop-migration.json",
        ):
            get("laptop-evaluation/" + name)
        for name in ("resources.json", "authorization.json"):
            get("previous-evaluation/" + name)
        get("transfer-manifest.json")
        protocol = evidence["records"]["laptop-evaluation/evaluation-amendment.json"]
        for record in protocol["plans"]:
            get("laptop-evaluation/" + record["path"])
            cell = record["cell"]
            plan_id = "issue124-reduced-eval-" + cell.lower()
            base = "laptop-evaluation/run-plans/" + plan_id + "/"
            status = get(base + "status.json")
            get(base + "resolved_plan.json")
            for job in status["jobs"].values():
                for attempt in job["attempts"]:
                    if attempt.get("metadata"):
                        get(base + attempt["metadata"])
                if job["status"] == "completed":
                    get(base + job["attempts"][-1]["output"] + "/episodes.csv")
                    relative = base + job["artifact"]["path"]
                    if relative not in evidence["artifact_verification"]:
                        data = archive.read(relative)
                        evidence["artifact_verification"][relative] = {
                            "sha256": hashlib.sha256(data).hexdigest(),
                            "size_bytes": len(data),
                        }
    evidence["training_provenance"] = {}
    for cell in "ABCD":
        directory = training_root / "run-plans" / ("issue124-cell-" + cell.lower())
        status = json.loads((directory / "status.json").read_text())
        finals = {}
        for replica in range(1, 5 if cell == "D" else 6):
            job = status["jobs"][f"train-r{replica}-block-20"]
            artifact_path = directory / job["artifact"]["path"]
            finals[f"r{replica}"] = dict(
                job=job,
                metadata=json.loads((directory / job["attempts"][-1]["metadata"]).read_text()),
                artifact_sha256=hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
                artifact_size_bytes=artifact_path.stat().st_size,
            )
        evidence["training_provenance"][cell] = dict(status=status, finals=finals)
    return evidence


def verify(evidence):
    records, hashes = evidence["records"], evidence["hashes"]
    base = "laptop-evaluation/"
    protocol = records[base + "evaluation-amendment.json"]
    auth = records[base + "authorization.json"]
    resource = records[base + "resources.json"]
    migration = records[base + "laptop-migration.json"]
    require(
        records[base + "status.json"]["status"] == "evaluation_completed", "Evaluation incomplete"
    )
    require(
        auth["protocol_sha256"] == hashes[base + "evaluation-amendment.json"]["sha256"],
        "Authorization hash mismatch",
    )
    limits = dict(cpu_seconds=115200, wall_seconds=86400, memory_bytes=2147483648)
    require(
        resource["limits"] == limits
        and not resource["active_root_pids"]
        and resource["limit_reached"] is None,
        "Resource violation",
    )
    for field, ceiling in [
        ("cpu_seconds_consumed", 115200),
        ("wall_seconds_elapsed", 86400),
        ("peak_memory_bytes", 2147483648),
    ]:
        require(0 <= resource[field] <= ceiling, "Resource ceiling exceeded")
    previous = records["previous-evaluation/resources.json"]
    require(
        resource["authorized_at"] == auth["authorized_at"] == previous["authorized_at"],
        "Budget origin changed",
    )
    require(
        resource["cpu_seconds_consumed"] >= previous["cpu_seconds_consumed"], "CPU accounting reset"
    )
    for name, field in [
        ("resources.json", "original_resources_sha256"),
        ("authorization.json", "original_authorization_sha256"),
    ]:
        require(
            migration[field] == hashes["previous-evaluation/" + name]["sha256"],
            "Migration history hash mismatch",
        )
    require(
        migration["transfer_manifest_sha256"] == hashes["transfer-manifest.json"]["sha256"],
        "Transfer manifest changed",
    )
    require({p["cell"] for p in protocol["plans"]} == set(CELLS), "Wrong cells")
    for cell in "ABCD":
        training = evidence["training_provenance"][cell]
        for final in training["finals"].values():
            require(
                final["job"]["status"] == "completed"
                and final["metadata"]["status"] == "completed",
                "Incomplete final training",
            )
            require(
                final["metadata"]["git_commit"] == EXECUTION
                and final["metadata"]["mode"] == "training"
                and final["metadata"]["rounds"] == 500,
                "Final training provenance changed",
            )
            require(
                final["artifact_sha256"] == final["job"]["artifact"]["sha256"],
                "Training artifact hash mismatch",
            )
        require(
            len(training["finals"]) == (4 if cell == "D" else 5), "Wrong final training population"
        )
    observations = []
    failures = []
    artifacts = {}
    fingerprints = []
    for item in protocol["plans"]:
        cell = item["cell"]
        plan_id = "issue124-reduced-eval-" + cell.lower()
        prefix = base + "run-plans/" + plan_id + "/"
        require(hashes[base + item["path"]]["sha256"] == item["sha256"], "Registered YAML changed")
        raw = yaml.safe_load(records[base + item["path"]])
        original = yaml.safe_load(PLANS[cell].read_text())
        expected = dict(original, plan_id=plan_id, training_stages=[], max_parallel_training=1)
        expected["replicas"] = [
            r for r in original["replicas"] if not (cell == "D" and r["id"] == "r5")
        ]

        def without_paths(value):
            value = json.loads(json.dumps(value))
            for replica in value["replicas"]:
                replica.pop("parent_artifact", None)
            return value

        require(
            without_paths(raw) == without_paths(expected),
            "Registered scenarios/seeds/settings changed",
        )
        resolved = records[prefix + "resolved_plan.json"]
        status = records[prefix + "status.json"]
        fingerprints.append(resolved["fingerprints"])
        expected_jobs = []
        for replica in raw["replicas"]:
            for suite in raw["evaluation_suites"]:
                for index, (world, agent) in enumerate(
                    zip(suite["world_seeds"], suite["agent_seeds"], strict=True), 1
                ):
                    expected_jobs.append(
                        dict(
                            run_id=f"eval-{replica['id']}-{suite['id']}-seed-{index:03d}",
                            kind="evaluation",
                            replica=replica["id"],
                            stage_or_suite=suite["id"],
                            population=suite["population"],
                            scenario=suite["scenario"],
                            rounds=suite["rounds"],
                            world_seed=world,
                            agent_seed=agent,
                            opponents=suite["opponents"],
                        )
                    )
        require(resolved["jobs"] == expected_jobs, "Resolved job matrix changed")
        require(
            status["status"] == "completed"
            and set(status["jobs"]) == {j["run_id"] for j in expected_jobs},
            "Incomplete jobs",
        )
        require(
            {r["replica_id"]: r["parent_artifact_sha256"] for r in resolved["replicas"]}
            == item["artifacts"],
            "Parent binding changed",
        )
        for job in expected_jobs:
            record = status["jobs"][job["run_id"]]
            require(
                record["status"] == "completed" and record["attempts"][-1]["status"] == "completed",
                "Incomplete attempt",
            )
            failures.extend(
                dict(cell=cell, job=job["run_id"], attempt=a)
                for a in record["attempts"]
                if a["status"] != "completed"
            )
            run = prefix + record["attempts"][-1]["output"] + "/"
            meta = records[run + "metadata.json"]
            details = meta["run_plan"]
            require(
                meta["status"] == "completed"
                and meta["return_code"] == 0
                and meta["git_dirty"] is False
                and meta["git_commit"] == EXECUTION,
                "Invalid execution provenance",
            )
            require(meta["python_version"] == "3.13.15", "Unexpected Python population")
            require(
                meta["mode"] == "evaluation" and meta["opponents"] == [], "Wrong execution mode"
            )
            require(
                details["campaign"]
                == dict(
                    authorization=auth,
                    reduced_evaluation=protocol,
                    runtime_extension=records[base + "runtime-extension.json"],
                ),
                "Campaign metadata changed",
            )
            for field in ("world_seed", "agent_seed", "scenario", "rounds"):
                require(meta[field] == job[field], "Job condition changed: " + field)
            for field in (
                "action_masking",
                "escape_continuations",
                "replay_treatment",
                "reward_variant",
                "useful_bomb_reward",
                "fingerprints",
            ):
                require(details[field] == resolved[field], "Treatment changed: " + field)
            require(
                details["processes"] == 1 and details["artifact_writable"] is False,
                "Evaluation contract violated",
            )
            for field, expected in (
                ("job_id", job["run_id"]),
                ("replica", job["replica"]),
                ("stage_or_suite", job["stage_or_suite"]),
                ("plan_id", plan_id),
            ):
                require(details[field] == expected, "Job identity mismatch")
            artifact = evidence["artifact_verification"][prefix + record["artifact"]["path"]]
            require(
                artifact["sha256"]
                == record["artifact"]["sha256"]
                == item["artifacts"][job["replica"]],
                "Artifact checksum mismatch",
            )
            if cell in "ABCD":
                final = evidence["training_provenance"][cell]["finals"][job["replica"]]
                require(
                    artifact["sha256"] == final["artifact_sha256"],
                    "Evaluation not bound to final training",
                )
            artifacts[cell + "/" + job["replica"]] = artifact
            csv_rows = list(csv.DictReader(io.StringIO(records[run + "episodes.csv"])))
            require(len(csv_rows) == 1, "Wrong episode count")
            row = _parse_csv_row(csv_rows[0], 2)
            require(
                all(row.get(k) is not None for k in stats.DETERMINISTIC_COLUMNS),
                "Missing repeat evidence",
            )
            require(row["initially_available_coins"] > 0, "Missing total coin denominator")
            observations.append(
                dict(
                    row,
                    cell=cell,
                    model=job["replica"] if cell in "ABCD" else cell,
                    scenario=job["scenario"],
                    world_seed=job["world_seed"],
                    agent_seed=job["agent_seed"],
                    suite=job["stage_or_suite"],
                    collection_fraction=row["coins_collected"] / row["initially_available_coins"],
                )
            )
    require(len(observations) == 4880, "Incomplete amended matrix")
    for field in ("source", "framework", "dependencies_sha256"):
        require(len({f[field] for f in fingerprints}) == 1, "Mixed environment/source")
    return observations, artifacts, failures


def analyze(evidence):
    observations, artifacts, failures = verify(evidence)
    rows = [r for r in observations if r["suite"].endswith("-primary")]
    lookup = {(r["cell"], r["model"], r["suite"], r["world_seed"]): r for r in observations}
    mismatches = []
    for row in rows:
        repeat = lookup[
            (
                row["cell"],
                row["model"],
                row["suite"].replace("-primary", "-repeat"),
                row["world_seed"],
            )
        ]
        fields = [k for k in stats.DETERMINISTIC_COLUMNS if row[k] != repeat[k]]
        if fields:
            mismatches.append(
                {k: row[k] for k in ("cell", "model", "suite", "world_seed")} | {"fields": fields}
            )
    deterministic = not mismatches
    latency = all(stats._latency_ok(r) for r in observations)
    summaries = stats._summarize(rows)
    values = stats._metric_values(rows)
    contrasts = {}
    for name, treatment, scenario, seed in [
        ("rehearsal_b_minus_a", "B", "coin-heaven", 12401),
        ("mask_c_minus_a", "C", "classic", 12403),
    ]:
        contrasts[name] = stats._contrast(values, treatment, "A", scenario, "collection", seed, 1.0)
    for cell in "BC":
        guards = {}
        for scenario in stats.PRIMARY_SUITES.values():
            for metric in ("collection", "survival"):
                name = f"124_{cell}_A_{scenario}_{metric}"
                guards[f"{scenario}_{metric}"] = stats._contrast(
                    values, cell, "A", scenario, metric, stats._stable_seed(name), 1.0
                )
        contrasts[f"guards_{cell.lower()}_minus_a"] = guards
    for cell in "ABC":
        for suffix, baseline, scenario in [
            ("classic_minus_untrained", "untrained", "classic"),
            ("coin_heaven_minus_task1", "frozen_task1", "coin-heaven"),
        ]:
            name = f"{cell.lower()}_{suffix}"
            contrasts[name] = stats._contrast(
                values,
                cell,
                baseline,
                scenario,
                "collection",
                stats._stable_seed("124_" + name),
                1.0,
            )
    paired_four = stats._metric_values([row for row in rows if row["model"] != "r5"])
    exploratory = {
        "scope": "D and matched A/B/C r1-r4 only; cannot establish eligibility",
        "rehearsal_d_minus_c": stats._contrast(
            paired_four, "D", "C", "coin-heaven", "collection", 12402, 1.0
        ),
        "mask_d_minus_b": stats._contrast(
            paired_four, "D", "B", "classic", "collection", 12404, 1.0
        ),
        "interaction": stats._interaction(paired_four, 12405),
    }
    gates = {c: stats._absolute_gates(c, rows, summaries, contrasts, deterministic) for c in "ABC"}
    eligible = {
        "A": True,
        "B": stats._efficacy(contrasts["rehearsal_b_minus_a"], 0.10)
        and stats._guards(contrasts["guards_b_minus_a"]),
        "C": stats._efficacy(contrasts["mask_c_minus_a"], 0.10)
        and stats._guards(contrasts["guards_c_minus_a"]),
    }
    selected = stats._select_cell(dict(eligible, D=False), gates, summaries)
    replica = stats._select_representative(selected, summaries)
    result = dict(
        issue=124,
        scope="registered_reduced_evaluation",
        primary_episodes=len(rows),
        repeat_episodes=len(rows),
        summaries=summaries,
        comparisons=contrasts,
        absolute_gates=gates,
        eligibility=eligible,
        deterministic=deterministic,
        latency=latency,
        determinism_mismatches=mismatches,
        analysis_valid=deterministic and latency,
        failed_attempts=failures,
        selection=dict(
            cell=selected,
            replica=replica,
            artifact=artifacts[selected + "/" + replica],
            task2_complete=gates[selected]["overall_passed"],
        )
        if deterministic and latency
        else None,
        D_status="exploratory_four_replicas_not_eligible",
        exploratory=exploratory,
    )
    return result, observations


def verify_checkpoints(path, evidence):
    with zipfile.ZipFile(path) as archive:
        for original, record in evidence["artifact_verification"].items():
            parts = original.split("/")
            name = (
                parts[2].replace("issue124-reduced-eval-", "") + "/" + parts[4] + "/checkpoint.pt"
            )
            data = archive.read(name)
            require(
                len(data) == record["size_bytes"]
                and hashlib.sha256(data).hexdigest() == record["sha256"],
                "Retrievable checkpoint mismatch",
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--training-root", type=Path)
    parser.add_argument("--checkpoints", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.archive:
        require(args.training_root is not None, "Archive import requires --training-root")
        evidence = import_archive(args.archive, args.training_root)
        with gzip.open(args.output / "evidence.json.gz", "wt", encoding="utf-8") as f:
            json.dump(evidence, f, separators=(",", ":"))
    else:
        with gzip.open(args.evidence, "rt", encoding="utf-8") as f:
            evidence = json.load(f)
    if args.checkpoints:
        verify_checkpoints(args.checkpoints, evidence)
    result, rows = analyze(evidence)
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    stats._write_csv(args.output / "episodes.csv", rows)
    stats._write_csv(args.output / "summary.csv", result["summaries"])
    print({k: result[k] for k in ("analysis_valid", "deterministic", "latency", "selection")})


if __name__ == "__main__":
    main()
