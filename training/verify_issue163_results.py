"""Audit completed #163 evidence, retaining failed repeat/runtime/resource gates."""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import io
import json
import subprocess
import tarfile
import tempfile
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from training import analyze_task3_campaign as analysis
from training import run_plan
from training import task3_attack_campaign as study
from training import verify_task3_mask_results as historical

SOURCE = "1e18b9c659041b4044d04e9ec6563b98c8c471dc"
PARENT = "c0b6081acf5a733258005d46b9430f3ef50e403470a3bb5428661729f9aeedf7"
CONFIG_REL = "experiments/2026-09-12-task3-safe-attack-penalty/config.yaml"
require, sha, read, write = study.require, study.sha256, study.read_json, study.write_json


def fingerprint(source, names, directory=False):
    """Reproduce Windows Path ordering from recorded execution on any verifier host."""
    digest = hashlib.sha256()
    for name in names:
        root = source / name
        paths = root.rglob("*" if directory else "*.py") if root.is_dir() else [root]
        for path in sorted(paths, key=lambda p: tuple(x.casefold() for x in p.parts)):
            relative = path.relative_to(root if directory else source)
            if not path.is_file() or "__pycache__" in relative.parts:
                continue
            if directory and (
                "logs" in relative.parts
                or path.suffix == ".pyc"
                or path.name == run_plan.STAGED_EVALUATION_CHECKPOINT_NAME
            ):
                continue
            digest.update(relative.as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def pinned_code(source):
    source_record = source / "source-record.json"
    if source_record.exists():
        recorded = read(source_record)
        require(recorded["execution_source"] == SOURCE, "Wrong source bundle")
        data = subprocess.check_output(["git", "archive", "--format=tar", SOURCE], cwd=study.ROOT)
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            members = [m for m in archive if m.isfile()]
            require({m.name for m in members} == set(recorded["files"]), "Source inventory differs")
            for member in members:
                path = study.campaign.relative_file(source, member.name)
                actual, canonical = path.read_bytes(), archive.extractfile(member).read()
                require(
                    sha(path) == recorded["files"][member.name]["sha256"], "Source bytes changed"
                )
                require(
                    actual == canonical
                    or (b"\0" not in actual and actual.replace(b"\r\n", b"\n") == canonical),
                    "Source differs from immutable Git content",
                )
    else:
        require(
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
            == SOURCE,
            "Use exact executed source checkout",
        )
        require(
            not subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=source, text=True
            ).strip(),
            "Execution source must be clean",
        )
    names = [
        "training/task3_attack_campaign.py",
        "training/task3_mask_campaign.py",
        "training/metrics.py",
        "training/run_plan.py",
    ]
    names += [
        str(p.relative_to(study.ROOT))
        for p in (study.ROOT / "agent_code/DagobertDuckDQNTask3").rglob("*.py")
    ]
    for name in names:
        require(
            (study.ROOT / name).read_bytes().replace(b"\r\n", b"\n")
            == (source / name).read_bytes().replace(b"\r\n", b"\n"),
            "Analysis code differs: " + name,
        )

    def functions(path):
        return {
            n.name: ast.dump(n, include_attributes=False)
            for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.FunctionDef)
        }

    old, new = (
        functions(source / "training/analyze_task3_campaign.py"),
        functions(Path(analysis.__file__)),
    )
    require(
        all(new[k] == v for k, v in old.items() if k != "load_evidence"),
        "Registered metric/gate functions changed",
    )


def resources(root):
    amendment = read(root / "wall-amendment166.json")
    require(
        amendment["source"] == SOURCE and amendment["wall_extension_seconds"] == 7200,
        "Wrong wall amendment",
    )
    require(
        sha(root / "resume_issue163_bound.py") == amendment["helper_sha256"],
        "Resume helper changed",
    )
    for name, digest in amendment["original_files"].items():
        require(sha(root / name) == digest, "Original amendment input changed: " + name)
    result = {}
    for label, oldname, newname, wall in [
        ("campaign", "runs/resources.json", "runs/resources-amended166.json", 36000),
        ("pipeline", "supervisor-resources.json", "supervisor-resources-amended166.json", 43200),
    ]:
        old, new = read(root / oldname), read(root / newname)
        expected = {"cpu_seconds": 43200, "wall_seconds": wall, "memory_bytes": 2 * 1024**3}
        require(
            old["limits"] == expected
            and new["limits"] == {**expected, "wall_seconds": wall + 7200},
            "Unapproved limits",
        )
        require(
            new["authorized_at"] == old["authorized_at"]
            and not new["active_root_pids"]
            and not new["limit_reached"],
            "Unfinished or reset resource record",
        )
        require(new["cpu_seconds_consumed"] >= old["cpu_seconds_consumed"], "CPU reset")
        elapsed = (
            datetime.fromisoformat(new["updated_at"].replace("Z", "+00:00"))
            - datetime.fromisoformat(new["authorized_at"].replace("Z", "+00:00"))
        ).total_seconds()
        require(abs(elapsed - new["wall_seconds_elapsed"]) < 0.1, "Wall time mismatch")
        result[label] = {
            "original_wall_pass": elapsed <= wall,
            "approved_wall_pass": elapsed <= wall + 7200,
            "cpu_pass": new["cpu_seconds_consumed"] <= 43200,
            "memory_pass": new["peak_memory_bytes"] < 2 * 1024**3,
            "original": old,
            "amended": new,
        }
    return result


def recovery(root):
    audit = root / "recovery-166-001"
    before, after = read(audit / "before.json"), read(audit / "after.json")
    require(before["protected"] == after["protected"], "Recovery protected hashes changed")
    for name, record in before["protected"].items():
        portable = name.replace("\\", "/")
        path = root / portable
        if portable.endswith("/status.json"):
            prior = read(audit / "original-records" / portable)
            current = read(path)
            for key, job in prior["jobs"].items():
                if job["status"] == "completed":
                    require(current["jobs"][key] == job, "Completed job was replaced")
        else:
            require(
                sha(path) == record["sha256"] and path.stat().st_size == record["size"],
                "Protected input changed: " + portable,
            )
    operations = [
        json.loads(line) for line in (audit / "operations.jsonl").read_text().splitlines()
    ]
    rotated = [x for x in operations if x["operation"] == "rotate_complete"]
    for item in rotated:
        require(sha(audit / item["name"]) == item["name"], "Retained storage object changed")
    require(
        len(rotated) == sum(x["rotate"] for x in before["objects"]) == 19, "Wrong recovery count"
    )
    return {
        "rotated_objects": len(rotated),
        "protected_records": len(before["protected"]),
        "completed_jobs_preserved": True,
        "partial_input_retained": True,
    }


def payload_differences(first, second, path=""):
    """Compare persisted learned/replay/RNG state exactly, independent of ZIP serialization."""
    import torch

    if type(first) is not type(second):
        return [path + "/type"]
    if isinstance(first, torch.Tensor):
        return [] if first.dtype == second.dtype and torch.equal(first, second) else [path]
    if isinstance(first, dict):
        if set(first) != set(second):
            return [path + "/keys"]
        return [
            difference
            for key in first
            for difference in payload_differences(first[key], second[key], path + "/" + str(key))
        ]
    if isinstance(first, (tuple, list)):
        if len(first) != len(second):
            return [path + "/length"]
        return [
            difference
            for i, (a, b) in enumerate(zip(first, second, strict=True))
            for difference in payload_differences(a, b, path + "/" + str(i))
        ]
    return [] if first == second else [path]


def matched_training(root, artifacts, rows):
    import torch

    report = []
    ignored = {"arm", "agent", "artifact_sha256", *analysis.LATENCY}
    for replica in ("r1", "r2", "r3", "r4", "r5"):
        pair = [
            next(a for a in artifacts if a["replica"] == replica and a["arm"] == arm)
            for arm in ("control", "neutral")
        ]
        payloads = [
            torch.load(root / a["path"], weights_only=True, map_location="cpu") for a in pair
        ]
        differences = payload_differences(*payloads)
        groups = [
            [r for r in rows if r["arm"] == arm and r["replica"] == replica]
            for arm in ("control", "neutral")
        ]
        require(len(groups[0]) == len(groups[1]) == 5000, "Missing matched training episodes")
        changed = sum(
            any(a[k] != b[k] for k in a if k not in ignored) for a, b in zip(*groups, strict=True)
        )
        replay = payloads[0]["replay_state"]
        states = replay["states"]
        safe, free = states[:, 31] > 0, states[:, 19] == 0
        bombs = replay["action_indices"] == payloads[0]["actions"].index("BOMB")
        report.append(
            {
                "replica": replica,
                "checkpoint_differences": differences,
                "paired_training_episodes": 5000,
                "differing_nonlatency_training_episodes": changed,
                "control_retained_replay_transitions": len(states),
                "bomb_actions": int(bombs.sum()),
                "safe_attack_states": int(safe.sum()),
                "safe_cratefree_attack_states": int((safe & free).sum()),
                "eligible_bomb_actions": int((safe & free & bombs).sum()),
                "eligible_nonterminal_bomb_actions": int(
                    (safe & free & bombs & ~replay["terminals"]).sum()
                ),
            }
        )
    return report


def blocked_decision(performance, failures, resource_report):
    return {
        "status": "exploratory_mixed_or_negative_with_integrity_failure",
        "selected_arm": None,
        "selected_replica": None,
        "task2_complete": False,
        "automatic_continuation_allowed": False,
        "repeat_gate_pass": not failures,
        "original_campaign_wall_gate_pass": resource_report["campaign"]["original_wall_pass"],
        "performance": performance,
        "repeat_failures": failures,
        "resources": resource_report,
    }


def table(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def verify(root, source, output):
    require(not output.exists(), "Preserve previous analysis")
    pinned_code(source)
    resource_report = resources(root)
    recovery_report = recovery(root)
    auth = read(root / "runs/authorization.json")
    dependencies = auth["identity"]["plans"]["reference"]["fingerprints"]["dependencies"]
    require(set(dependencies) == {"python", *run_plan.DEPENDENCIES}, "Incomplete dependencies")
    parent = read(root / "binding/migration/binding.json")
    require(parent["parent"]["sha256"] == PARENT, "Wrong provisional parent")
    for name, digest in parent["preparation_source"]["files"].items():
        require(sha(source / name) == digest, "Migration source differs")
    failures = []
    artifacts = []
    with tempfile.TemporaryDirectory(prefix="issue163-verify-") as scratch:
        relocated = Path(scratch) / "binding"
        with patch.object(historical.mask, "ARMS", study.ARMS):
            configurations = historical.relocate(root / "binding", relocated)
        with (
            patch.object(study, "ROOT", source),
            patch.object(study, "CONFIG", source / CONFIG_REL),
            patch.object(study.campaign, "ROOT", source),
            patch.object(run_plan, "REPOSITORY_ROOT", source),
            patch.object(run_plan, "_dependency_record", lambda: dict(dependencies)),
            patch.object(run_plan, "_fingerprint_paths", lambda names: fingerprint(source, names)),
            patch.object(
                run_plan,
                "_fingerprint_directory",
                lambda path: fingerprint(source, (str(path),), True),
            ),
        ):
            config, plans, report = study.validate(relocated)
            for arm, plan in list(plans.items()):
                plans[arm] = replace(
                    plan, fingerprints={**plan.fingerprints, "configuration": configurations[arm]}
                )

            def validator(*_):
                return config, plans, report

            config, rows, training, manifest, _, _ = analysis.load_evidence(
                root / "runs",
                root / "binding",
                validator=validator,
                config_path=source / CONFIG_REL,
                issue=163,
                repeat_failures=failures,
            )
            from agent_code.DagobertDuckDQNTask3.persistence import load_training_checkpoint

            for arm, plan in plans.items():
                directory = root / "runs/plans" / plan.plan_id
                status = read(directory / "status.json")
                require(
                    read(directory / "resolved_plan.json")["fingerprints"] == plan.fingerprints,
                    "Source fingerprint mismatch",
                )
                for job in plan.jobs:
                    record = status["jobs"][job.run_id]
                    meta = read(directory / record["attempts"][-1]["output"] / "metadata.json")
                    require(
                        meta["run_plan"]["fingerprints"] == plan.fingerprints,
                        "Job fingerprints differ",
                    )
                    require(
                        meta["logging_policy"] == "warning_only"
                        and meta["snapshot_storage"] == config["storage_policy"],
                        "Logging/storage policy differs",
                    )
                    if job.kind == "training":
                        path = directory / record["artifact"]["path"]
                        loaded = load_training_checkpoint(path)
                        require(
                            loaded.completed_episodes == 5000
                            and loaded.config.neutral_safe_attack_bombs == (arm == "neutral")
                            and not loaded.config.double_dqn
                            and loaded.config.action_masking,
                            "Checkpoint intervention differs",
                        )
                        artifacts.append(
                            {
                                "arm": arm,
                                "replica": job.replica,
                                "sha256": sha(path),
                                "size_bytes": path.stat().st_size,
                                "episodes": loaded.completed_episodes,
                                "path": path.relative_to(root).as_posix(),
                            }
                        )
            performance = study.decide(rows, config)
    paired_audit = matched_training(root, artifacts, training)
    output.mkdir(parents=True)
    with gzip.GzipFile(filename=str(output / "observations.json.gz"), mode="wb", mtime=0) as file:
        file.write(
            json.dumps(
                {"evaluation": rows, "training": training}, sort_keys=True, separators=(",", ":")
            ).encode()
        )
    result = blocked_decision(performance, failures, resource_report)
    result.update(
        execution_source=SOURCE,
        protocol_sha256=sha(source / CONFIG_REL),
        training_episodes=len(training),
        evaluation_episodes=len(rows),
        completed_jobs=sum(len(p.jobs) for p in plans.values()),
        recovery=recovery_report,
        artifacts=artifacts,
        execution_dependencies=dependencies,
        observations_sha256=sha(output / "observations.json.gz"),
        matched_training=paired_audit,
    )
    write(output / "result.json", result)
    write(output / "source-manifest.json", manifest)
    write(output / "repeat-failures.json", failures)
    write(output / "matched-training.json", paired_audit)
    table(
        output / "evaluation.csv",
        [{k: v for k, v in row.items() if k != "decision_times_ms"} for row in rows],
    )
    tables = {"summary": [], "replicas": [], "gates": [], "contrasts": []}
    for arm, record in performance["arms"].items():
        tables["gates"] += [
            {"arm": arm, "gate": k, "passed": v} for k, v in record["gates"].items()
        ]
        tables["contrasts"] += [
            {"arm": arm, "contrast": k, **v} for k, v in record["contrasts"].items()
        ]
        for suite, groups in record["summaries"].items():
            for group, data in groups.items():
                if group == "per_model":
                    tables["replicas"] += [
                        {
                            "arm": arm,
                            "suite": suite,
                            "replica": key,
                            **{k: v for k, v in val.items() if k != "action_counts"},
                        }
                        for key, val in data.items()
                    ]
                elif group == "candidate" or arm == "control":
                    tables["summary"].append(
                        {
                            "arm": arm if group == "candidate" else "reference",
                            "suite": suite,
                            **{k: v for k, v in data.items() if k != "action_counts"},
                        }
                    )
    tables["gates"] += [
        {"arm": "neutral", "gate": "treatment_benefit", "passed": performance["treatment_gate"]},
        {"arm": "all", "gate": "deterministic_repeats", "passed": not failures},
        {
            "arm": "all",
            "gate": "original_campaign_wall",
            "passed": resource_report["campaign"]["original_wall_pass"],
        },
    ]
    for name, data in tables.items():
        table(output / (name + ".csv"), data)
    print(
        json.dumps(
            {
                "decision": result["status"],
                "effect": performance["neutral_minus_control_elimination"],
                "repeat_failures": failures,
                "summaries": tables["summary"],
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.root.resolve(), args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
