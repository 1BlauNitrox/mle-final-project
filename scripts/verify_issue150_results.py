"""Verify Issue 150 downloads and reproduce its registered analysis without games."""

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from unittest.mock import patch

EXECUTION = "6f014485a3026cc3707fa2cc3a379880dd0b74bd"
ARCHIVE_SHA = "c40f4b934fc77607c6f6f0a05a139f96818943671dc82040c2a710495a73da4f"
RECOVERY_SHA = "a958a7856dcb3fdf0787699817ff2f85d218397b1df7c521ea82ddad2531b980"
HELPERS = {
    "recover.py": "e82606b17dd075067591cf388ed2822c14c75bd75907cc79fc2449d0d4902c50",
    "resume.py": "fb19a744da9af849f8f291e1fbc921dafc8a5c26e0b25c2975c06d54ab1ceb62",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def extract(archive, destination):
    """Reject links, duplicates and cross-platform path traversal before file writes."""
    seen = set()
    for entry in archive:
        name = PurePosixPath(entry.name)
        require(
            not name.is_absolute()
            and ".." not in name.parts
            and "\\" not in entry.name
            and ":" not in entry.name,
            "Unsafe archive path",
        )
        if entry.isdir():
            continue
        require(entry.isfile() and entry.name not in seen, "Archive link or duplicate")
        seen.add(entry.name)
        target = destination / entry.name
        require(target.resolve().is_relative_to(destination.resolve()), "Escaped extraction")
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.extractfile(entry) as source, target.open("xb") as output:
            shutil.copyfileobj(source, output)


def verify_tree(root, records):
    files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    require(files == set(records), "Evidence member inventory mismatch")
    for name, record in records.items():
        path = root / name
        require(path.resolve().is_relative_to(root.resolve()), "Manifest path escape")
        require(
            path.stat().st_size == record["size_bytes"] and sha(path) == record["sha256"],
            "Evidence member hash/size mismatch: " + name,
        )


def verify_recovery(evidence, recovery):
    """Cross-check both journals against retained attempts, artifacts and accounting."""
    storage_files = list(recovery.glob("recovery-*.jsonl"))
    resume_files = list(recovery.glob("resume-*.jsonl"))
    require(len(storage_files) == len(resume_files) == 1, "Expected one storage and resume journal")
    storage = [json.loads(line) for line in storage_files[0].read_text().splitlines()]
    resumed = [json.loads(line) for line in resume_files[0].read_text().splitlines()]
    require(storage[0]["apply"] and storage[-1]["phase"] == "complete", "Incomplete storage audit")
    require(
        len(resumed) == 2
        and resumed[0]["phase"] == "before"
        and resumed[-1]["phase"] == "campaign_complete",
        "Incomplete resume audit",
    )
    before, completed = resumed
    require(
        before["source"] == EXECUTION and before["helper_sha256"] == HELPERS["resume.py"],
        "Resume source/helper differs",
    )
    root = evidence / "campaign"
    resources = read(root / "resources.json")
    auth = read(root / "authorization.json")
    require(completed["resources"] == resources, "Final resource state differs from journal")
    require(
        before["authorization_time"] == auth["authorized_at"]
        and before["resource_time"] == resources["authorized_at"],
        "Authorization time changed",
    )
    previous = before["retained_resources"]
    require(
        previous["limits"] == resources["limits"]
        and previous["authorized_at"] == resources["authorized_at"]
        and previous["cpu_seconds_consumed"] <= resources["cpu_seconds_consumed"]
        and previous["wall_seconds_elapsed"] <= resources["wall_seconds_elapsed"],
        "Accounting was reset or limits changed",
    )
    require(
        (evidence / "handoff/run-completed").read_text().strip() == EXECUTION,
        "Wrong completion source",
    )
    statuses = {p.parent.name: read(p) for p in (root / "plans").glob("*/status.json")}
    # execute_plan rewrites top-level status/updated_at even when every job is skipped.
    # Before-status hashes survive in both journals; the old status bytes are not exported.
    changed_records = {
        "runs/resources.json",
        *("runs/plans/" + name + "/status.json" for name in statuses),
    }
    for name, digest in before["before_sha256"].items():
        parts = PurePosixPath(name).parts
        path = root.joinpath(*parts[1:]) if parts[0] == "runs" else evidence / name
        require(path.resolve().is_relative_to(evidence.resolve()), "Resume audit path escape")
        if name not in changed_records:
            require(sha(path) == digest, "Protected resume input changed: " + name)
    prefix = PurePosixPath(storage[0]["root"])
    for name, digest in storage[0]["controls"].items():
        relative = PurePosixPath(name).relative_to(prefix).as_posix()
        if relative == "resources.json":
            original_bytes = (json.dumps(previous, indent=2, sort_keys=True) + "\n").encode()
            require(
                hashlib.sha256(original_bytes).hexdigest() == digest,
                "Storage/resume resource history differs",
            )
        elif relative in {"plans/" + name + "/status.json" for name in statuses}:
            require(
                before["before_sha256"]["runs/" + relative] == digest,
                "Storage/resume prior status hashes differ",
            )
        else:
            require(sha(root / relative) == digest, "Protected storage input changed: " + relative)
    require(len(storage) == 2 + 2 * storage[0]["replacement_paths"], "Missing storage operations")
    targets, inputs, byte_sum = set(), set(), 0

    def input_artifact(name):
        relative = PurePosixPath(name).relative_to(prefix)
        parts = relative.parts
        require(
            len(parts) == 6
            and parts[0] == "plans"
            and parts[2] == "jobs"
            and parts[4].endswith("-input-agent")
            and parts[5] == "checkpoint.pt",
            "Non-input path in recovery operation",
        )
        job = statuses[parts[1]]["jobs"][parts[3]]
        attempt_output = "/".join(parts[2:4]) + "/" + parts[4].removesuffix("-input-agent")
        require(
            job["kind"] == "evaluation"
            and job["status"] == "completed"
            and any(
                a["output"] == attempt_output and a["status"] == "completed"
                for a in job["attempts"]
            ),
            "Recovery touched incomplete/training input",
        )
        artifact = root / "plans" / parts[1] / job["artifact"]["path"]
        return job["artifact"]["sha256"], artifact.stat().st_size

    for operation, checked in zip(storage[1:-1:2], storage[2:-1:2], strict=True):
        require(
            operation["phase"] == "before"
            and checked["phase"] == "verified"
            and operation["target"] == checked["target"]
            and operation["sha256"] == checked["sha256"],
            "Unpaired storage operation",
        )
        require(operation["target"] not in targets, "Repeated recovery target")
        targets.add(operation["target"])
        for name in (operation["source"], operation["target"]):
            require(
                input_artifact(name) == (operation["sha256"], operation["size_bytes"]),
                "Recovered input differs from canonical artifact",
            )
            inputs.add(name)
        byte_sum += operation["size_bytes"]
    require(
        len(inputs) == storage[0]["verified_inputs"]
        and byte_sum == storage[0]["estimated_reclaimed_bytes"],
        "Storage count/size mismatch",
    )
    retries = [
        {"plan": name, "job": key, "attempts": job["attempts"]}
        for name, status in statuses.items()
        for key, job in status["jobs"].items()
        if len(job["attempts"]) > 1
    ]
    require(
        len(retries) == 1
        and retries[0]["attempts"][0]["status"] == "interrupted"
        and retries[0]["attempts"][1]["status"] == "completed",
        "Unexpected retry history",
    )
    def parse_time(value):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    resume_time = parse_time(auth["authorized_at"]) + timedelta(
        seconds=resources["limits"]["wall_seconds"] - before["remaining_wall_seconds"]
    )
    reconstructed = {}
    for name, status in statuses.items():
        counts = {}
        for job in status["jobs"].values():
            prior = [a for a in job["attempts"] if parse_time(a["started_at"]) <= resume_time]
            label = (
                "pending"
                if not prior
                else (
                    "completed"
                    if prior[-1]["finished_at"] is not None
                    and parse_time(prior[-1]["finished_at"]) <= resume_time
                    else "running"
                )
            )
            counts[label] = counts.get(label, 0) + 1
        reconstructed[name] = counts
    require(
        reconstructed == before["jobs"], "Retained attempts disagree with pre-resume job counts"
    )
    return {
        "storage_journal_verified": True,
        "resume_journal_verified": True,
        "verified_inputs": len(inputs),
        "replacement_paths": len(targets),
        "estimated_reclaimed_bytes": byte_sum,
        "protected_checkpoint_files": sum(name.endswith(".pt") for name in storage[0]["controls"]),
        "retries": retries,
        "resource_before_resume": previous,
        "resource_final": resources,
        "limitation": "Archived input duplicates are omitted from compact export; journals are "
        "cross-checked against retained canonical bytes, not live server inodes or permissions. "
        "Pre-resume status bytes are absent; their hashes agree across journals and prior job "
        "counts are reconstructed from retained attempt timestamps.",
    }


def worker(source, evidence, output):
    """Import the actual executed analyzer, with only path/dependency provenance adapters."""
    sys.path.insert(0, str(source))
    import yaml
    from training.verify_task3_results import fingerprint

    from training import run_plan
    from training import task3_double_campaign as double
    from training import verify_task3_mask_results as previous

    root, binding = evidence / "campaign", evidence / "binding"
    auth = read(root / "authorization.json")
    require(auth["identity"]["reviewed_commit"] == EXECUTION, "Wrong execution source")
    dependencies = auth["identity"]["plans"]["reference"]["fingerprints"]["dependencies"]
    require(
        set(dependencies) == {"python", *run_plan.DEPENDENCIES}, "Missing execution dependencies"
    )
    with tempfile.TemporaryDirectory(prefix="issue150-binding-") as temporary:
        temporary = Path(temporary)
        # Reuse the existing verified relocation algorithm with the registered arm names.
        with patch.object(previous.mask, "ARMS", double.ARMS):
            original_configs = previous.relocate(binding, temporary / "binding")
        migration = read(binding / "migration/binding.json")
        require(migration["preparation_source"]["commit"] == EXECUTION, "Wrong migration source")
        for name, digest in migration["preparation_source"]["files"].items():
            require(sha(source / name) == digest, "Migration source bytes changed")
        with (
            patch.object(run_plan, "_dependency_record", lambda: dict(dependencies)),
            patch.object(run_plan, "_fingerprint_paths", lambda names: fingerprint(source, names)),
            patch.object(
                run_plan,
                "_fingerprint_directory",
                lambda path: fingerprint(source, (str(path),), directory=True),
            ),
        ):
            config, plans, report = double.validate(temporary / "binding")
            for arm, plan in plans.items():
                expected = dict(plan.fingerprints, configuration=original_configs[arm])
                plans[arm] = replace(plan, fingerprints=expected)
                directory = root / "plans" / plan.plan_id
                require(
                    read(directory / "resolved_plan.json")["fingerprints"] == expected,
                    "Executed source/dependency/configuration fingerprints differ",
                )
                status = read(directory / "status.json")
                for job in plan.jobs:
                    record = status["jobs"][job.run_id]
                    run = double.campaign.relative_file(directory, record["attempts"][-1]["output"])
                    meta = read(run / "metadata.json")
                    require(meta["run_plan"]["fingerprints"] == expected, "Job fingerprints differ")
                    require(
                        meta["python_version"].startswith(dependencies["python"]), "Wrong Python"
                    )
            with patch.object(double, "validate", lambda *_: (config, plans, report)):
                result = double.analyze(root, binding, output)
        original = read(evidence / "analysis/result.json")
        reproduced = read(output / "result.json")
        # gzip embeds a creation timestamp; compare decoded observations, not container hashes.
        import gzip

        with gzip.open(evidence / "analysis/observations.json.gz", "rt", encoding="utf-8") as file:
            old_rows = json.load(file)
        with gzip.open(output / "observations.json.gz", "rt", encoding="utf-8") as file:
            new_rows = json.load(file)
        require(old_rows == new_rows, "Raw-derived observations differ from server analysis")
        require(
            {k: v for k, v in original.items() if k != "observations_sha256"}
            == {k: v for k, v in reproduced.items() if k != "observations_sha256"},
            "Registered result differs from server analysis",
        )
        require(
            read(evidence / "analysis/source-manifest.json")
            == read(output / "source-manifest.json"),
            "Raw evidence source manifests differ",
        )
        (output / "reproduction.json").write_text(
            json.dumps(
                {
                    "execution_source": EXECUTION,
                    "analysis_source": EXECUTION,
                    "raw_observations_equal": True,
                    "registered_result_equal": True,
                    "source_manifest_equal": True,
                    "execution_dependencies": dependencies,
                    "analysis_python": sys.version,
                    "analysis_numpy": __import__("numpy").__version__,
                    "training_episodes": sum(
                        j.rounds for p in plans.values() for j in p.jobs if j.kind == "training"
                    ),
                    "evaluation_episodes": len(new_rows["evaluation"]),
                    "decision": result,
                    "registered_bootstrap": yaml.safe_load(double.CONFIG.read_text())["bootstrap"],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result), flush=True)


def write_tables(analysis, output):
    """Export scalar observations and every registered gate without narrative interpretation."""
    import gzip

    result = read(analysis / "result.json")
    decision = {
        k: v for k, v in result.items() if k not in ("authorization", "observations_sha256")
    }
    (output / "decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    with gzip.open(analysis / "observations.json.gz", "rt", encoding="utf-8") as file:
        observations = json.load(file)

    def table(name, rows):
        fields = list(rows[0])
        with (output / name).open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    table(
        "evaluation.csv",
        [
            {k: v for k, v in row.items() if k != "decision_times_ms"}
            for row in observations["evaluation"]
        ],
    )
    summaries, per_replica, gates, contrasts = [], [], [], []
    for arm, record in result["arms"].items():
        gates.extend(
            {"arm": arm, "gate": key, "passed": value} for key, value in record["gates"].items()
        )
        contrasts.extend(
            {"arm": arm, "contrast": key, **value} for key, value in record["contrasts"].items()
        )
        for suite, groups in record["summaries"].items():
            for name, summary in groups.items():
                if name == "per_model":
                    per_replica.extend(
                        {
                            "arm": arm,
                            "suite": suite,
                            "replica": replica,
                            **{k: v for k, v in values.items() if k != "action_counts"},
                        }
                        for replica, values in summary.items()
                    )
                elif name == "candidate" or arm == "control":
                    summaries.append(
                        {
                            "arm": arm if name == "candidate" else "reference",
                            "suite": suite,
                            **{k: v for k, v in summary.items() if k != "action_counts"},
                        }
                    )
    # Hunting-only columns are unavailable on solo suites, not zero observations.
    for rows in (summaries, per_replica):
        fields = list(dict.fromkeys(key for row in rows for key in row))
        for row in rows:
            for key in fields:
                row.setdefault(key, None)
    gates.append(
        {
            "arm": "double",
            "gate": "double_minus_control_elimination",
            "passed": result["treatment_gate"],
        }
    )
    table("summary.csv", summaries)
    table("per-replica.csv", per_replica)
    table("gates.csv", gates)
    table("contrasts.csv", contrasts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--recovery", type=Path)
    parser.add_argument("--import-root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--worker-source", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--evidence-root", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_source:
        worker(args.worker_source.resolve(), args.evidence_root.resolve(), args.output.resolve())
        return
    require(
        args.archive and args.manifest and args.recovery, "Archive, manifest and recovery required"
    )
    require(not args.output.exists(), "Preserve previous verification output")
    require(
        sha(args.archive) == ARCHIVE_SHA and sha(args.recovery) == RECOVERY_SHA,
        "Wrong Issue 150 download",
    )
    manifest = read(args.manifest)
    require(
        manifest["sha256"] == ARCHIVE_SHA and manifest["size_bytes"] == args.archive.stat().st_size,
        "Manifest archive identity mismatch",
    )
    args.output.mkdir(parents=True)
    root = args.import_root or args.output / "import"
    if not args.import_root:
        root.mkdir()
        for archive, folder in ((args.archive, "evidence"), (args.recovery, "recovery")):
            destination = root / folder
            destination.mkdir()
            with tarfile.open(archive) as stream:
                extract(stream, destination)
    verify_tree(root / "evidence", manifest["files"])
    for name, digest in HELPERS.items():
        require(sha(root / "recovery" / name) == digest, "Wrong recovery helper")
    # The recovery archive is small: verify extracted journals against its exact bytes too.
    recovery_records = {}
    with tarfile.open(args.recovery) as stream:
        for entry in stream:
            require(entry.isfile(), "Unexpected recovery member")
            recovery_records[entry.name] = {
                "size_bytes": entry.size,
                "sha256": hashlib.sha256(stream.extractfile(entry).read()).hexdigest(),
            }
    verify_tree(root / "recovery", recovery_records)
    recovery_report = verify_recovery(root / "evidence", root / "recovery")
    (args.output / "recovery-verification.json").write_text(
        json.dumps(recovery_report, indent=2) + "\n", encoding="utf-8"
    )
    repo = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="issue150-source-") as temporary:
        temporary = Path(temporary)
        archive = temporary / "source.tar"
        source = temporary / "source"
        source.mkdir()
        subprocess.run(
            ["git", "archive", "--format=tar", "-o", str(archive), EXECUTION], cwd=repo, check=True
        )
        with tarfile.open(archive) as stream:
            extract(stream, source)
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker-source",
                str(source),
                "--evidence-root",
                str((root / "evidence").resolve()),
                "--output",
                str((args.output / "analysis").resolve()),
            ],
            check=True,
        )
    write_tables(args.output / "analysis", args.output)
    (args.output / "downloads.json").write_text(
        json.dumps(
            {
                "archive_sha256": ARCHIVE_SHA,
                "archive_size_bytes": args.archive.stat().st_size,
                "archive_members": len(manifest["files"]),
                "recovery_sha256": RECOVERY_SHA,
                "recovery_size_bytes": args.recovery.stat().st_size,
                "recovery_members": recovery_records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
