"""Reproduce Issue #107 with its exact source and explicit cross-platform provenance.

Run this script from a clean checkout of the recorded execution revision. The
script may reside in a newer result branch. It adapts only plan construction's
host-dependent directory ordering and installed-package inventory; execution
metadata remains immutable and the registered analyzer is used unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import subprocess
import sys
import tarfile
from dataclasses import replace
from functools import lru_cache
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    path = Path(path)
    stat = path.stat()
    return _cached_digest(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=256)
def _cached_digest(path, size, modified):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    stat = Path(path).stat()
    if (stat.st_size, stat.st_mtime_ns) != (size, modified):
        raise ValueError("Evidence changed during checksum verification")
    return result.hexdigest()


def linux_directory_digest(path):
    """Match the reviewed runner's PosixPath ordering, including directory parts."""
    path = Path(path)
    result = hashlib.sha256()
    for item in sorted(path.rglob("*"), key=lambda p: tuple(p.relative_to(path).parts)):
        if (
            not item.is_file()
            or {"__pycache__", "logs"}.intersection(item.relative_to(path).parts)
            or item.name == ".evaluation-checkpoint.pt"
            or item.suffix == ".pyc"
        ):
            continue
        result.update(item.relative_to(path).as_posix().encode() + b"\0")
        result.update(item.read_bytes() + b"\0")
    return result.hexdigest()


def check_dependency_record(fingerprints):
    record = fingerprints["dependencies"]
    expected = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    if expected != fingerprints["dependencies_sha256"]:
        raise ValueError("Execution dependency inventory checksum mismatch")
    return record


def contained(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Evidence path escapes its plan directory: {relative}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-result", type=Path, required=True)
    parser.add_argument("--archive", type=Path, help="Package only audited evidence files")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    plan_root = args.plan_root.resolve()
    authorization_paths = list(plan_root.parent.glob("issue107-campaign-authorization-*.json"))
    if len(authorization_paths) != 1:
        raise ValueError("Expected one authorization record")
    authorization = read_json(authorization_paths[0])
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if revision != authorization["reviewed_commit"]:
        raise ValueError("Run from the exact execution revision")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Execution checkout must be clean")
    sys.path.insert(0, str(root))
    runner = importlib.import_module("training.run_plan")
    analyzer = importlib.import_module("training.analyze_issue107_task2_factorial")
    campaign = importlib.import_module("training.run_issue107_campaign")
    expected_campaign = analyzer._validate_campaign_execution(plan_root)
    native_load_plan = runner.load_plan
    execution_dependencies = None
    final_artifacts = []
    job_count = 0
    failed_attempts = []
    file_records = {}

    def record(path):
        relative = path.relative_to(plan_root.parent).as_posix()
        file_records[relative] = {"sha256": digest(path), "size_bytes": path.stat().st_size}

    for path in plan_root.parent.glob("issue107-campaign-*.json"):
        record(path)
    for cell, plan_path in campaign.PLAN_PATHS.items():
        directory = plan_root / plan_path.stem
        resolved = read_json(directory / "resolved_plan.json")
        status = read_json(directory / "status.json")
        dependencies = check_dependency_record(resolved["fingerprints"])
        if execution_dependencies is None:
            execution_dependencies = dependencies
        if dependencies != execution_dependencies:
            raise ValueError("Plans used different execution dependencies")
        if status.get("status") != "completed":
            raise ValueError(f"Incomplete plan: {cell}")
        jobs = {job["run_id"]: job for job in resolved["jobs"]}
        if set(jobs) != set(status["jobs"]):
            raise ValueError(f"Job matrix mismatch: {cell}")
        record(directory / "resolved_plan.json")
        record(directory / "status.json")
        for job_id, job in status["jobs"].items():
            job_count += 1
            if job.get("status") != "completed" or not job.get("attempts"):
                raise ValueError(f"Incomplete job: {cell}/{job_id}")
            for attempt in job["attempts"]:
                metadata_path = contained(directory, attempt["metadata"])
                metadata = read_json(metadata_path)
                record(metadata_path)
                analyzer._validate_job_campaign(metadata, expected_campaign, job_id)
                if metadata.get("git_dirty") is not False:
                    raise ValueError(f"Dirty or unknown job source: {job_id}")
                if metadata["run_plan"]["fingerprints"] != resolved["fingerprints"]:
                    raise ValueError(f"Job fingerprint mismatch: {job_id}")
                if metadata.get("status") != "completed":
                    failed_attempts.append(
                        {"cell": cell, "job": job_id, "metadata": str(metadata_path)}
                    )
            if metadata.get("status") != "completed" or metadata.get("return_code") != 0:
                raise ValueError(f"Last attempt did not complete: {job_id}")
            csv_path = contained(directory, job["attempts"][-1]["output"]) / "episodes.csv"
            rows = analyzer.read_episodes_csv(csv_path)
            if len(rows) != jobs[job_id]["rounds"]:
                raise ValueError(f"Episode count mismatch: {job_id}")
            record(csv_path)
            artifact = job.get("artifact")
            if artifact:
                artifact_path = contained(directory, artifact["path"])
                if digest(artifact_path) != artifact["sha256"]:
                    raise ValueError(f"Artifact checksum mismatch: {job_id}")
                record(artifact_path)
                if (
                    jobs[job_id]["kind"] == "training"
                    and jobs[job_id]["stage_or_suite"] == "classic-crates"
                ):
                    final_artifacts.append(
                        {"cell": cell, "replica": jobs[job_id]["replica"], **artifact}
                    )
    if len(final_artifacts) != 20:
        raise ValueError("Expected twenty verified final training artifacts")

    def portable_load_plan(path):
        plan = native_load_plan(path)
        fingerprints = dict(plan.fingerprints)
        fingerprints["agent"] = linux_directory_digest(root / "agent_code" / plan.agent)
        fingerprints["dependencies"] = execution_dependencies
        fingerprints["dependencies_sha256"] = hashlib.sha256(
            json.dumps(execution_dependencies, sort_keys=True).encode()
        ).hexdigest()
        return replace(plan, fingerprints=fingerprints)

    # Preserve every registered plan field and validate source, framework,
    # agent bytes and configuration against the execution checkout. Installed
    # packages on the analysis host are separately recorded below, not presented
    # as packages that produced the training/evaluation observations.
    analyzer.load_plan = portable_load_plan
    campaign.load_plan = portable_load_plan
    result = analyzer.analyze(plan_root, args.output)
    expected = read_json(args.expected_result)
    if result != expected:
        raise ValueError("Recomputed result differs from server result")
    audit = {
        "execution_revision": revision,
        "execution_dependencies": execution_dependencies,
        "analysis_dependencies": runner._dependency_record(),
        "analysis_platform": platform.platform(),
        "jobs_verified": job_count,
        "final_artifacts": final_artifacts,
        "failed_attempts": failed_attempts,
        "exact_server_result_match": True,
        "analysis_valid": result["analysis_valid"],
        "files": file_records,
    }
    (args.output / "verification.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    if args.archive:
        with tarfile.open(args.archive, "w:gz") as archive:
            for relative in sorted(file_records):
                archive.add(plan_root.parent / relative, arcname=relative, recursive=False)
        print(
            json.dumps(
                {
                    "archive": args.archive.name,
                    "sha256": digest(args.archive),
                    "size_bytes": args.archive.stat().st_size,
                }
            )
        )
    print(
        json.dumps(
            {k: audit[k] for k in ("jobs_verified", "exact_server_result_match", "analysis_valid")}
        )
    )


if __name__ == "__main__":
    main()
