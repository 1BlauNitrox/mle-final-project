"""Verify the historical Issue 109 campaign without launching games or rewriting evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path, PurePosixPath
from unittest.mock import patch

import yaml

from training import analyze_task3_campaign as analysis
from training import run_plan
from training import run_task3_campaign as campaign

EXECUTION = "0c9b0c1add52e53269646a1b2ef2aec96fb297dd"
PARENT = "c0b6081acf5a733258005d46b9430f3ef50e403470a3bb5428661729f9aeedf7"


def extract_files(archive, directory):
    """Extract regular files only; reject traversal, duplicates and links."""
    seen = set()
    for member in archive:
        name = PurePosixPath(member.name)
        campaign.require(not name.is_absolute() and ".." not in name.parts, "Unsafe archive path")
        if member.isdir():
            continue
        campaign.require(member.isfile() and name not in seen, "Archive link or duplicate")
        seen.add(name)
        target = campaign.relative_file(directory, name.as_posix())
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.extractfile(member) as source, target.open("xb") as output:
            shutil.copyfileobj(source, output)


def fingerprint(root, paths, *, directory=False):
    """Match the executed Linux path ordering and Git blob bytes on every host."""
    digest = hashlib.sha256()
    for name in paths:
        path = root / name
        files = path.rglob("*" if directory else "*.py") if path.is_dir() else [path]
        for file in sorted(files, key=lambda p: p.relative_to(root).as_posix()):
            if not file.is_file() or "__pycache__" in file.parts:
                continue
            relative = file.relative_to(path if directory else root)
            if directory and (
                "logs" in relative.parts
                or file.suffix == ".pyc"
                or file.name == run_plan.STAGED_EVALUATION_CHECKPOINT_NAME
            ):
                continue
            digest.update(relative.as_posix().encode() + b"\0" + file.read_bytes() + b"\0")
    return digest.hexdigest()


def rebind_copy(source, destination):
    """Verify original binding bytes, then relocate only parent paths in a scratch copy."""
    binding = campaign.read_json(source / "binding.json")
    campaign.require(binding["parent"]["sha256"] == PARENT, "Wrong registered #91 parent")
    campaign.require(
        binding["parent"]["source_commit"] == "cbd52be8392f5003a91c5600fda4efd544b48ec5",
        "Wrong parent source",
    )
    destination.mkdir()
    for key in ("parent", "successor"):
        record = binding[key]
        path = campaign.relative_file(source, record["path"])
        campaign.require(
            campaign.sha256(path) == record["sha256"]
            and path.stat().st_size == record["size_bytes"],
            "Binding artifact changed",
        )
        shutil.copyfile(path, destination / record["path"])
    original = {}
    for arm in ("candidate", "reference"):
        record = binding["plans"][arm]
        path = campaign.relative_file(source, record["path"])
        campaign.require(campaign.sha256(path) == record["sha256"], "Bound YAML changed")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        original[arm] = hashlib.sha256(
            json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        key = "successor" if arm == "candidate" else "parent"
        for replica in data["replicas"]:
            recorded = PurePosixPath(replica["parent_artifact"])
            campaign.require(recorded.name == binding[key]["path"], "Wrong bound parent path")
            replica["parent_artifact"] = str(destination / binding[key]["path"])
        target = destination / record["path"]
        target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8", newline="\n")
        record["sha256"] = campaign.sha256(target)
    (destination / "binding.json").write_text(json.dumps(binding), encoding="utf-8")
    return original


@contextmanager
def historical_context(root, binding_directory):
    """Validate historical sources/dependencies separately from the analysis environment."""
    auth = campaign.read_json(root / "authorization.json")
    campaign.require(auth["identity"]["reviewed_commit"] == EXECUTION, "Unsupported execution SHA")
    dependencies = auth["identity"]["plans"]["reference"]["fingerprints"]["dependencies"]
    campaign.require(
        set(dependencies) == {"python", *run_plan.DEPENDENCIES},
        "Incomplete execution dependency record",
    )
    with tempfile.TemporaryDirectory(prefix="task3-verification-") as temporary:
        temporary = Path(temporary)
        snapshot = temporary / "source"
        snapshot.mkdir()
        archive_path = temporary / "source.tar"
        subprocess.run(
            ["git", "archive", "--format=tar", "-o", str(archive_path), EXECUTION],
            cwd=campaign.ROOT,
            check=True,
        )
        with tarfile.open(archive_path) as archive:
            extract_files(archive, snapshot)
        original_fingerprints = rebind_copy(binding_directory, temporary / "binding")
        binding = campaign.read_json(binding_directory / "binding.json")
        campaign.require(
            binding["preparation_source"]["commit"] == EXECUTION, "Preparation source mismatch"
        )
        for name, digest in binding["preparation_source"]["files"].items():
            campaign.require(
                campaign.sha256(campaign.relative_file(snapshot, name)) == digest,
                "Preparation source bytes mismatch",
            )
        config_path = snapshot / "experiments/2026-09-08-dqn-task3-peaceful-opponent/config.yaml"
        original_validator = campaign.validate_protocol
        with (
            patch.object(campaign, "ROOT", snapshot),
            patch.object(campaign, "CONFIG", config_path),
            patch.object(analysis, "CONFIG", config_path),
            patch.object(run_plan, "REPOSITORY_ROOT", snapshot),
            patch.object(run_plan, "_dependency_record", lambda: dict(dependencies)),
            patch.object(
                run_plan, "_fingerprint_paths", lambda names: fingerprint(snapshot, names)
            ),
            patch.object(
                run_plan,
                "_fingerprint_directory",
                lambda path: fingerprint(snapshot, (str(path),), directory=True),
            ),
        ):
            config, plans, report = original_validator(temporary / "binding")
            for arm, plan in plans.items():
                expected = dict(plan.fingerprints, configuration=original_fingerprints[arm])
                plans[arm] = replace(plan, fingerprints=expected)
                directory = root / "plans" / plan.plan_id
                resolved = campaign.read_json(directory / "resolved_plan.json")
                campaign.require(
                    resolved["fingerprints"] == expected,
                    "Execution source/dependency fingerprints mismatch",
                )
                status = campaign.read_json(directory / "status.json")
                for job in plan.jobs:
                    record = status["jobs"][job.run_id]
                    run = campaign.relative_file(directory, record["attempts"][-1]["output"])
                    meta = campaign.read_json(run / "metadata.json")
                    campaign.require(
                        meta["run_plan"]["fingerprints"] == expected,
                        "Job source/dependency fingerprints mismatch",
                    )
                    campaign.require(
                        meta["python_version"].startswith(dependencies["python"]),
                        "Job Python version mismatch",
                    )
                for replica in plan.replicas:
                    if arm == "candidate":
                        from agent_code.DagobertDuckDQNTask3.persistence import (
                            load_training_checkpoint,
                        )

                        jobs = [
                            j
                            for j in plan.jobs
                            if j.kind == "training" and j.replica == replica.replica_id
                        ]
                        artifact = status["jobs"][jobs[-1].run_id]["artifact"]
                        loaded = load_training_checkpoint(
                            campaign.relative_file(directory, artifact["path"])
                        )
                        campaign.require(
                            loaded.completed_episodes == 10000,
                            "Wrong final checkpoint training budget",
                        )
                        campaign.require(
                            loaded.config.discount_factor == 0.9
                            and loaded.config.escape_continuation_features
                            and not loaded.config.action_masking,
                            "Wrong trained checkpoint configuration",
                        )
            with patch.object(analysis, "validate_protocol", lambda *_: (config, plans, report)):
                yield


def verify(root, binding_directory, output):
    with historical_context(root.resolve(), binding_directory.resolve()):
        return analysis.analyze(root.resolve(), binding_directory.resolve(), output.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--binding-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.campaign_root, args.binding_dir, args.output)
    print(
        json.dumps({key: result[key] for key in ("status", "selected_replica", "gates")}, indent=2)
    )


if __name__ == "__main__":
    main()
