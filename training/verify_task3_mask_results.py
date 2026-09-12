"""Reproduce Issue 147 from immutable server evidence without launching games."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import yaml

from training import run_plan
from training import task3_mask_campaign as mask
from training.verify_task3_results import extract_files, fingerprint

EXECUTION = "15bedeed078d073e4acefea378e1a06cd41ca7df"


def verify_export(archive_path, manifest_path, destination):
    """Verify the archive and exact member inventory; never overwrite an import."""
    manifest = mask.read_json(manifest_path)
    mask.require(
        mask.sha256(archive_path) == manifest["sha256"]
        and archive_path.stat().st_size == manifest["size_bytes"],
        "Export checksum or size mismatch",
    )
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive_path) as archive:
        extract_files(archive, destination)
    files = {p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file()}
    mask.require(files == set(manifest["files"]), "Export member inventory mismatch")
    for name, record in manifest["files"].items():
        path = mask.campaign.relative_file(destination, name)
        mask.require(
            mask.sha256(path) == record["sha256"] and path.stat().st_size == record["size_bytes"],
            "Export member checksum or size mismatch",
        )


def relocate(source, destination):
    """Check original files before relocating only absolute parent paths in scratch."""
    binding = mask.read_json(source / "binding.json")
    shutil.copytree(source, destination)
    fingerprints = {}
    for arm in mask.ARMS:
        record = binding["artifacts"][arm]
        artifact = mask.campaign.relative_file(source, record["path"])
        mask.require(
            mask.sha256(artifact) == record["sha256"]
            and artifact.stat().st_size == record["size_bytes"],
            "Original artifact changed",
        )
        record = binding["plans"][arm]
        path = mask.campaign.relative_file(source, record["path"])
        mask.require(mask.sha256(path) == record["sha256"], "Original plan changed")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        fingerprints[arm] = hashlib.sha256(
            json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        for replica in data["replicas"]:
            replica["parent_artifact"] = str(destination / binding["artifacts"][arm]["path"])
        target = destination / record["path"]
        target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8", newline="\n")
        record["sha256"] = mask.sha256(target)
    mask.write_json(destination / "binding.json", binding)
    return fingerprints


def verify(root, binding, output):
    """Validate the executed source/dependencies, then run the registered raw analyzer."""
    root, binding, output = (p.resolve() for p in (root, binding, output))
    auth = mask.read_json(root / "authorization.json")
    mask.require(auth["identity"]["reviewed_commit"] == EXECUTION, "Unsupported execution SHA")
    dependencies = auth["identity"]["plans"]["reference"]["fingerprints"]["dependencies"]
    mask.require(set(dependencies) == {"python", *run_plan.DEPENDENCIES}, "Missing dependencies")
    with tempfile.TemporaryDirectory(prefix="issue147-verification-") as temporary:
        temporary = Path(temporary)
        snapshot = temporary / "source"
        snapshot.mkdir()
        source_tar = temporary / "source.tar"
        subprocess.run(
            ["git", "archive", "--format=tar", "-o", str(source_tar), EXECUTION],
            cwd=mask.ROOT,
            check=True,
        )
        with tarfile.open(source_tar) as archive:
            extract_files(archive, snapshot)
        for name in (
            "training/task3_mask_campaign.py",
            "training/analyze_task3_campaign.py",
            "training/run_task3_campaign.py",
            "training/run_plan.py",
            "training/metrics.py",
        ):
            mask.require(
                (mask.ROOT / name).read_bytes().replace(b"\r\n", b"\n")
                == (snapshot / name).read_bytes(),
                "Historical verifier requires unchanged registered analysis code",
            )
        originals = relocate(binding, temporary / "binding")
        migration = mask.read_json(binding / "migration/binding.json")
        mask.require(
            migration["preparation_source"]["commit"] == EXECUTION, "Wrong migration source"
        )
        for name, digest in migration["preparation_source"]["files"].items():
            mask.require(mask.sha256(snapshot / name) == digest, "Migration source changed")
        config_path = snapshot / "experiments/2026-09-12-task3-legal-mask/config.yaml"
        validator = mask.validate
        with (
            patch.object(mask, "ROOT", snapshot),
            patch.object(mask, "CONFIG", config_path),
            patch.object(mask.campaign, "ROOT", snapshot),
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
            config, plans, report = validator(temporary / "binding")
            for arm, plan in plans.items():
                expected = dict(plan.fingerprints, configuration=originals[arm])
                plans[arm] = replace(plan, fingerprints=expected)
                directory = root / "plans" / plan.plan_id
                resolved = mask.read_json(directory / "resolved_plan.json")
                mask.require(resolved["fingerprints"] == expected, "Executed fingerprints differ")
                status = mask.read_json(directory / "status.json")
                for job in plan.jobs:
                    record = status["jobs"][job.run_id]
                    run = mask.campaign.relative_file(directory, record["attempts"][-1]["output"])
                    meta = mask.read_json(run / "metadata.json")
                    mask.require(
                        meta["run_plan"]["fingerprints"] == expected, "Job fingerprints differ"
                    )
                    mask.require(
                        meta["python_version"].startswith(dependencies["python"]), "Wrong Python"
                    )
                    if job.kind == "training":
                        from agent_code.DagobertDuckDQNTask3.persistence import (
                            load_training_checkpoint,
                        )

                        checkpoint = load_training_checkpoint(
                            mask.campaign.relative_file(directory, record["artifact"]["path"])
                        )
                        mask.require(
                            checkpoint.completed_episodes == 10000, "Wrong final episode count"
                        )
                        mask.require(
                            checkpoint.config.discount_factor == 0.9
                            and checkpoint.config.escape_continuation_features
                            and checkpoint.config.action_masking == (arm == "masked"),
                            "Wrong trained configuration",
                        )
            with patch.object(mask, "validate", lambda *_: (config, plans, report)):
                return mask.analyze(root, binding, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path)
    parser.add_argument("--binding-dir", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--extract-to", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.archive:
        if not args.manifest or not args.extract_to or args.campaign_root or args.binding_dir:
            parser.error("Use --archive with --manifest and --extract-to only")
        verify_export(args.archive, args.manifest, args.extract_to)
        args.campaign_root = args.extract_to / "campaign"
        args.binding_dir = args.extract_to / "binding"
    elif not args.campaign_root or not args.binding_dir or args.manifest or args.extract_to:
        parser.error("Provide --campaign-root and --binding-dir, or an archive import")
    result = verify(args.campaign_root, args.binding_dir, args.output)
    print(json.dumps({k: result[k] for k in ("status", "selected_arm", "selected_replica")}))


if __name__ == "__main__":
    main()
