"""Create the compact external evidence archive for Issue #110."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from training.run_experiment import REPOSITORY_ROOT

PLAN_IDS = (
    "issue110-tabular-task1-frozen",
    "issue110-tabular-task2-unmasked",
    "issue110-tabular-task2-masked",
)

EXPERIMENT_DIRECTORY = (
    REPOSITORY_ROOT
    / "experiments"
    / "2026-09-07-task2-legal-action-masking-DerKleineSprengstoffkapitalist"
)

PLAN_ROOT = REPOSITORY_ROOT / "training_outputs" / "run-plans"
ARCHIVE_PATH = REPOSITORY_ROOT / "issue110-evidence-v1.tar.gz"
ARCHIVE_ROOT = "issue110-evidence-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")

    return value


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def package_plan(plan_id: str, staging_root: Path) -> None:
    source = PLAN_ROOT / plan_id
    destination = staging_root / "training_outputs" / "run-plans" / plan_id

    status = read_json(source / "status.json")

    if status.get("status") != "completed":
        raise ValueError(f"Plan is not completed: {plan_id}")

    copy_file(
        source / "status.json",
        destination / "status.json",
    )
    copy_file(
        source / "resolved_plan.json",
        destination / "resolved_plan.json",
    )

    for model_path in sorted((source / "replicas").glob("*/agent/model.npz")):
        relative = model_path.relative_to(source)
        copy_file(model_path, destination / relative)

    for run_id, job in status["jobs"].items():
        if not run_id.startswith("eval-"):
            continue

        attempts = job.get("attempts", [])

        if job.get("status") != "completed" or not attempts:
            raise ValueError(f"Evaluation job is incomplete: {plan_id}/{run_id}")

        output = Path(attempts[-1]["output"])

        for filename in ("metadata.json", "episodes.csv"):
            copy_file(
                source / output / filename,
                destination / output / filename,
            )


def write_manifest(staging_root: Path) -> None:
    entries = []

    for path in sorted(staging_root.rglob("*")):
        if not path.is_file():
            continue

        entries.append(
            {
                "path": path.relative_to(staging_root).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )

    manifest = {
        "schema_version": 1,
        "issue": 110,
        "description": (
            "Compact primary and repeat evaluation evidence, "
            "resolved plans, statuses and evaluated model artifacts."
        ),
        "files": entries,
    }

    (staging_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="issue110-evidence-") as temporary_directory:
        staging_root = Path(temporary_directory) / ARCHIVE_ROOT
        staging_root.mkdir(parents=True)

        for plan_id in PLAN_IDS:
            package_plan(plan_id, staging_root)

        result_destination = staging_root / "experiments" / EXPERIMENT_DIRECTORY.name

        for filename in (
            "evidence.csv",
            "summary.csv",
            "result.json",
            "config.yaml",
            "README.md",
        ):
            copy_file(
                EXPERIMENT_DIRECTORY / filename,
                result_destination / filename,
            )

        write_manifest(staging_root)

        with tarfile.open(ARCHIVE_PATH, "w:gz") as archive:
            archive.add(
                staging_root,
                arcname=ARCHIVE_ROOT,
            )

    print(f"Archive: {ARCHIVE_PATH}")
    print(f"SHA-256: {sha256_file(ARCHIVE_PATH)}")
    print(f"Size: {ARCHIVE_PATH.stat().st_size} bytes")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
