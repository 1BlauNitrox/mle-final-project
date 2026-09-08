"""Export a completed run plan's per-episode data as committable evidence.

The full raw output tree of a run plan (per-job attempt directories,
including a full agent-source snapshot in every job) can run to gigabytes
and is not meant to be committed. What a reviewer actually needs to
independently re-derive an experiment's compact result -- the per-episode
rows the registered analyzer reads -- fits in a few megabytes once every
job's episodes.csv is concatenated and identifying columns (which plan,
job, replica, suite/stage) are added. This produces exactly that, plus a
manifest recording the plan's own fingerprints and per-job provenance, so
the evidence can be checked for internal consistency without needing the
raw tree or an external archive at all.

Mirrors the naming from earlier experiments' `evidence/` directories
(e.g. `experiments/2026-09-01-dqn-task1-development-baseline/evidence/`):
`training-episodes.csv.gz`, `evaluation-episodes.csv`, `manifest.json`.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from training.aggregate import read_episodes_csv

EXTRA_COLUMNS = (
    "plan_id",
    "run_id",
    "kind",
    "replica",
    "stage_or_suite",
    "world_seed",
    "agent_seed",
)


def export_plan_evidence(
    plan_directory: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Write training/evaluation evidence CSVs and a manifest for one plan."""
    plan_directory = Path(plan_directory).resolve()
    output_directory = Path(output_directory).resolve()
    status = _read_json(plan_directory / "status.json")
    resolved = _read_json(plan_directory / "resolved_plan.json")
    if status.get("status") != "completed":
        raise ValueError(f"Run plan is not completed: {plan_directory}")

    training_rows: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    job_manifest: list[dict[str, Any]] = []

    jobs_by_id = {job["run_id"]: job for job in resolved["jobs"]}
    for run_id, job_status in sorted(status["jobs"].items()):
        job = jobs_by_id[run_id]
        attempts = job_status.get("attempts", [])
        if job_status.get("status") != "completed" or not attempts:
            raise ValueError(f"Job is not complete: {run_id}")
        attempt = attempts[-1]
        run_directory = plan_directory / attempt["output"]
        metadata = _read_json(run_directory / "metadata.json")
        episode_rows = read_episodes_csv(run_directory / "episodes.csv")

        tag = {
            "plan_id": resolved["plan_id"],
            "run_id": run_id,
            "kind": job["kind"],
            "replica": job["replica"],
            "stage_or_suite": job["stage_or_suite"],
            "world_seed": job["world_seed"],
            "agent_seed": job["agent_seed"],
        }
        tagged_rows = [{**tag, **row} for row in episode_rows]
        (training_rows if job["kind"] == "training" else evaluation_rows).extend(tagged_rows)

        job_manifest.append(
            {
                "run_id": run_id,
                "kind": job["kind"],
                "replica": job["replica"],
                "stage_or_suite": job["stage_or_suite"],
                "world_seed": job["world_seed"],
                "agent_seed": job["agent_seed"],
                "attempts": len(attempts),
                "status": job_status["status"],
                "git_commit": metadata.get("git_commit"),
                "git_dirty": metadata.get("git_dirty"),
                "started_at": metadata.get("started_at"),
                "duration_seconds": metadata.get("duration_seconds"),
                "episode_rows": len(episode_rows),
            }
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    training_path = output_directory / "training-episodes.csv.gz"
    evaluation_path = output_directory / "evaluation-episodes.csv"
    _write_gzipped_csv(training_path, training_rows)
    _write_plain_csv(evaluation_path, evaluation_rows)

    manifest = {
        "schema_version": 1,
        "plan_id": resolved["plan_id"],
        "action_masking": resolved.get("action_masking"),
        "reward_variant": resolved.get("reward_variant"),
        "fingerprints": resolved["fingerprints"],
        "jobs": job_manifest,
        "evidence_files": {
            "training-episodes.csv.gz": _file_record(training_path),
            "evaluation-episodes.csv": _file_record(evaluation_path),
        },
    }
    manifest_path = output_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return list(EXTRA_COLUMNS)
    ordered = list(EXTRA_COLUMNS)
    ordered.extend(key for key in rows[0] if key not in EXTRA_COLUMNS)
    return ordered


def _write_plain_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_fieldnames(rows), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_gzipped_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_fieldnames(rows), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _file_record(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def verify_evidence_files(evidence_directory: Path, manifest: dict[str, Any]) -> None:
    """Raise if any evidence file's on-disk size/SHA-256 differs from its manifest record.

    A manifest is fingerprinted at export time, before Git's own clean/smudge
    filters (e.g. a `text eol=lf` .gitattributes rule) can rewrite a committed
    text file's bytes -- so an analyzer that trusts the manifest without this
    check can silently validate against evidence a reviewer's checkout does
    not actually contain.
    """
    for filename, recorded in manifest["evidence_files"].items():
        path = evidence_directory / filename
        if not path.is_file():
            raise ValueError(f"Evidence file listed in manifest is missing: {path}")
        actual = _file_record(path)
        if actual != recorded:
            raise ValueError(
                f"Evidence file does not match its manifest record: {path} "
                f"(manifest: {recorded}, actual: {actual})"
            )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = export_plan_evidence(args.plan_directory, args.output)
    print(
        json.dumps(
            {k: v for k, v in manifest.items() if k != "jobs"},
            indent=2,
            sort_keys=True,
        )
    )
    print(f"{len(manifest['jobs'])} jobs recorded in manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
