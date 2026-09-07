"""Package immutable Issue #114 evidence with a SHA-256 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import tempfile
from pathlib import Path

from training.run_experiment import REPOSITORY_ROOT

PLAN_IDS = (
    "issue114-tabular-task1-frozen",
    "issue114-tabular-task2-control",
    "issue114-tabular-task2-useful-bomb-reward",
    "issue114-tabular-task2-control-diagnostics-v2",
    "issue114-tabular-task2-useful-bomb-reward-diagnostics-v2",
)
EXPERIMENT = "2026-09-07-task2-useful-bomb-reward-DerKleineSprengstoffkapitalist"
ROOT_NAME = "issue114-evidence-v1"


def package(output: Path) -> dict[str, object]:
    sources: list[tuple[Path, Path]] = []
    plan_root = REPOSITORY_ROOT / "training_outputs" / "run-plans"
    for plan_id in PLAN_IDS:
        directory = plan_root / plan_id
        sources.extend(
            (directory / name, Path("run-plans") / plan_id / name)
            for name in ("status.json", "resolved_plan.json")
        )
        status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
        for run_id, job in status["jobs"].items():
            if job["status"] != "completed":
                raise ValueError(f"Incomplete evidence job: {plan_id}/{run_id}")
            attempt = job["attempts"][-1]["output"]
            if run_id.startswith("eval-") or "diagnostics" in plan_id:
                for name in ("metadata.json", "episodes.csv"):
                    source = directory / attempt / name
                    sources.append((source, Path("run-plans") / plan_id / attempt / name))
        for model in sorted(directory.glob("replicas/*/agent/model.npz")):
            sources.append((model, Path("run-plans") / plan_id / model.relative_to(directory)))

    experiment = REPOSITORY_ROOT / "experiments" / EXPERIMENT
    for name in (
        "README.md",
        "config.yaml",
        "evidence.csv",
        "training-summary.csv",
        "summary.csv",
        "result.json",
    ):
        sources.append((experiment / name, Path("experiment") / name))
    for figure in sorted((experiment / "figures").glob("*.png")):
        sources.append((figure, Path("experiment") / "figures" / figure.name))

    manifest_files = []
    for source, archive_path in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
        manifest_files.append(
            {
                "path": archive_path.as_posix(),
                "sha256": _sha(source),
                "size_bytes": source.stat().st_size,
            }
        )
    manifest = {"schema_version": 1, "issue": 114, "root": ROOT_NAME, "files": manifest_files}

    output = Path(output).resolve()
    with tempfile.TemporaryDirectory() as temporary:
        manifest_path = Path(temporary) / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with tarfile.open(output, "w:gz") as archive:
            for source, archive_path in sources:
                archive.add(
                    source, arcname=(Path(ROOT_NAME) / archive_path).as_posix(), recursive=False
                )
            archive.add(manifest_path, arcname=f"{ROOT_NAME}/manifest.json", recursive=False)
    return {
        "path": str(output),
        "sha256": _sha(output),
        "size_bytes": output.stat().st_size,
        "files": len(sources) + 1,
    }


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=REPOSITORY_ROOT / "issue114-evidence-v1.tar.gz"
    )
    args = parser.parse_args()
    print(json.dumps(package(args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
