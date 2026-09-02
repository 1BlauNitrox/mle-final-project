"""Run the five preregistered Issue #41 DQN training jobs serially."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from training.plot_run import plot_run
from training.run_experiment import (
    DEFAULT_OUTPUT_ROOT,
    REPOSITORY_ROOT,
    _agent_configuration_reference,
    _git_commit,
    _git_is_dirty,
    run_experiment,
)

ISSUE_NUMBER = 41
AGENT = "DagobertDuckDQN"
SCENARIO = "coin-heaven"
EPISODES_PER_RUN = 10_000
EXPECTED_AGENT_SOURCE_SHA256 = (
    "117e29862c5834b9c78a3479206491542465ebb10e55090cbcfc0c6736f2a8a1"
)
CHECKPOINT_PATH = (
    REPOSITORY_ROOT / "agent_code" / AGENT / "checkpoint.pt"
)
DEFAULT_SERIES_ROOT = DEFAULT_OUTPUT_ROOT / "experiment-dqn-slower-learning"


@dataclass(frozen=True)
class RegisteredRun:
    """One immutable training seed pair from Issue #41."""

    run: int
    world_seed: int
    agent_seed: int


REGISTERED_RUNS = tuple(
    RegisteredRun(
        run=index,
        world_seed=14_000 + index,
        agent_seed=24_000 + index,
    )
    for index in range(1, 4)
)


def run_registered_series(output_root: Path) -> Path:
    """Train all registered models and preserve every final checkpoint."""
    source_reference = _preflight(output_root)
    series_directory = _create_series_directory(output_root)
    runs_directory = series_directory / "runs"
    artifacts_directory = series_directory / "artifacts"
    runs_directory.mkdir()
    artifacts_directory.mkdir()

    manifest_path = series_directory / "series.json"
    manifest: dict[str, Any] = {
        "issue": ISSUE_NUMBER,
        "status": "running",
        "agent": AGENT,
        "scenario": SCENARIO,
        "episodes_per_run": EPISODES_PER_RUN,
        "opponents": [],
        "checkpoint_selection": "final_checkpoint",
        "git_commit": _git_commit(),
        "git_dirty": False,
        "agent_configuration": source_reference,
        "started_at": _utc_now(),
        "finished_at": None,
        "runtime": _runtime_metadata(),
        "registered_runs": [asdict(run) for run in REGISTERED_RUNS],
        "runs": [],
    }
    _write_json(manifest_path, manifest)

    for registered_run in REGISTERED_RUNS:
        run_record: dict[str, Any] = {
            **asdict(registered_run),
            "status": "running",
            "run_directory": None,
            "artifact": None,
            "error": None,
        }
        manifest["runs"].append(run_record)
        _write_json(manifest_path, manifest)

        before_directories = set(runs_directory.iterdir())
        try:
            run_directory = run_experiment(
                agent=AGENT,
                mode="training",
                scenario=SCENARIO,
                rounds=EPISODES_PER_RUN,
                world_seed=registered_run.world_seed,
                agent_seed=registered_run.agent_seed,
                opponents=[],
                output_root=runs_directory,
            )
            run_record["run_directory"] = _relative_to_series(
                run_directory,
                series_directory,
            )
            if not CHECKPOINT_PATH.is_file():
                raise FileNotFoundError(
                    "Training completed without producing checkpoint.pt"
                )

            artifact_path = artifacts_directory / (
                f"run-{registered_run.run:02d}-final-checkpoint.pt"
            )
            shutil.move(CHECKPOINT_PATH, artifact_path)
            run_record["artifact"] = _artifact_record(
                artifact_path,
                series_directory,
            )
            plot_run(run_directory)
            run_record["status"] = "completed"
        except Exception as error:
            run_record["status"] = "failed"
            run_record["error"] = f"{type(error).__name__}: {error}"
            created_directories = (
                set(runs_directory.iterdir()) - before_directories
            )
            if len(created_directories) == 1:
                run_record["run_directory"] = _relative_to_series(
                    created_directories.pop(),
                    series_directory,
                )
            _preserve_failed_checkpoint(
                registered_run=registered_run,
                artifacts_directory=artifacts_directory,
                series_directory=series_directory,
                run_record=run_record,
            )
        finally:
            _write_json(manifest_path, manifest)

    failures = [
        run_record
        for run_record in manifest["runs"]
        if run_record["status"] != "completed"
    ]
    manifest["status"] = "failed" if failures else "completed"
    manifest["finished_at"] = _utc_now()
    _write_json(manifest_path, manifest)

    if failures:
        raise RuntimeError(
            f"{len(failures)} of {len(REGISTERED_RUNS)} training runs failed; "
            f"see {manifest_path}"
        )

    return series_directory


def _preflight(output_root: Path) -> dict[str, str | None]:
    """Reject state that could resume, overwrite, or contaminate a run."""
    if _git_is_dirty():
        raise RuntimeError(
            "Scientific training requires a clean committed worktree."
        )
    if CHECKPOINT_PATH.exists():
        raise FileExistsError(
            f"Refusing to resume or overwrite existing checkpoint: "
            f"{CHECKPOINT_PATH}"
        )

    source_reference = _agent_configuration_reference(AGENT)
    source_hash = source_reference["sha256"]
    if source_hash != EXPECTED_AGENT_SOURCE_SHA256:
        raise RuntimeError(
            "Agent source fingerprint differs from the preregistered value: "
            f"{source_hash}"
        )

    collisions = _find_seed_collisions(output_root)
    if collisions:
        rendered = ", ".join(str(path) for path in collisions)
        raise RuntimeError(
            "Registered training seeds already occur in run metadata: "
            f"{rendered}"
        )

    return source_reference


def _find_seed_collisions(output_root: Path) -> list[Path]:
    """Find prior runner metadata using a registered world or agent seed."""
    registered_world_seeds = {run.world_seed for run in REGISTERED_RUNS}
    registered_agent_seeds = {run.agent_seed for run in REGISTERED_RUNS}
    search_root = output_root.resolve().parent
    collisions: list[Path] = []

    if not search_root.exists():
        return collisions

    for metadata_path in search_root.rglob("metadata.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            metadata.get("world_seed") in registered_world_seeds
            or metadata.get("agent_seed") in registered_agent_seeds
        ):
            collisions.append(metadata_path)

    return sorted(collisions)


def _create_series_directory(output_root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    series_directory = output_root.resolve() / timestamp
    series_directory.mkdir(parents=True, exist_ok=False)
    return series_directory


def _preserve_failed_checkpoint(
    *,
    registered_run: RegisteredRun,
    artifacts_directory: Path,
    series_directory: Path,
    run_record: dict[str, Any],
) -> None:
    if not CHECKPOINT_PATH.is_file():
        return
    artifact_path = artifacts_directory / (
        f"run-{registered_run.run:02d}-failed-checkpoint.pt"
    )
    shutil.move(CHECKPOINT_PATH, artifact_path)
    run_record["artifact"] = _artifact_record(
        artifact_path,
        series_directory,
    )


def _artifact_record(path: Path, series_directory: Path) -> dict[str, Any]:
    return {
        "path": _relative_to_series(path, series_directory),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _relative_to_series(path: Path, series_directory: Path) -> str:
    return path.resolve().relative_to(series_directory.resolve()).as_posix()


def _runtime_metadata() -> dict[str, Any]:
    dependencies = {}
    for package in ("numpy", "torch", "pygame", "scipy", "scikit-learn"):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = None
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor() or None,
        "dependencies": dependencies,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_SERIES_ROOT,
        help="Root for the ignored series directory and model artifacts",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    try:
        series_directory = run_registered_series(arguments.output_root)
    except Exception as error:
        print(f"Training series failed: {error}", file=sys.stderr)
        return 1
    print(f"Training series completed: {series_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
