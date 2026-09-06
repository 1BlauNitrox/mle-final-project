"""Evaluate the five Issue #41 DQN artifacts on registered development seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import sleep
from typing import Any

from training.aggregate import read_episodes_csv
from training.run_dqn_task1_baseline import (
    AGENT,
    CHECKPOINT_PATH,
    DEFAULT_PROFILE,
    PROFILES,
    SCENARIO,
    ExperimentProfile,
)
from training.run_experiment import STAGED_EVALUATION_CHECKPOINT_NAME, run_experiment

DEVELOPMENT_SEEDS = tuple(range(31_001, 31_041))
EVALUATION_PASSES = ("primary", "repeat")
MANIFEST_REPLACE_ATTEMPTS = 10
MANIFEST_REPLACE_RETRY_SECONDS = 0.1
ACTION_SEQUENCE_COLUMN = "executed_action_sequence_sha256"
EVALUATION_CHECKPOINT_ENV = "BOMBERMAN_EVALUATION_CHECKPOINT"
STAGED_CHECKPOINT_PATH = CHECKPOINT_PATH.with_name(STAGED_EVALUATION_CHECKPOINT_NAME)
DETERMINISTIC_COLUMNS = (
    ACTION_SEQUENCE_COLUMN,
    "episode_steps",
    "survival_steps",
    "score",
    "coins_collected",
    "invalid_actions",
    "attempted_actions",
    "invalid_action_rate",
    "survived",
    "termination_reason",
    "action_up",
    "action_right",
    "action_down",
    "action_left",
    "action_wait",
    "action_bomb",
    "action_unknown",
)


@dataclass(frozen=True)
class EvaluationModel:
    """One mechanically selected model and its evaluation agent seed."""

    run: int
    agent_seed: int
    artifact_name: str


def models_for(profile: ExperimentProfile = DEFAULT_PROFILE) -> tuple[EvaluationModel, ...]:
    """Evaluation uses each model's own training agent seed, as issue #41 did."""
    return tuple(
        EvaluationModel(
            run=run.run,
            agent_seed=run.agent_seed,
            artifact_name=f"run-{run.run:02d}-final-checkpoint.pt",
        )
        for run in profile.registered_runs
    )


MODELS = models_for()


def evaluate_series(
    series_directory: Path,
    profile: ExperimentProfile = DEFAULT_PROFILE,
) -> Path:
    """Run or resume all primary and deterministic-repeat evaluations."""
    models = models_for(profile)
    series_directory = series_directory.resolve()
    artifacts_directory = series_directory / "artifacts"
    evaluations_directory = series_directory / "evaluations"
    manifest_path = series_directory / "evaluation_manifest.json"

    artifact_records = _validate_artifacts(artifacts_directory, models)
    _remove_known_checkpoint(artifact_records)
    manifest = _load_or_create_manifest(
        manifest_path=manifest_path,
        series_directory=series_directory,
        artifact_records=artifact_records,
        models=models,
    )
    evaluations_directory.mkdir(exist_ok=True)

    try:
        for model in models:
            model_key = f"run-{model.run:02d}"
            model_record = manifest["models"][model_key]
            artifact_path = artifacts_directory / model.artifact_name
            artifact_hash_before = _sha256(artifact_path)
            model_record["artifact_sha256_before"] = artifact_hash_before
            shutil.copy2(artifact_path, STAGED_CHECKPOINT_PATH)

            for pass_name in EVALUATION_PASSES:
                for world_seed in DEVELOPMENT_SEEDS:
                    job_key = _job_key(
                        model=model,
                        pass_name=pass_name,
                        world_seed=world_seed,
                    )
                    existing = manifest["jobs"].get(job_key)
                    if existing is not None and _job_is_complete(
                        existing,
                        series_directory,
                    ):
                        continue

                    output_root = (
                        evaluations_directory
                        / model_key
                        / pass_name
                        / f"seed-{world_seed}"
                    )
                    job_record: dict[str, Any] = {
                        "model_run": model.run,
                        "pass": pass_name,
                        "world_seed": world_seed,
                        "agent_seed": model.agent_seed,
                        "status": "running",
                        "run_directory": None,
                        "error": None,
                    }
                    manifest["jobs"][job_key] = job_record
                    _write_json(manifest_path, manifest)

                    try:
                        run_directory = run_experiment(
                            agent=AGENT,
                            mode="evaluation",
                            scenario=SCENARIO,
                            rounds=1,
                            world_seed=world_seed,
                            agent_seed=model.agent_seed,
                            opponents=[],
                            output_root=output_root,
                            environment_overrides={
                                EVALUATION_CHECKPOINT_ENV: STAGED_CHECKPOINT_PATH.name
                            },
                        )
                        job_record["run_directory"] = run_directory.relative_to(
                            series_directory
                        ).as_posix()
                        job_record["status"] = "completed"
                    except Exception as error:
                        job_record["status"] = "failed"
                        job_record["error"] = (
                            f"{type(error).__name__}: {error}"
                        )
                        manifest["status"] = "failed"
                        _write_json(manifest_path, manifest)
                        raise
                    _write_json(manifest_path, manifest)

            artifact_hash_after = _sha256(artifact_path)
            checkpoint_hash_after = _sha256(STAGED_CHECKPOINT_PATH)
            model_record["artifact_sha256_after"] = artifact_hash_after
            model_record["checkpoint_sha256_after"] = checkpoint_hash_after
            model_record["immutable"] = (
                artifact_hash_before
                == artifact_hash_after
                == checkpoint_hash_after
            )
            model_record["deterministic"] = _model_is_deterministic(
                manifest=manifest,
                series_directory=series_directory,
                model=model,
            )
            model_record["status"] = (
                "completed"
                if model_record["immutable"]
                and model_record["deterministic"]
                else "failed"
            )
            _write_json(manifest_path, manifest)

            if model_record["status"] != "completed":
                raise RuntimeError(
                    f"Evaluation validation failed for {model_key}"
                )
            STAGED_CHECKPOINT_PATH.unlink()

        manifest["status"] = "completed"
        manifest["finished_at"] = _utc_now()
        _write_json(manifest_path, manifest)
    finally:
        _remove_known_checkpoint(artifact_records)

    return manifest_path


def _validate_artifacts(
    artifacts_directory: Path,
    models: tuple[EvaluationModel, ...] = MODELS,
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for model in models:
        path = artifacts_directory / model.artifact_name
        if not path.is_file():
            raise FileNotFoundError(f"Missing model artifact: {path}")
        records[f"run-{model.run:02d}"] = {
            **asdict(model),
            "path": path.name,
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
    return records


def _remove_known_checkpoint(
    artifact_records: dict[str, dict[str, Any]],
) -> None:
    if not STAGED_CHECKPOINT_PATH.exists():
        return
    checkpoint_hash = _sha256(STAGED_CHECKPOINT_PATH)
    known_hashes = {
        record["sha256"] for record in artifact_records.values()
    }
    if checkpoint_hash not in known_hashes:
        raise RuntimeError(
            "Refusing to replace an unrecognized agent checkpoint: "
            f"{checkpoint_hash}"
        )
    STAGED_CHECKPOINT_PATH.unlink()


def _load_or_create_manifest(
    *,
    manifest_path: Path,
    series_directory: Path,
    artifact_records: dict[str, dict[str, Any]],
    models: tuple[EvaluationModel, ...] = MODELS,
) -> dict[str, Any]:
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("development_seeds") != list(DEVELOPMENT_SEEDS):
            raise ValueError("Evaluation manifest seed list mismatch")
        if manifest.get("artifacts") != artifact_records:
            raise ValueError("Evaluation manifest artifact mismatch")
        return manifest

    manifest = {
        "issue": 41,
        "status": "running",
        "series_directory": series_directory.name,
        "started_at": _utc_now(),
        "finished_at": None,
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "evaluation_passes": list(EVALUATION_PASSES),
        "artifacts": artifact_records,
        "models": {
            f"run-{model.run:02d}": {
                "status": "pending",
                "artifact_sha256_before": None,
                "artifact_sha256_after": None,
                "checkpoint_sha256_after": None,
                "immutable": None,
                "deterministic": None,
            }
            for model in models
        },
        "jobs": {},
    }
    _write_json(manifest_path, manifest)
    return manifest


def _model_is_deterministic(
    *,
    manifest: dict[str, Any],
    series_directory: Path,
    model: EvaluationModel,
) -> bool:
    for world_seed in DEVELOPMENT_SEEDS:
        primary = _read_job_row(
            manifest=manifest,
            series_directory=series_directory,
            job_key=_job_key(
                model=model,
                pass_name="primary",
                world_seed=world_seed,
            ),
        )
        repeat = _read_job_row(
            manifest=manifest,
            series_directory=series_directory,
            job_key=_job_key(
                model=model,
                pass_name="repeat",
                world_seed=world_seed,
            ),
        )
        # An absent digest must fail loudly. Treating it as an equal empty
        # string would let an uninstrumented run pass the determinism gate.
        for row, label in ((primary, "primary"), (repeat, "repeat")):
            digest = row.get(ACTION_SEQUENCE_COLUMN)
            if not isinstance(digest, str) or not digest:
                raise ValueError(
                    f"The {label} evaluation of run-{model.run:02d} on world seed "
                    f"{world_seed} recorded no {ACTION_SEQUENCE_COLUMN}, so "
                    "determinism cannot be verified"
                )

        if any(primary[column] != repeat[column] for column in DETERMINISTIC_COLUMNS):
            return False
    return True


def _read_job_row(
    *,
    manifest: dict[str, Any],
    series_directory: Path,
    job_key: str,
) -> dict[str, Any]:
    job = manifest["jobs"][job_key]
    rows = read_episodes_csv(
        series_directory / job["run_directory"] / "episodes.csv"
    )
    matching = [row for row in rows if row["agent"] == AGENT]
    if len(matching) != 1:
        raise ValueError(f"Expected one DQN row for {job_key}")
    return matching[0]


def _job_is_complete(
    job: dict[str, Any],
    series_directory: Path,
) -> bool:
    if job.get("status") != "completed":
        return False
    relative = job.get("run_directory")
    if not isinstance(relative, str):
        return False
    run_directory = series_directory / relative
    outputs_exist = (
        (run_directory / "metadata.json").is_file()
        and (run_directory / "episodes.csv").is_file()
        and (run_directory / "summary.json").is_file()
    )
    if not outputs_exist:
        return False
    try:
        rows = read_episodes_csv(run_directory / "episodes.csv")
    except (OSError, ValueError):
        return False
    matching = [row for row in rows if row["agent"] == AGENT]
    return (
        len(matching) == 1
        and isinstance(matching[0].get(ACTION_SEQUENCE_COLUMN), str)
        and bool(matching[0][ACTION_SEQUENCE_COLUMN])
    )


def _job_key(
    *,
    model: EvaluationModel,
    pass_name: str,
    world_seed: int,
) -> str:
    return f"run-{model.run:02d}:{pass_name}:{world_seed}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for attempt in range(MANIFEST_REPLACE_ATTEMPTS):
        try:
            temporary_path.replace(path)
            return
        except PermissionError:
            if attempt == MANIFEST_REPLACE_ATTEMPTS - 1:
                raise
            sleep(MANIFEST_REPLACE_RETRY_SECONDS * (attempt + 1))


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "series_directory",
        type=Path,
        help="Completed training-series directory",
    )
    parser.add_argument(
        "--issue",
        type=int,
        default=DEFAULT_PROFILE.issue,
        choices=sorted(PROFILES),
        help="Preregistered experiment profile the series belongs to",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    try:
        manifest_path = evaluate_series(
            arguments.series_directory,
            PROFILES[arguments.issue],
        )
    except Exception as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 1
    print(f"Evaluation completed: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
