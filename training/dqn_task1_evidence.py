"""Export and verify the compact, reviewable evidence for Issue #41."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import sys
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from training.aggregate import read_episodes_csv
from training.analyze_dqn_task1_baseline import (
    _aggregate_summaries,
    _evaluate_criteria,
    _plot_evaluation,
    _plot_failures,
    _plot_learning_curves,
    _summarize_rows,
    _write_json,
    _write_summary_csv,
)
from training.evaluate_dqn_task1_baseline import AGENT, DEVELOPMENT_SEEDS, MODELS

EVIDENCE_SCHEMA_VERSION = 1
EVALUATION_EPISODES_FILE = "evaluation-episodes.csv"
DECISION_TIMES_FILE = "evaluation-decision-times.csv.gz"
TRAINING_EPISODES_FILE = "training-episodes.csv.gz"
MANIFEST_FILE = "manifest.json"


def export_evidence(series_directory: Path, evidence_directory: Path) -> None:
    """Export compact source observations without raw logs or snapshots."""
    series_directory = series_directory.resolve()
    evidence_directory = evidence_directory.resolve()
    evidence_directory.mkdir(parents=True, exist_ok=True)

    evaluation_manifest_path = series_directory / "evaluation_manifest.json"
    evaluation_manifest = _read_json(evaluation_manifest_path)
    if evaluation_manifest.get("status") != "completed":
        raise ValueError("Evaluation manifest is not completed")

    evaluation_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    sanitized_jobs: dict[str, dict[str, Any]] = {}
    evaluation_commits: set[str] = set()
    primary_duration = 0.0

    for job_key, job in sorted(evaluation_manifest["jobs"].items()):
        run_directory = series_directory / job["run_directory"]
        rows = read_episodes_csv(run_directory / "episodes.csv")
        matching = [row for row in rows if row["agent"] == AGENT]
        if len(matching) != 1:
            raise ValueError(f"Expected one DQN row for {job_key}")
        statistics = _agent_round_statistics(run_directory / "framework_stats.json")
        metadata = _read_json(run_directory / "metadata.json")
        model = f"run-{job['model_run']:02d}"
        evaluation_rows.append(
            {
                "model": model,
                "pass": job["pass"],
                "world_seed": job["world_seed"],
                "initially_available_coins": statistics[
                    "initially_available_coins"
                ],
                "evaluation_git_commit": metadata["git_commit"],
                "evaluation_duration_seconds": metadata["duration_seconds"],
                **matching[0],
            }
        )
        evaluation_commits.add(metadata["git_commit"])
        if job["pass"] == "primary":
            primary_duration += metadata["duration_seconds"]
        decision_rows.extend(
            {
                "model": model,
                "pass": job["pass"],
                "world_seed": job["world_seed"],
                "decision_index": index,
                "duration_ms": duration,
            }
            for index, duration in enumerate(
                statistics["decision_times_ms"], start=1
            )
        )
        sanitized_jobs[job_key] = {
            key: job[key]
            for key in (
                "model_run",
                "pass",
                "world_seed",
                "agent_seed",
                "status",
                "error",
            )
        }

    training_runs, training_rows = _collect_training_evidence(series_directory)
    _write_csv(evidence_directory / EVALUATION_EPISODES_FILE, evaluation_rows)
    _write_gzip_csv(evidence_directory / DECISION_TIMES_FILE, decision_rows)
    _write_gzip_csv(evidence_directory / TRAINING_EPISODES_FILE, training_rows)

    training_series_path = series_directory / "series.json"
    manifest = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "issue": 41,
        "source_series_directory": series_directory.name,
        "source_evaluation_manifest_sha256": _sha256(evaluation_manifest_path),
        "source_training_series_sha256": _sha256(training_series_path),
        "evaluation_commits": sorted(evaluation_commits),
        "evaluation_duration_seconds_primary": primary_duration,
        "development_seeds": evaluation_manifest["development_seeds"],
        "evaluation_passes": evaluation_manifest["evaluation_passes"],
        "artifacts": evaluation_manifest["artifacts"],
        "models": evaluation_manifest["models"],
        "jobs": sanitized_jobs,
        "training_runs": training_runs,
        "files": {},
    }
    for name in (
        EVALUATION_EPISODES_FILE,
        DECISION_TIMES_FILE,
        TRAINING_EPISODES_FILE,
    ):
        path = evidence_directory / name
        manifest["files"][name] = {
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
    _write_json(evidence_directory / MANIFEST_FILE, manifest)


def verify_evidence(experiment_directory: Path) -> None:
    """Rebuild every committed table and figure and require byte equality."""
    experiment_directory = experiment_directory.resolve()
    evidence_directory = experiment_directory / "evidence"
    manifest = _read_json(evidence_directory / MANIFEST_FILE)
    _validate_evidence_files(evidence_directory, manifest)

    evaluation_rows = read_episodes_csv(
        evidence_directory / EVALUATION_EPISODES_FILE
    )
    decisions = _read_gzip_csv(evidence_directory / DECISION_TIMES_FILE)
    decisions_by_model: dict[str, list[float]] = defaultdict(list)
    for row in decisions:
        if row["pass"] == "primary":
            decisions_by_model[row["model"]].append(float(row["duration_ms"]))

    summaries: list[dict[str, Any]] = []
    training_by_model = {
        run["model"]: run
        for run in manifest["training_runs"]
        if run["status"] == "completed"
    }
    for model in MODELS:
        model_name = f"run-{model.run:02d}"
        rows = [
            row
            for row in evaluation_rows
            if row["model"] == model_name and row["pass"] == "primary"
        ]
        for row in rows:
            row["initially_available_coins"] = int(
                row["initially_available_coins"]
            )
        training = training_by_model[model_name]
        summaries.append(
            _summarize_rows(
                model=model_name,
                rows=rows,
                decision_times=decisions_by_model[model_name],
                training_world_seed=training["world_seed"],
                agent_seed=training["agent_seed"],
                training_episodes=training["rounds"],
                training_duration_seconds=training["duration_seconds"],
                model_sha256=manifest["artifacts"][model_name]["sha256"],
            )
        )

    aggregate = _aggregate_summaries(summaries)
    result = _result_from_evidence(manifest, summaries, aggregate)

    with TemporaryDirectory() as temporary_directory_name:
        temporary_directory = Path(temporary_directory_name)
        figures_directory = temporary_directory / "figures"
        figures_directory.mkdir()
        _write_summary_csv(
            temporary_directory / "summary.csv", [*summaries, aggregate]
        )
        _write_json(temporary_directory / "result.json", result)
        _plot_evaluation(summaries, figures_directory)
        _plot_failures(summaries, figures_directory)
        training_metadata = _materialize_training_rows(
            evidence_directory / TRAINING_EPISODES_FILE,
            temporary_directory,
            training_by_model,
        )
        _plot_learning_curves(training_metadata, figures_directory)

        for relative_path in (
            Path("summary.csv"),
            Path("result.json"),
            Path("figures/evaluation-performance.png"),
            Path("figures/failure-modes.png"),
            Path("figures/learning-curves.png"),
        ):
            generated = temporary_directory / relative_path
            committed = experiment_directory / relative_path
            if generated.read_bytes() != committed.read_bytes():
                raise ValueError(f"Committed output differs: {relative_path}")


def _result_from_evidence(
    manifest: dict[str, Any],
    summaries: list[dict[str, Any]],
    aggregate: dict[str, Any],
) -> dict[str, Any]:
    completed_repeats = sum(
        job["pass"] == "repeat" and job["status"] == "completed"
        for job in manifest["jobs"].values()
    )
    expected_repeats = len(MODELS) * len(DEVELOPMENT_SEEDS)
    if completed_repeats != expected_repeats:
        raise ValueError(
            f"Expected {expected_repeats} completed repeats, found "
            f"{completed_repeats}"
        )
    return {
        "schema_version": 1,
        "series_directory": manifest["source_series_directory"],
        "evaluation_manifest_sha256": manifest[
            "source_evaluation_manifest_sha256"
        ],
        "evaluation_commits": manifest["evaluation_commits"],
        "evaluation_duration_seconds_primary": manifest[
            "evaluation_duration_seconds_primary"
        ],
        "evaluation_repeat_episodes": completed_repeats,
        "criteria": _evaluate_criteria(summaries, aggregate, manifest),
    }


def _collect_training_evidence(
    series_directory: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    model_by_seed = {model.agent_seed: f"run-{model.run:02d}" for model in MODELS}
    runs: list[dict[str, Any]] = []
    episode_rows: list[dict[str, str]] = []
    for metadata_path in sorted(series_directory.glob("**/metadata.json")):
        if "evaluations" in metadata_path.parts:
            continue
        metadata = _read_json(metadata_path)
        agent_seed = metadata.get("agent_seed")
        if agent_seed not in model_by_seed or metadata.get("mode") != "training":
            continue
        model = model_by_seed[agent_seed]
        runs.append(
            {
                "model": model,
                "status": metadata["status"],
                "world_seed": metadata["world_seed"],
                "agent_seed": agent_seed,
                "rounds": metadata["rounds"],
                "duration_seconds": metadata["duration_seconds"],
                "git_commit": metadata["git_commit"],
                "error": metadata.get("error"),
                "source_run_directory": metadata_path.parent.relative_to(
                    series_directory
                ).as_posix(),
            }
        )
        episodes_path = metadata_path.parent / "episodes.csv"
        source_run_directory = metadata_path.parent.relative_to(
            series_directory
        ).as_posix()
        if episodes_path.is_file():
            with episodes_path.open(encoding="utf-8", newline="") as file:
                for row in csv.DictReader(file):
                    episode_rows.append(
                        {
                            "model": model,
                            "source_run_directory": source_run_directory,
                            **row,
                        }
                    )
    return runs, episode_rows


def _materialize_training_rows(
    source_path: Path,
    target_root: Path,
    training_by_model: dict[str, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    rows_by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _read_gzip_csv(source_path):
        rows_by_model[row["model"]].append(row)
    result: dict[int, dict[str, Any]] = {}
    for model_name, training in training_by_model.items():
        directory = target_root / model_name
        directory.mkdir()
        rows = [
            row
            for row in rows_by_model[model_name]
            if row["source_run_directory"] == training["source_run_directory"]
        ]
        _write_csv(
            directory / "episodes.csv",
            rows,
            excluded_fields={"model", "source_run_directory"},
        )
        result[int(training["agent_seed"])] = {**training, "directory": directory}
    return result


def _agent_round_statistics(path: Path) -> dict[str, Any]:
    rounds = _read_json(path)["by_round"]
    if len(rounds) != 1:
        raise ValueError(f"Expected one round in {path}")
    return next(iter(rounds.values()))["agents"][AGENT]


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    excluded_fields: set[str] | None = None,
) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty evidence table: {path.name}")
    excluded_fields = excluded_fields or set()
    fieldnames = [field for field in rows[0] if field not in excluded_fields]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_gzip_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty evidence table: {path.name}")
    with (
        path.open("wb") as raw_file,
        gzip.GzipFile(
            filename="", fileobj=raw_file, mode="wb", mtime=0
        ) as compressed,
        io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text_file,
    ):
        writer = csv.DictWriter(text_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_gzip_csv(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, mode="rt", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _validate_evidence_files(
    evidence_directory: Path, manifest: dict[str, Any]
) -> None:
    if manifest.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise ValueError("Unsupported DQN Task 1 evidence schema")
    for name, record in manifest["files"].items():
        path = evidence_directory / name
        if _sha256(path) != record["sha256"]:
            raise ValueError(f"Evidence checksum mismatch: {name}")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("series_directory", type=Path)
    export_parser.add_argument("evidence_directory", type=Path)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("experiment_directory", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    try:
        if arguments.command == "export":
            export_evidence(arguments.series_directory, arguments.evidence_directory)
        else:
            verify_evidence(arguments.experiment_directory)
    except Exception as error:
        print(f"Evidence {arguments.command} failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
