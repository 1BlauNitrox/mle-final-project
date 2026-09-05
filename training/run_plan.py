"""Validate, expand, and execute version-one staged experiment run plans."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import stat
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from time import sleep
from typing import Any

import yaml

import settings
from training.run_experiment import REPOSITORY_ROOT, run_experiment

RUN_PLAN_SCHEMA_VERSION = 1
VALID_POPULATIONS = ("training", "development", "confirmation", "final")
SUPPORTED_OPPONENTS = {
    "peaceful_agent",
    "coin_collector_agent",
    "random_agent",
    "rule_based_agent",
}
SOURCE_PATHS = (
    "agents.py",
    "environment.py",
    "main.py",
    "requirements.txt",
    "requirements-dev.txt",
    "settings.py",
    "training",
)
DEPENDENCIES = (
    "matplotlib",
    "numpy",
    "pygame",
    "pytest",
    "PyYAML",
    "ruff",
    "scikit-learn",
    "scipy",
    "torch",
    "tqdm",
)


@dataclass(frozen=True)
class Job:
    """One exact framework invocation in an expanded run plan."""

    run_id: str
    kind: str
    replica: str
    stage_or_suite: str
    population: str
    scenario: str
    opponents: tuple[str, ...]
    rounds: int
    world_seed: int
    agent_seed: int


@dataclass(frozen=True)
class Replica:
    """Independent training state and seed pair."""

    replica_id: str
    world_seed: int
    agent_seed: int
    parent_artifact: str | None
    parent_artifact_sha256: str | None


@dataclass(frozen=True)
class ResolvedPlan:
    """Validated plan plus reproducibility fingerprints and exact jobs."""

    schema_version: int
    plan_id: str
    agent: str
    artifact_path: str | None
    max_parallel_training: int
    replicas: tuple[Replica, ...]
    jobs: tuple[Job, ...]
    fingerprints: dict[str, Any]

    @property
    def episode_budget(self) -> int:
        return sum(job.rounds for job in self.jobs)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["episode_budget"] = self.episode_budget
        return json.loads(json.dumps(data))


def load_plan(path: Path) -> ResolvedPlan:
    """Load and fully validate a YAML run plan before any episode starts."""
    plan_path = path.resolve()
    try:
        raw = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Could not read run plan {plan_path}: {error}") from error
    if not isinstance(raw, dict):
        raise ValueError("Run plan must be a YAML mapping")

    schema_version = _integer(raw, "schema_version")
    if schema_version != RUN_PLAN_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported run-plan schema version {schema_version}; "
            f"expected {RUN_PLAN_SCHEMA_VERSION}"
        )
    plan_id = _identifier(raw, "plan_id")
    agent = _text(raw, "agent")
    if not (REPOSITORY_ROOT / "agent_code" / agent).is_dir():
        raise ValueError(f"Unknown observed agent {agent!r}")

    artifact_value = raw.get("artifact_path")
    artifact_path = None
    if artifact_value is not None:
        artifact_path = _relative_file_path(artifact_value, "artifact_path").as_posix()

    max_parallel = raw.get("max_parallel_training", 1)
    if not isinstance(max_parallel, int) or isinstance(max_parallel, bool) or max_parallel < 1:
        raise ValueError("max_parallel_training must be a positive integer")

    raw_replicas = _list(raw, "replicas")
    if not raw_replicas:
        raise ValueError("replicas must contain at least one replica")
    replicas = tuple(_parse_replica(item, plan_path.parent) for item in raw_replicas)
    _require_unique([replica.replica_id for replica in replicas], "replica IDs")
    if artifact_path is None and any(replica.parent_artifact for replica in replicas):
        raise ValueError("artifact_path is required when a parent_artifact is present")

    raw_stages = raw.get("training_stages", [])
    raw_suites = raw.get("evaluation_suites", [])
    if not isinstance(raw_stages, list) or not isinstance(raw_suites, list):
        raise ValueError("training_stages and evaluation_suites must be lists")
    if raw_stages and artifact_path is None:
        raise ValueError("artifact_path is required when training_stages are present")

    stages = [_parse_stage(item) for item in raw_stages]
    suites = [_parse_suite(item) for item in raw_suites]
    _require_unique([stage["id"] for stage in stages], "training stage IDs")
    _require_unique([suite["id"] for suite in suites], "evaluation suite IDs")

    jobs = _expand_jobs(replicas, stages, suites)
    _require_unique([job.run_id for job in jobs], "expanded run IDs")
    _validate_seed_populations(replicas, suites)

    fingerprints = {
        "configuration": _sha256_bytes(
            json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ),
        "source": _fingerprint_paths(SOURCE_PATHS),
        "framework": _fingerprint_paths(
            ("agents.py", "environment.py", "main.py", "settings.py")
        ),
        "agent": _fingerprint_directory(REPOSITORY_ROOT / "agent_code" / agent),
        "parent_artifacts": {
            replica.replica_id: replica.parent_artifact_sha256
            for replica in replicas
        },
        "dependencies": _dependency_record(),
    }
    fingerprints["dependencies_sha256"] = _sha256_bytes(
        json.dumps(fingerprints["dependencies"], sort_keys=True).encode("utf-8")
    )

    return ResolvedPlan(
        schema_version=schema_version,
        plan_id=plan_id,
        agent=agent,
        artifact_path=artifact_path,
        max_parallel_training=max_parallel,
        replicas=replicas,
        jobs=jobs,
        fingerprints=fingerprints,
    )


def execute_plan(
    plan: ResolvedPlan,
    *,
    output_root: Path,
    resume: bool = False,
    evaluation_only: bool = False,
    workspace_root: Path | None = None,
) -> Path:
    """Execute or resume a validated plan and retain every attempt record."""
    if evaluation_only:
        plan = replace(
            plan,
            jobs=tuple(job for job in plan.jobs if job.kind == "evaluation"),
        )
        if not plan.jobs:
            raise ValueError("Evaluation-only mode requires evaluation jobs")
        if resume:
            raise ValueError("Evaluation-only mode cannot resume an existing plan")
        if workspace_root is None:
            raise ValueError("Evaluation-only mode requires --workspace-root")
        workspace_root = Path(workspace_root).resolve()
    plan_directory = output_root.resolve() / plan.plan_id
    resolved_path = plan_directory / "resolved_plan.json"
    status_path = plan_directory / "status.json"
    current = plan.to_dict()

    if resume:
        if not resolved_path.is_file() or not status_path.is_file():
            raise ValueError(f"No resumable run plan exists at {plan_directory}")
        recorded = _read_json(resolved_path)
        if recorded != current:
            raise ValueError(_resume_mismatch(recorded, current))
        status = _read_json(status_path)
    else:
        plan_directory.mkdir(parents=True, exist_ok=False)
        _write_json_atomic(resolved_path, current)
        status = _new_status(plan)
        _write_json_atomic(status_path, status)

    lock = threading.Lock()
    status["status"] = "running"
    status["updated_at"] = _timestamp()
    _mark_abandoned_attempts_interrupted(status)
    _write_json_atomic(status_path, status)

    try:
        training_by_replica = {
            replica.replica_id: [
                job
                for job in plan.jobs
                if job.kind == "training" and job.replica == replica.replica_id
            ]
            for replica in plan.replicas
        }
        with ThreadPoolExecutor(max_workers=plan.max_parallel_training) as executor:
            futures = [
                executor.submit(
                    _run_training_sequence,
                    plan,
                    replica,
                    training_by_replica[replica.replica_id],
                    plan_directory,
                    status,
                    status_path,
                    lock,
                    workspace_root,
                )
                for replica in plan.replicas
                if training_by_replica[replica.replica_id]
            ]
            for future in as_completed(futures):
                future.result()

        for job in plan.jobs:
            if job.kind == "evaluation":
                _run_job(
                    plan,
                    job,
                    plan_directory,
                    status,
                    status_path,
                    lock,
                    workspace_root,
                )
    except BaseException as error:
        status["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        status["error"] = f"{type(error).__name__}: {error}"
        status["updated_at"] = _timestamp()
        _write_json_atomic(status_path, status)
        _discard_staging_aliases(plan)
        raise

    status["status"] = "completed"
    status["error"] = None
    status["updated_at"] = _timestamp()
    _write_json_atomic(status_path, status)
    _remove_staging_aliases(plan, plan_directory)
    return plan_directory


def _run_training_sequence(
    plan: ResolvedPlan,
    replica: Replica,
    jobs: list[Job],
    plan_directory: Path,
    status: dict[str, Any],
    status_path: Path,
    lock: threading.Lock,
    workspace_root: Path | None = None,
) -> None:
    for job in jobs:
        _run_job(plan, job, plan_directory, status, status_path, lock, workspace_root)


def _run_job(
    plan: ResolvedPlan,
    job: Job,
    plan_directory: Path,
    status: dict[str, Any],
    status_path: Path,
    lock: threading.Lock,
    workspace_root: Path | None = None,
) -> None:
    job_status = status["jobs"][job.run_id]
    if job_status["status"] == "completed":
        return

    replica = next(item for item in plan.replicas if item.replica_id == job.replica)
    alias_directory = _prepare_replica_workspace(
        plan, replica, plan_directory, workspace_root
    )
    alias = alias_directory.name
    artifact = alias_directory / plan.artifact_path if plan.artifact_path else None
    artifact_before = _sha256_file(artifact) if artifact and artifact.is_file() else None
    artifact_record: dict[str, Any] | None = None
    if job.kind == "evaluation" and artifact is not None and artifact.is_file():
        artifact.chmod(stat.S_IREAD)

    attempt_number = len(job_status["attempts"]) + 1
    attempt_id = f"attempt-{attempt_number:03d}"
    attempt = {
        "attempt": attempt_number,
        "status": "running",
        "started_at": _timestamp(),
        "finished_at": None,
        "error": None,
        "output": f"jobs/{job.run_id}/{attempt_id}",
    }
    attempt_root = plan_directory / "jobs" / job.run_id
    attempt_root.mkdir(parents=True, exist_ok=True)
    attempt_input = attempt_root / f"{attempt_id}-input-agent"
    _snapshot_workspace(alias_directory, attempt_input)
    with lock:
        job_status["status"] = "running"
        job_status["attempts"].append(attempt)
        status["updated_at"] = _timestamp()
        _write_json_atomic(status_path, status)

    try:
        run_directory = run_experiment(
            agent=alias,
            mode=job.kind,
            scenario=job.scenario,
            rounds=job.rounds,
            world_seed=job.world_seed,
            agent_seed=job.agent_seed,
            opponents=list(job.opponents),
            output_root=plan_directory / "jobs" / job.run_id,
            run_id=attempt_id,
            metadata_extra={
                "run_plan": {
                    "plan_id": plan.plan_id,
                    "logical_agent": plan.agent,
                    "job_id": job.run_id,
                    "replica": job.replica,
                    "stage_or_suite": job.stage_or_suite,
                    "population": job.population,
                    "processes": 1,
                    "artifact_writable": job.kind == "training",
                    "fingerprints": plan.fingerprints,
                }
            },
        )
        if job.kind == "evaluation":
            artifact_after = _sha256_file(artifact) if artifact and artifact.is_file() else None
            if artifact_before != artifact_after:
                raise RuntimeError(f"Evaluation job {job.run_id} modified its artifact")
            if artifact is not None:
                artifact_record = {
                    "path": (
                        Path("replicas")
                        / job.replica
                        / "agent"
                        / str(plan.artifact_path)
                    ).as_posix(),
                    "sha256": artifact_before,
                    "selection": "immutable evaluation input",
                }
        else:
            if artifact is None or not artifact.is_file():
                raise FileNotFoundError(
                    f"Training job {job.run_id} did not produce {plan.artifact_path}"
                )
            checkpoint = (
                plan_directory
                / "artifacts"
                / job.replica
                / job.stage_or_suite
                / artifact.name
            )
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(artifact, checkpoint)
            artifact_record = {
                "path": checkpoint.relative_to(plan_directory).as_posix(),
                "sha256": _sha256_file(checkpoint),
                "selection": "final checkpoint after the exact stage budget",
            }
        _snapshot_workspace(alias_directory, _workspace_directory(plan_directory, job.replica))
        with lock:
            if artifact_record is not None:
                job_status["artifact"] = artifact_record
            attempt["status"] = "completed"
            attempt["finished_at"] = _timestamp()
            attempt["metadata"] = (
                run_directory / "metadata.json"
            ).relative_to(plan_directory).as_posix()
            job_status["status"] = "completed"
    except BaseException as error:
        if alias_directory.is_dir():
            _snapshot_workspace(
                alias_directory,
                attempt_root / f"{attempt_id}-failed-agent",
            )
            _remove_tree(alias_directory)
        with lock:
            attempt["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            attempt["finished_at"] = _timestamp()
            attempt["error"] = f"{type(error).__name__}: {error}"
            job_status["status"] = attempt["status"]
        raise
    finally:
        with lock:
            status["updated_at"] = _timestamp()
            _write_json_atomic(status_path, status)


def _prepare_replica_workspace(
    plan: ResolvedPlan,
    replica: Replica,
    plan_directory: Path,
    workspace_root: Path | None = None,
) -> Path:
    workspace = _workspace_directory(plan_directory, replica.replica_id)
    alias = _alias_directory(plan, replica)
    if not workspace.is_dir():
        if workspace_root is not None:
            source = _workspace_directory(
                workspace_root / plan.plan_id,
                replica.replica_id,
            )
            if not source.is_dir():
                raise FileNotFoundError(
                    f"Evaluation-only source workspace does not exist: {source}"
                )
        else:
            source = REPOSITORY_ROOT / "agent_code" / plan.agent
        _snapshot_workspace(source, workspace)
        if workspace_root is None and replica.parent_artifact:
            target = workspace / str(plan.artifact_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(replica.parent_artifact, target)
    # The alias is disposable. Recreate it before every job so a stale or
    # partially written process workspace can never contaminate a retry.
    if alias.is_dir():
        _remove_tree(alias)
    _snapshot_workspace(workspace, alias)
    return alias


def _snapshot_workspace(source: Path, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    if temporary.exists():
        _remove_tree(temporary)
    shutil.copytree(
        source,
        temporary,
        ignore=shutil.ignore_patterns("__pycache__", "logs", "*.pyc"),
    )
    _make_tree_writable(temporary)
    for attempt in range(5):
        if destination.exists():
            _remove_tree(destination)
        try:
            temporary.replace(destination)
            return
        except PermissionError:
            if attempt == 4:
                raise
            sleep(0.05 * (attempt + 1))


def _expand_jobs(
    replicas: tuple[Replica, ...],
    stages: list[dict[str, Any]],
    suites: list[dict[str, Any]],
) -> tuple[Job, ...]:
    jobs: list[Job] = []
    for replica in replicas:
        for stage in stages:
            jobs.append(
                Job(
                    run_id=f"train-{replica.replica_id}-{stage['id']}",
                    kind="training",
                    replica=replica.replica_id,
                    stage_or_suite=stage["id"],
                    population="training",
                    scenario=stage["scenario"],
                    opponents=tuple(stage["opponents"]),
                    rounds=stage["rounds"],
                    world_seed=replica.world_seed,
                    agent_seed=replica.agent_seed,
                )
            )
    for replica in replicas:
        for suite in suites:
            for index, (world_seed, agent_seed) in enumerate(
                zip(suite["world_seeds"], suite["agent_seeds"], strict=True), start=1
            ):
                jobs.append(
                    Job(
                        run_id=f"eval-{replica.replica_id}-{suite['id']}-seed-{index:03d}",
                        kind="evaluation",
                        replica=replica.replica_id,
                        stage_or_suite=suite["id"],
                        population=suite["population"],
                        scenario=suite["scenario"],
                        opponents=tuple(suite["opponents"]),
                        rounds=suite["rounds"],
                        world_seed=world_seed,
                        agent_seed=agent_seed,
                    )
                )
    return tuple(jobs)


def _parse_replica(value: Any, plan_directory: Path) -> Replica:
    mapping = _mapping(value, "replica")
    replica_id = _identifier(mapping, "id")
    world_seed = _non_negative_integer(mapping, "world_seed")
    agent_seed = _non_negative_integer(mapping, "agent_seed")
    parent_value = mapping.get("parent_artifact")
    if parent_value is None:
        return Replica(replica_id, world_seed, agent_seed, None, None)
    parent = Path(_text(mapping, "parent_artifact"))
    resolved = parent if parent.is_absolute() else (plan_directory / parent).resolve()
    if not resolved.is_file():
        raise ValueError(f"Parent artifact does not exist: {resolved}")
    return Replica(replica_id, world_seed, agent_seed, str(resolved), _sha256_file(resolved))


def _parse_stage(value: Any) -> dict[str, Any]:
    mapping = _mapping(value, "training stage")
    return {
        "id": _identifier(mapping, "id"),
        "scenario": _scenario(mapping),
        "rounds": _positive_integer(mapping, "rounds"),
        "opponents": _opponents(mapping),
    }


def _parse_suite(value: Any) -> dict[str, Any]:
    mapping = _mapping(value, "evaluation suite")
    population = _text(mapping, "population")
    if population not in VALID_POPULATIONS[1:]:
        raise ValueError(
            f"Evaluation population must be one of {list(VALID_POPULATIONS[1:])}"
        )
    world_seeds = _integer_list(mapping, "world_seeds")
    agent_seeds = _integer_list(mapping, "agent_seeds")
    if len(world_seeds) != len(agent_seeds) or not world_seeds:
        raise ValueError("world_seeds and agent_seeds must be non-empty equal-length lists")
    return {
        "id": _identifier(mapping, "id"),
        "population": population,
        "scenario": _scenario(mapping),
        "rounds": _positive_integer(mapping, "rounds"),
        "opponents": _opponents(mapping),
        "world_seeds": world_seeds,
        "agent_seeds": agent_seeds,
    }


def _validate_seed_populations(
    replicas: tuple[Replica, ...], suites: list[dict[str, Any]]
) -> None:
    populations: dict[str, set[int]] = {name: set() for name in VALID_POPULATIONS}
    for replica in replicas:
        populations["training"].update((replica.world_seed, replica.agent_seed))
    for suite in suites:
        populations[suite["population"]].update(
            suite["world_seeds"] + suite["agent_seeds"]
        )
    for index, left in enumerate(VALID_POPULATIONS):
        for right in VALID_POPULATIONS[index + 1 :]:
            overlap = populations[left] & populations[right]
            if overlap:
                raise ValueError(
                    f"Protected seed populations {left!r} and {right!r} overlap: "
                    f"{sorted(overlap)}"
                )


def _new_status(plan: ResolvedPlan) -> dict[str, Any]:
    return {
        "schema_version": RUN_PLAN_SCHEMA_VERSION,
        "plan_id": plan.plan_id,
        "status": "pending",
        "error": None,
        "created_at": _timestamp(),
        "updated_at": _timestamp(),
        "jobs": {
            job.run_id: {
                "status": "pending",
                "kind": job.kind,
                "replica": job.replica,
                "population": job.population,
                "attempts": [],
            }
            for job in plan.jobs
        },
    }


def _mark_abandoned_attempts_interrupted(status: dict[str, Any]) -> None:
    for job in status["jobs"].values():
        if job["status"] != "running":
            continue
        job["status"] = "interrupted"
        attempt = job["attempts"][-1]
        attempt["status"] = "interrupted"
        attempt["finished_at"] = _timestamp()
        attempt["error"] = "Run-plan process ended before recording completion"


def _resume_mismatch(recorded: dict[str, Any], current: dict[str, Any]) -> str:
    old = recorded.get("fingerprints", {})
    new = current.get("fingerprints", {})
    changed = sorted(key for key in set(old) | set(new) if old.get(key) != new.get(key))
    if not changed:
        changed = ["resolved plan"]
    return "Resume rejected because fingerprints differ: " + ", ".join(changed)


def _alias_directory(plan: ResolvedPlan, replica: Replica) -> Path:
    token = plan.fingerprints["configuration"][:10]
    replica_token = _sha256_bytes(replica.replica_id.encode("utf-8"))[:10]
    return REPOSITORY_ROOT / "agent_code" / f"_run_plan_{token}_{replica_token}"


def _workspace_directory(plan_directory: Path, replica_id: str) -> Path:
    return plan_directory / "replicas" / replica_id / "agent"


def _remove_staging_aliases(plan: ResolvedPlan, plan_directory: Path) -> None:
    for replica in plan.replicas:
        alias = _alias_directory(plan, replica)
        if alias.is_dir():
            _snapshot_workspace(
                alias, _workspace_directory(plan_directory, replica.replica_id)
            )
            _remove_tree(alias)


def _discard_staging_aliases(plan: ResolvedPlan) -> None:
    """Remove disposable aliases after a failed plan without touching snapshots."""
    for replica in plan.replicas:
        alias = _alias_directory(plan, replica)
        if alias.is_dir():
            _remove_tree(alias)


def _make_tree_writable(path: Path) -> None:
    for candidate in [path, *path.rglob("*")]:
        candidate.chmod(candidate.stat().st_mode | stat.S_IWUSR)


def _remove_tree(path: Path) -> None:
    """Remove one known staging tree, including read-only Windows directories."""
    resolved = path.resolve()
    if not resolved.exists():
        return
    if resolved == Path(resolved.anchor) or len(resolved.parts) < 3:
        raise ValueError(f"Refusing to remove broad path: {resolved}")

    def make_writable_and_retry(function: Any, target: str, _: Any) -> None:
        os.chmod(target, stat.S_IWRITE)
        function(target)

    shutil.rmtree(resolved, onerror=make_writable_and_retry)


def _dependency_record() -> dict[str, str]:
    record = {"python": platform.python_version()}
    for name in DEPENDENCIES:
        try:
            record[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            record[name] = "missing"
    return record


def _fingerprint_paths(paths: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for name in paths:
        path = REPOSITORY_ROOT / name
        candidates = sorted(path.rglob("*.py")) if path.is_dir() else [path]
        for candidate in candidates:
            if not candidate.is_file() or "__pycache__" in candidate.parts:
                continue
            digest.update(candidate.relative_to(REPOSITORY_ROOT).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(candidate.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def _fingerprint_directory(path: Path) -> str:
    digest = hashlib.sha256()
    ignored_parts = {"__pycache__", "logs"}
    ignored_files = {".evaluation-checkpoint.pt"}
    for candidate in sorted(path.rglob("*")):
        if (
            not candidate.is_file()
            or ignored_parts.intersection(candidate.relative_to(path).parts)
            or candidate.name in ignored_files
            or candidate.suffix == ".pyc"
        ):
            continue
        digest.update(candidate.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(candidate.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _scenario(mapping: dict[str, Any]) -> str:
    scenario = _text(mapping, "scenario")
    if scenario not in settings.SCENARIOS:
        raise ValueError(
            f"Unsupported scenario {scenario!r}; expected one of {sorted(settings.SCENARIOS)}"
        )
    return scenario


def _opponents(mapping: dict[str, Any]) -> list[str]:
    values = mapping.get("opponents", [])
    if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
        raise ValueError("opponents must be a list of agent names")
    if len(values) > 3:
        raise ValueError("A job may contain at most three opponents")
    unsupported = [
        item
        for item in values
        if item not in SUPPORTED_OPPONENTS
        and not (REPOSITORY_ROOT / "agent_code" / item).is_dir()
    ]
    if unsupported:
        raise ValueError(f"Unsupported supplied opponents: {unsupported}")
    return values


def _relative_file_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty relative file path")
    path = Path(value)
    if (
        path.is_absolute()
        or path.drive
        or ".." in path.parts
        or path.name in {"", "."}
    ):
        raise ValueError(f"{field} must be an unambiguous path inside the agent directory")
    return path


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return value


def _list(mapping: dict[str, Any], field: str) -> list[Any]:
    value = mapping.get(field)
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return value


def _text(mapping: dict[str, Any], field: str) -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _identifier(mapping: dict[str, Any], field: str) -> str:
    value = _text(mapping, field)
    if any(not (character.isalnum() or character in "-_") for character in value):
        raise ValueError(
            f"{field} must contain only letters, digits, hyphens, and underscores"
        )
    return value


def _integer(mapping: dict[str, Any], field: str) -> int:
    value = mapping.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    return value


def _positive_integer(mapping: dict[str, Any], field: str) -> int:
    value = _integer(mapping, field)
    if value < 1:
        raise ValueError(f"{field} must be positive")
    return value


def _non_negative_integer(mapping: dict[str, Any], field: str) -> int:
    value = _integer(mapping, field)
    if value < 0:
        raise ValueError(f"{field} must be non-negative")
    return value


def _integer_list(mapping: dict[str, Any], field: str) -> list[int]:
    values = _list(mapping, field)
    if not all(
        isinstance(item, int) and not isinstance(item, bool) and item >= 0
        for item in values
    ):
        raise ValueError(f"{field} must contain only non-negative integers")
    if len(values) != len(set(values)):
        raise ValueError(f"{field} must not contain duplicate seeds")
    return values


def _require_unique(values: list[str], description: str) -> None:
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ValueError(f"Duplicate {description}: {duplicates}")


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON mapping in {path}")
    return value


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, help="Version-one YAML run plan")
    parser.add_argument("--dry-run", action="store_true", help="Print the complete matrix only")
    parser.add_argument("--resume", action="store_true", help="Resume a matching recorded plan")
    parser.add_argument(
        "--evaluation-only",
        action="store_true",
        help="Run only evaluation jobs using workspaces from an earlier plan",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        help="Earlier run-plan output root used as the evaluation-only workspace source",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "training_outputs" / "run-plans",
        help="Parent directory for resolved plans and job outputs",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    try:
        plan = load_plan(arguments.plan)
        if arguments.dry_run:
            print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
            return 0
        directory = execute_plan(
            plan,
            output_root=arguments.output_root,
            resume=arguments.resume,
            evaluation_only=arguments.evaluation_only,
            workspace_root=arguments.workspace_root,
        )
    except Exception as error:
        print(f"Run plan failed: {error}", file=sys.stderr)
        return 1
    print(f"Run plan completed: {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
