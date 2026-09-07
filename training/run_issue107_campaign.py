"""Validate and execute the preregistered Issue #107 Task 2 campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from training.run_experiment import REPOSITORY_ROOT
from training.run_plan import ResolvedPlan, execute_plan, load_plan

EXPERIMENT_DIRECTORY = REPOSITORY_ROOT / "experiments" / "2026-09-07-dqn-task2-factorial"
CONFIG_PATH = EXPERIMENT_DIRECTORY / "config.yaml"
PLAN_DIRECTORY = REPOSITORY_ROOT / "training" / "run_plans"
PLAN_PATHS = {
    "A": PLAN_DIRECTORY / "issue107-cell-a-control.yaml",
    "B": PLAN_DIRECTORY / "issue107-cell-b-escape.yaml",
    "C": PLAN_DIRECTORY / "issue107-cell-c-replay.yaml",
    "D": PLAN_DIRECTORY / "issue107-cell-d-combined.yaml",
    "untrained": PLAN_DIRECTORY / "issue107-untrained.yaml",
    "frozen_task1": PLAN_DIRECTORY / "issue107-frozen-task1.yaml",
}
CELL_TREATMENTS = {
    "A": ("off", "uniform"),
    "B": ("on", "uniform"),
    "C": ("off", "protected_task1"),
    "D": ("on", "protected_task1"),
}
SOURCE_SHA256 = "4ad409472e7ca008dfcc82aa26017017c90259b65f9e593020d1b63921430f60"
TASK1_SHA256 = "eb08e3f67b620ac2a253a2af4db3d5b4c6ea9e667a2aaf1d91e3fccf4ba8b05e"
TRAINING_SEEDS = tuple(zip(range(107_001, 107_006), range(207_001, 207_006), strict=True))
SCENARIOS = ("classic", "coin-heaven", "loot-crate")
EXPECTED_STAGES = (
    ("visible-coins", "coin-heaven", 2_000),
    ("dense-crates", "loot-crate", 2_000),
    ("classic-crates", "classic", 6_000),
)


def validate_protocol() -> dict[str, Any]:
    """Fail closed unless all six plans form the registered campaign."""
    config = _read_yaml(CONFIG_PATH)
    plans = {name: load_plan(path) for name, path in PLAN_PATHS.items()}

    _require(config.get("issue") == 107, "Experiment config must identify Issue #107")
    _require(config.get("status") == "prospective_protocol", "Protocol status changed")
    _verify_artifact(
        REPOSITORY_ROOT / config["source_checkpoint"]["path"],
        SOURCE_SHA256,
        int(config["source_checkpoint"]["size_bytes"]),
    )
    _verify_artifact(
        REPOSITORY_ROOT / config["retention_baseline"]["path"],
        TASK1_SHA256,
        int(config["retention_baseline"]["size_bytes"]),
    )

    reference_replicas: tuple[tuple[str, int, int], ...] | None = None
    reference_suites: tuple[tuple[Any, ...], ...] | None = None
    for cell, expected_treatment in CELL_TREATMENTS.items():
        plan = plans[cell]
        _require(plan.agent == "DagobertDuckDQNTask2", f"Cell {cell} agent mismatch")
        _require(plan.action_masking == "none", f"Cell {cell} must keep masking off")
        _require(
            (plan.escape_continuations, plan.replay_treatment) == expected_treatment,
            f"Cell {cell} treatment mismatch",
        )
        _require(plan.max_parallel_training == 4, f"Cell {cell} worker ceiling mismatch")
        _require(plan.artifact_path == "checkpoint.pt", f"Cell {cell} artifact mismatch")
        replicas = tuple(
            (item.replica_id, item.world_seed, item.agent_seed) for item in plan.replicas
        )
        _require(
            tuple((f"r{index}", *seeds) for index, seeds in enumerate(TRAINING_SEEDS, 1))
            == replicas,
            f"Cell {cell} replica seeds mismatch",
        )
        _require(
            all(item.parent_artifact_sha256 == SOURCE_SHA256 for item in plan.replicas),
            f"Cell {cell} does not use the registered source artifact",
        )
        stages = tuple(
            (job.stage_or_suite, job.scenario, job.rounds)
            for job in plan.jobs
            if job.kind == "training" and job.replica == "r1"
        )
        _require(stages == EXPECTED_STAGES, f"Cell {cell} curriculum mismatch")
        _require(
            sum(job.rounds for job in plan.jobs if job.kind == "training") == 50_000,
            f"Cell {cell} training budget mismatch",
        )
        suites = _suite_signature(plan)
        _require(
            sum(job.rounds for job in plan.jobs if job.kind == "evaluation") == 1_200,
            f"Cell {cell} evaluation budget mismatch",
        )
        if reference_replicas is None:
            reference_replicas = replicas
            reference_suites = suites
        else:
            _require(replicas == reference_replicas, "Cells are not paired by replica")
            _require(suites == reference_suites, "Cells do not share evaluation pairs")

    untrained = plans["untrained"]
    _require(not any(job.kind == "training" for job in untrained.jobs), "Untrained plan trains")
    _require(
        untrained.replicas[0].parent_artifact_sha256 == SOURCE_SHA256,
        "Untrained source mismatch",
    )
    _require(
        sum(job.rounds for job in untrained.jobs if job.kind == "evaluation") == 240,
        "Untrained evaluation budget mismatch",
    )
    frozen = plans["frozen_task1"]
    _require(frozen.agent == "DagobertDuckDQN", "Frozen Task 1 agent mismatch")
    _require(frozen.replicas[0].parent_artifact_sha256 == TASK1_SHA256, "Task 1 source mismatch")
    _require(
        sum(job.rounds for job in frozen.jobs if job.kind == "evaluation") == 80,
        "Frozen Task 1 evaluation budget mismatch",
    )
    _require(
        {job.scenario for job in frozen.jobs if job.kind == "evaluation"} == {"coin-heaven"},
        "Frozen Task 1 plan must evaluate only retention",
    )

    _audit_seed_collisions(plans)
    return {
        "issue": 107,
        "plans": {name: plan.plan_id for name, plan in plans.items()},
        "training_replicas": 20,
        "training_episodes": 200_000,
        "evaluation_episodes": 5_120,
        "source_sha256": SOURCE_SHA256,
        "task1_sha256": TASK1_SHA256,
        "compute_authorized": False,
        "scientific_result": None,
    }


def execute_campaign(
    *,
    reviewed_commit: str,
    authorized_by: str,
    hardware_description: str,
    available_memory_gib: float,
    output_root: Path,
    resume: bool,
) -> None:
    """Execute every immutable plan after an explicit human authorization flag."""
    report = validate_protocol()
    head = _git("rev-parse", "HEAD")
    if reviewed_commit != head:
        raise ValueError(f"Reviewed commit {reviewed_commit!r} does not match HEAD {head}")
    if len(reviewed_commit) != 40:
        raise ValueError("--reviewed-commit must be a full 40-character Git SHA")
    if _git("status", "--porcelain"):
        raise ValueError("Scientific execution requires a clean worktree")
    if not hardware_description.strip():
        raise ValueError("--hardware-description must identify the actual server hardware")
    if not authorized_by.strip():
        raise ValueError("--authorized-by must identify the human compute authorizer")
    if available_memory_gib < 8.0:
        raise ValueError("The server must provide at least the registered 8 GiB aggregate ceiling")

    output_root = Path(output_root).resolve()
    authorization_path = (
        output_root.parent / f"issue107-campaign-authorization-{reviewed_commit}.json"
    )
    authorization_path.parent.mkdir(parents=True, exist_ok=True)
    authorization = {
        **report,
        "compute_authorized": True,
        "authorized_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "authorized_by": authorized_by.strip(),
        "reviewed_commit": reviewed_commit,
        "hardware_description": hardware_description.strip(),
        "available_memory_gib": available_memory_gib,
        "platform": platform.platform(),
        "python": sys.version,
        "logical_cpu_count": os.cpu_count(),
        "cpu_hours_max": 48,
        "wall_clock_hours_max": 24,
        "memory_gib_max": 8,
        "max_training_workers": 4,
        "evaluation_workers": 1,
    }
    if authorization_path.exists():
        existing = json.loads(authorization_path.read_text(encoding="utf-8"))
        immutable_fields = (
            "reviewed_commit",
            "authorized_by",
            "hardware_description",
            "available_memory_gib",
            "cpu_hours_max",
            "wall_clock_hours_max",
            "memory_gib_max",
            "max_training_workers",
            "evaluation_workers",
        )
        changed = any(existing.get(name) != authorization[name] for name in immutable_fields)
        if not resume or changed:
            raise ValueError("Existing Issue #107 authorization record is incompatible")
    else:
        authorization_path.write_text(
            json.dumps(authorization, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    for name in ("A", "B", "C", "D", "untrained", "frozen_task1"):
        plan = load_plan(PLAN_PATHS[name])
        plan_directory = output_root / plan.plan_id
        execute_plan(
            plan,
            output_root=output_root,
            resume=resume and plan_directory.exists(),
        )


def _suite_signature(plan: ResolvedPlan) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            job.stage_or_suite,
            job.scenario,
            job.world_seed,
            job.agent_seed,
            job.rounds,
            job.opponents,
        )
        for job in plan.jobs
        if job.kind == "evaluation" and job.replica == plan.replicas[0].replica_id
    )


def _audit_seed_collisions(plans: dict[str, ResolvedPlan]) -> None:
    registered = {
        seed
        for plan in plans.values()
        for job in plan.jobs
        for seed in (job.world_seed, job.agent_seed)
    }
    confirmation = set(range(307_041, 307_051)) | set(range(308_041, 308_051))
    confirmation |= set(range(309_041, 309_051)) | set(range(407_041, 407_051))
    confirmation |= set(range(408_041, 408_051)) | set(range(409_041, 409_051))
    registered |= confirmation
    collisions: dict[str, list[int]] = {}
    candidates = list((REPOSITORY_ROOT / "training" / "run_plans").glob("*.yaml"))
    candidates += list((REPOSITORY_ROOT / "experiments").glob("*/config.yaml"))
    for path in candidates:
        if "issue107" in path.name or path == CONFIG_PATH:
            continue
        overlap = sorted(registered & _seed_values_from_path(path))
        if overlap:
            collisions[path.relative_to(REPOSITORY_ROOT).as_posix()] = overlap
    _require(not collisions, f"Issue #107 seeds collide with prior records: {collisions}")


def _extract_seed_values(value: Any, key: str = "") -> set[int]:
    result: set[int] = set()
    if isinstance(value, dict):
        if key.endswith("seeds") and {"start", "stop_inclusive"} <= set(value):
            result.update(range(int(value["start"]), int(value["stop_inclusive"]) + 1))
        for child_key, child in value.items():
            result.update(_extract_seed_values(child, str(child_key)))
    elif isinstance(value, list):
        if key.endswith("seed_ranges"):
            for start, stop in value:
                result.update(range(int(start), int(stop) + 1))
        elif "seed" in key:
            result.update(int(item) for item in value if type(item) is int)
        else:
            for child in value:
                result.update(_extract_seed_values(child, key))
    elif "seed" in key and type(value) is int:
        result.add(value)
    return result


def _seed_values_from_path(path: Path) -> set[int]:
    """Read seed values even from historical YAML-like records with backticks."""
    text = path.read_text(encoding="utf-8")
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError:
        values: set[int] = set()
        for line in text.splitlines():
            if "seed" in line.lower():
                values.update(int(item) for item in re.findall(r"\b\d+\b", line))
        return values
    return _extract_seed_values(value)


def _verify_artifact(path: Path, expected_sha256: str, expected_size: int) -> None:
    _require(path.is_file(), f"Artifact is missing: {path}")
    _require(path.stat().st_size == expected_size, f"Artifact size mismatch: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    _require(digest == expected_sha256, f"Artifact checksum mismatch: {path}")


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--authorize-compute", action="store_true")
    parser.add_argument("--reviewed-commit")
    parser.add_argument("--authorized-by")
    parser.add_argument("--hardware-description")
    parser.add_argument("--available-memory-gib", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "training_outputs" / "run-plans",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    try:
        report = validate_protocol()
        if arguments.dry_run:
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        if not arguments.authorize_compute:
            raise ValueError("Execution requires the explicit --authorize-compute flag")
        if not arguments.reviewed_commit:
            raise ValueError("Execution requires --reviewed-commit")
        if not arguments.hardware_description:
            raise ValueError("Execution requires --hardware-description")
        if not arguments.authorized_by:
            raise ValueError("Execution requires --authorized-by")
        if arguments.available_memory_gib is None:
            raise ValueError("Execution requires --available-memory-gib")
        execute_campaign(
            reviewed_commit=arguments.reviewed_commit,
            authorized_by=arguments.authorized_by,
            hardware_description=arguments.hardware_description,
            available_memory_gib=arguments.available_memory_gib,
            output_root=arguments.output_root,
            resume=arguments.resume,
        )
    except Exception as error:
        print(f"Issue #107 campaign failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
