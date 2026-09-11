"""Guard the registered Task 3 campaigns; dry-run never starts a game."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

from training.run_experiment import REPOSITORY_ROOT as ROOT
from training.run_issue107_campaign import (
    CampaignLimits,
    CampaignResourceMonitor,
    _seed_values_from_path,
)
from training.run_plan import execute_plan, load_plan

CONFIG = ROOT / "experiments/2026-09-08-dqn-task3-peaceful-opponent/config.yaml"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def relative_file(root, name):
    path = (root / name).resolve()
    require(path.is_relative_to(root.resolve()), "Evidence path escapes its directory")
    return path


def portable_plan(plan):
    """Ignore only relocated parent paths and content-derived configuration hash."""
    data = plan.to_dict() if hasattr(plan, "to_dict") else json.loads(json.dumps(plan))
    for replica in data["replicas"]:
        replica["parent_artifact"] = None
    data["fingerprints"].pop("configuration", None)
    return data


def protocol_path(protocol):
    if protocol == "peaceful":
        return CONFIG
    if protocol == "coincollector":
        return ROOT / "experiments/2026-09-11-task3-coincollector/config.yaml"
    raise ValueError("Unknown Task 3 protocol")


def validate_protocol(binding_directory=None, protocol="peaceful"):
    config_path = protocol_path(protocol)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    require(config["schema_version"] == 2, "Unsupported Task 3 protocol")
    templates = {name: load_plan(ROOT / path) for name, path in config["plans"].items()}
    expected = {"candidate": (50000, 1600), "reference": (0, 320)}
    for name, plan in templates.items():
        budget = tuple(
            sum(j.rounds for j in plan.jobs if j.kind == kind)
            for kind in ("training", "evaluation")
        )
        require(budget == expected[name], f"Incorrect {name} budget")
        require(
            plan.max_parallel_training <= config["resources"]["max_parallel_training"],
            "Worker ceiling exceeded",
        )

    def signature(plan):
        return [
            (j.stage_or_suite, j.scenario, j.opponents, j.rounds, j.world_seed, j.agent_seed)
            for j in plan.jobs
            if j.kind == "evaluation" and j.replica == plan.replicas[0].replica_id
        ]

    require(
        signature(templates["candidate"]) == signature(templates["reference"]),
        "Candidate/reference evaluation conditions differ",
    )
    registered = {
        seed for p in templates.values() for j in p.jobs for seed in (j.world_seed, j.agent_seed)
    }
    protected = {
        seed for start, stop in config["protected_seed_ranges"] for seed in range(start, stop + 1)
    }
    require(not registered & protected, "Confirmation seeds would be consumed")
    excluded = {ROOT / p for p in config["plans"].values()} | {config_path}
    excluded.add(ROOT / "training/run_plans/issue109-task3-vs-coincollector.yaml")
    audit = {}
    for path in sorted(
        [
            *(ROOT / "training/run_plans").glob("*.yaml"),
            *(ROOT / "experiments").glob("*/config.yaml"),
        ]
    ):
        if path in excluded:
            continue
        collision = (registered | protected) & _seed_values_from_path(path)
        require(not collision, f"Seed collision in {path}: {sorted(collision)}")
        audit[path.relative_to(ROOT).as_posix()] = sha256(path)
    plans = templates
    binding = None
    if binding_directory is not None:
        directory = Path(binding_directory).resolve()
        binding = read_json(directory / "binding.json")
        require(binding.get("task2_complete") is False, "This campaign is exploratory")
        require(len(binding["parent"]["source_commit"]) == 40, "Full parent source commit required")
        for key in ("parent", "successor"):
            record = binding[key]
            path = relative_file(directory, record["path"])
            require(
                sha256(path) == record["sha256"] and path.stat().st_size == record["size_bytes"],
                f"{key} artifact hash/size mismatch",
            )
        verify_migration(directory, binding, parent_agent=templates["reference"].agent)
        if protocol == "coincollector":
            verify_prerequisite(directory, binding)
        plans = {}
        for name, template in templates.items():
            record = binding["plans"][name]
            path = relative_file(directory, record["path"])
            require(sha256(path) == record["sha256"], "Bound plan hash mismatch")
            plan = load_plan(path)
            actual, expected_plan = plan.to_dict(), template.to_dict()
            # Only binding identity, parent bytes and inherited modes may differ.
            expected_plan["plan_id"] = actual["plan_id"]
            for mode in ("action_masking", "escape_continuations"):
                require(actual[mode] == binding["modes"][mode], "Bound mode mismatch")
                expected_plan[mode] = actual[mode]
            key = "successor" if name == "candidate" else "parent"
            for replica in expected_plan["replicas"]:
                replica["parent_artifact"] = str(relative_file(directory, binding[key]["path"]))
                replica["parent_artifact_sha256"] = binding[key]["sha256"]
            # Fingerprints are validated against the actual bound plans at execution/resume.
            actual.pop("fingerprints")
            expected_plan.pop("fingerprints")
            require(actual == expected_plan, f"Bound {name} differs from registered matrix")
            plans[name] = plan
    report = {
        "issue": config["issue"],
        "exploratory": True,
        "task2_complete": False,
        "training_episodes": 50000,
        "evaluation_episodes": 1920,
        "protocol_sha256": sha256(config_path),
        "seed_audit": audit,
        "parent_bound": binding is not None,
        "compute_authorized": False,
    }
    return config, plans, report


def verify_migration(directory, binding, parent_agent="DagobertDuckDQNTask2"):
    """Verify function-preserving lineage, not just two unrelated file hashes."""
    import torch

    if parent_agent == "DagobertDuckDQNTask2":
        from agent_code.DagobertDuckDQNTask2.persistence import load_evaluation_checkpoint
    else:
        from agent_code.DagobertDuckDQNTask3.persistence import load_evaluation_checkpoint
    from agent_code.DagobertDuckDQNTask3.migration import successor_config
    from agent_code.DagobertDuckDQNTask3.persistence import load_training_checkpoint

    parent_path = relative_file(directory, binding["parent"]["path"])
    parent = load_evaluation_checkpoint(parent_path)
    successor = load_training_checkpoint(relative_file(directory, binding["successor"]["path"]))
    require(
        successor.config
        == (
            successor_config(parent.config)
            if parent_agent == "DagobertDuckDQNTask2"
            else parent.config
        ),
        "Migration changed inherited hyperparameters",
    )
    require(
        successor.completed_episodes
        == successor.learner.update_steps
        == len(successor.replay_buffer)
        == 0,
        "Successor must be a fresh migration",
    )
    require(
        successor.config.action_masking == parent.config.action_masking
        and successor.config.escape_continuation_features
        == parent.config.escape_continuation_features,
        "Migration changed inherited modes",
    )
    require(
        binding["modes"]
        == {
            "action_masking": "framework_legal" if parent.config.action_masking else "none",
            "escape_continuations": "on" if parent.config.escape_continuation_features else "off",
        },
        "Binding modes disagree with checkpoint",
    )
    payload = torch.load(parent_path, map_location="cpu", weights_only=True)
    for name, network in (
        ("online_network", successor.learner.online_network),
        ("target_network", successor.learner.target_network),
    ):
        for key, values in payload["learner_state"][name].items():
            actual = network.state_dict()[key]
            if key == "layers.0.weight":
                require(
                    torch.equal(actual[:, : parent.config.input_dim], values)
                    and torch.count_nonzero(actual[:, parent.config.input_dim :]) == 0,
                    "Migration input weights differ",
                )
            else:
                require(torch.equal(actual, values), "Migration network differs")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def execute(args):
    config, plans, report = validate_protocol(args.binding_dir, args.protocol)
    require(args.authorize_compute, "Execution requires --authorize-compute")
    require(args.binding_dir is not None, "Execution requires --binding-dir")
    require(args.reviewed_commit == git("rev-parse", "HEAD"), "Reviewed commit must equal HEAD")
    require(not git("status", "--porcelain"), "Execution requires a clean worktree")
    require(
        args.authorized_by and args.hardware_description,
        "Explicit human authorizer and actual hardware description required",
    )
    limits = config["resources"]
    require(
        args.available_memory_gib is not None and args.available_memory_gib >= limits["memory_gib"],
        "Insufficient declared available campaign memory",
    )
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".task3-campaign.lock"
    # Exclusive ownership: never run two launchers against shared budget records.
    with lock.open("x", encoding="utf-8") as file:
        file.write("Exclusive campaign lock; remove only after verifying the owner exited.\n")
    try:
        auth_path = root / "authorization.json"
        identity = {
            "issue": config["issue"],
            "reviewed_commit": args.reviewed_commit,
            "authorized_by": args.authorized_by,
            "hardware_description": args.hardware_description,
            "available_memory_gib": args.available_memory_gib,
            "protocol_sha256": report["protocol_sha256"],
            "limits": limits,
            "binding_sha256": sha256(args.binding_dir / "binding.json"),
            "plans": {n: portable_plan(p) for n, p in plans.items()},
            "opponent_sources": {
                name: sha256(ROOT / "agent_code" / name / "callbacks.py")
                for name in ("peaceful_agent", "coin_collector_agent")
            },
        }
        if auth_path.exists():
            auth = read_json(auth_path)
            require(args.resume and auth["identity"] == identity, "Authorization/resume mismatch")
            require((root / "resources.json").is_file(), "Missing retained resource record")
        else:
            require(not args.resume, "Cannot resume without authorization record")
            require(
                not (root / "plans").exists(),
                "Existing plan outputs require original authorization",
            )
            auth = {"identity": identity, "authorized_at": datetime.now(timezone.utc).isoformat()}
            auth_path.write_text(json.dumps(auth, indent=2) + "\n", encoding="utf-8")
            (root / "preflight.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
        campaign = {
            "issue": config["issue"],
            "opponent_seed_policy": "task3_per_slot_v1",
            "authorization_sha256": sha256(auth_path),
            "reviewed_commit": args.reviewed_commit,
        }
        monitor = CampaignResourceMonitor(
            state_path=root / "resources.json",
            authorized_at=auth["authorized_at"],
            limits=CampaignLimits(
                limits["cpu_hours"] * 3600,
                limits["wall_hours"] * 3600,
                limits["memory_gib"] * 1024**3,
            ),
            campaign_metadata=campaign,
        )
        # Serial plans share a monitor; only candidate training uses up to two workers.
        for name in ("reference", "candidate"):
            monitor.check()
            plan = plans[name]
            execute_plan(
                plan,
                output_root=root / "plans",
                resume=args.resume and (root / "plans" / plan.plan_id).exists(),
                process_monitor=monitor,
            )
        monitor.check()
    finally:
        lock.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("peaceful", "coincollector"), default="peaceful")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--binding-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--authorize-compute", action="store_true")
    parser.add_argument("--reviewed-commit")
    parser.add_argument("--authorized-by")
    parser.add_argument("--hardware-description")
    parser.add_argument("--available-memory-gib", type=float)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.output_root is None:
        issue = 109 if args.protocol == "peaceful" else 137
        args.output_root = ROOT / "training_outputs" / f"issue{issue}-campaign"
    if args.dry_run:
        print(json.dumps(validate_protocol(args.binding_dir, args.protocol)[2], indent=2))
    else:
        execute(args)


def verify_prerequisite(directory, binding):
    from training.analyze_task3_campaign import verify_compact

    prerequisite = relative_file(directory, binding["prerequisite"]["path"])
    require(
        sha256(prerequisite / "result.json") == binding["prerequisite"]["result_sha256"],
        "Peaceful result changed",
    )
    require(
        verify_compact(prerequisite)["status"] == "exploratory_pass",
        "Coin-collector requires a passing peaceful decision",
    )
    result = read_json(prerequisite / "result.json")
    require(
        result.get("selected_artifact_sha256") == binding["parent"]["sha256"],
        "Parent is not the mechanically selected peaceful checkpoint",
    )


if __name__ == "__main__":
    main()
