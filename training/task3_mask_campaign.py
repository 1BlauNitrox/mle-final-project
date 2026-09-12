"""Prospective Issue 147: matched legal masking, guarded execution and portable evidence."""

from __future__ import annotations

import argparse
import gzip
import json
import tarfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

from scripts.prepare_task3_baseline import prepare as prepare_baseline
from training import analyze_task3_campaign as analysis
from training import run_task3_campaign as campaign
from training.run_issue107_campaign import _seed_values_from_path
from training.run_plan import load_plan

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/2026-09-12-task3-legal-mask/config.yaml"
PARENT = "c0b6081acf5a733258005d46b9430f3ef50e403470a3bb5428661729f9aeedf7"
PARENT_SOURCE = "cbd52be8392f5003a91c5600fda4efd544b48ec5"
ARMS = ("reference", "control", "masked")
require, sha256, read_json = campaign.require, campaign.sha256, campaign.read_json


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def masked_initialization(source, target):
    """Change only the registered mode on a fresh migration; never alter learned state."""
    from agent_code.DagobertDuckDQNTask3.persistence import load_training_checkpoint

    loaded = load_training_checkpoint(source)
    require(
        loaded.completed_episodes == loaded.learner.update_steps == len(loaded.replay_buffer) == 0,
        "Mask intervention requires fresh migration",
    )
    require(not loaded.config.action_masking, "Expected unmasked initialization")
    require(not target.exists(), "Preserve existing masked initialization")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    payload["config"]["action_masking"] = True
    torch.save(payload, target)
    verify_masked_initialization(source, target)


def verify_masked_initialization(source, target):
    from agent_code.DagobertDuckDQNTask3.persistence import load_training_checkpoint

    control, masked = (load_training_checkpoint(p) for p in (source, target))
    require(masked.config == replace(control.config, action_masking=True), "Unexpected config")
    left = torch.load(source, map_location="cpu", weights_only=True)
    right = torch.load(target, map_location="cpu", weights_only=True)
    left["config"]["action_masking"] = True

    def equal(a, b):
        if isinstance(a, torch.Tensor):
            return isinstance(b, torch.Tensor) and torch.equal(a, b)
        if isinstance(a, dict):
            return (
                isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
            )
        if isinstance(a, (list, tuple)):
            return (
                type(a) is type(b)
                and len(a) == len(b)
                and all(equal(x, y) for x, y in zip(a, b, strict=True))
            )
        return type(a) is type(b) and a == b

    require(equal(left, right), "Masking changed weights, optimizer, replay or RNG state")


def prepare(parent, output):
    require(sha256(parent) == PARENT, "Wrong registered #91 parent")
    require(not output.exists(), "Preserve existing binding")
    output.mkdir(parents=True)
    base = output / "migration"
    original = prepare_baseline(parent, PARENT, base, PARENT_SOURCE)
    masked_initialization(base / original["successor"]["path"], output / "masked.pt")
    binding = {"issue": 147, "task2_complete": False, "artifacts": {}, "plans": {}}
    config = yaml.safe_load(CONFIG.read_text())
    for arm in ARMS:
        source = (
            output / "masked.pt"
            if arm == "masked"
            else base / original["parent" if arm == "reference" else "successor"]["path"]
        )
        binding["artifacts"][arm] = {
            "path": source.relative_to(output).as_posix(),
            "sha256": sha256(source),
            "size_bytes": source.stat().st_size,
        }
        data = yaml.safe_load((ROOT / config["plans"][arm]).read_text())
        for replica in data["replicas"]:
            replica.update(
                parent_artifact=str(source.resolve()), parent_artifact_sha256=sha256(source)
            )
        path = output / (arm + ".yaml")
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8", newline="\n")
        binding["plans"][arm] = {"path": path.name, "sha256": sha256(path)}
    write_json(output / "binding.json", binding)
    validate(output)
    return binding


def validate(directory=None):
    config = yaml.safe_load(CONFIG.read_text())
    require(
        config["issue"] == 147 and not config["scientific_training_authorized"], "Wrong protocol"
    )
    templates = {a: load_plan(ROOT / config["plans"][a]) for a in ARMS}
    control, masked = (templates[a].to_dict() for a in ("control", "masked"))
    require(
        control["action_masking"] == "none" and masked["action_masking"] == "framework_legal",
        "Wrong masking intervention",
    )
    for value in (control, masked):
        value.pop("plan_id")
        value.pop("fingerprints")
        value.pop("action_masking")
    require(control == masked, "Training arms differ beyond masking")
    signatures = []
    for arm, plan in templates.items():
        budget = [
            sum(j.rounds for j in plan.jobs if j.kind == kind)
            for kind in ("training", "evaluation")
        ]
        require(
            budget == ([0, 320] if arm == "reference" else [50000, 1600]), "Wrong episode budget"
        )
        signatures.append(
            [
                (j.stage_or_suite, j.scenario, j.opponents, j.rounds, j.world_seed, j.agent_seed)
                for j in plan.jobs
                if j.kind == "evaluation" and j.replica == plan.replicas[0].replica_id
            ]
        )
    require(all(s == signatures[0] for s in signatures), "Unmatched evaluation")
    require(
        [(r.world_seed, r.agent_seed) for r in templates["control"].replicas]
        == [(r.world_seed, r.agent_seed) for r in templates["masked"].replicas],
        "Unpaired training",
    )
    seeds = {s for p in templates.values() for j in p.jobs for s in (j.world_seed, j.agent_seed)}
    protected = {s for start, end in config["protected_seed_ranges"] for s in range(start, end + 1)}
    require(not seeds & protected, "Protected seed used")
    own = {CONFIG, *(ROOT / p for p in config["plans"].values())}
    audit = {}
    for path in sorted(
        [
            *(ROOT / "training/run_plans").glob("*.yaml"),
            *(ROOT / "experiments").glob("*/config.yaml"),
        ]
    ):
        if path not in own:
            require(
                not (seeds | protected) & _seed_values_from_path(path), f"Seed collision: {path}"
            )
            audit[path.relative_to(ROOT).as_posix()] = sha256(path)
    plans = templates
    if directory is not None:
        directory = directory.resolve()
        binding = read_json(directory / "binding.json")
        require(binding["issue"] == 147 and binding["task2_complete"] is False, "Wrong binding")
        original = read_json(directory / "migration/binding.json")
        require(
            original["parent"]["sha256"] == PARENT
            and original["parent"]["source_commit"] == PARENT_SOURCE,
            "Wrong lineage",
        )
        campaign.verify_migration(directory / "migration", original)
        verify_masked_initialization(
            directory / "migration" / original["successor"]["path"], directory / "masked.pt"
        )
        plans = {}
        for arm, template in templates.items():
            record = binding["artifacts"][arm]
            artifact = campaign.relative_file(directory, record["path"])
            require(
                sha256(artifact) == record["sha256"]
                and artifact.stat().st_size == record["size_bytes"],
                "Artifact changed",
            )
            expected_path = (
                directory / "masked.pt"
                if arm == "masked"
                else directory
                / "migration"
                / original["parent" if arm == "reference" else "successor"]["path"]
            )
            require(artifact == expected_path.resolve(), "Wrong arm artifact")
            record = binding["plans"][arm]
            path = campaign.relative_file(directory, record["path"])
            require(sha256(path) == record["sha256"], "Plan changed")
            plan = load_plan(path)
            expected = template.to_dict()
            for replica in expected["replicas"]:
                replica.update(
                    parent_artifact=str(artifact), parent_artifact_sha256=sha256(artifact)
                )
            actual = plan.to_dict()
            actual.pop("fingerprints")
            expected.pop("fingerprints")
            require(actual == expected, "Bound plan differs from registered matrix")
            plans[arm] = plan
    return (
        config,
        plans,
        {
            "issue": 147,
            "training_episodes": 100000,
            "evaluation_episodes": 3520,
            "protocol_sha256": sha256(CONFIG),
            "seed_audit": audit,
            "parent_bound": directory is not None,
            "compute_authorized": False,
        },
    )


def paired_interval(masked, control, *, seed=147, samples=10000):
    masked, control = np.asarray(masked, float), np.asarray(control, float)
    require(masked.shape == control.shape and masked.ndim == 2, "Unpaired treatment arrays")
    require(np.isfinite(masked).all() and np.isfinite(control).all(), "Nonfinite observations")
    rng = np.random.default_rng(seed)
    model = rng.integers(masked.shape[0], size=(samples, masked.shape[0]))
    pairs = rng.integers(masked.shape[1], size=(samples, masked.shape[1]))
    draws = (masked - control)[model[:, :, None], pairs[:, None, :]].mean(axis=(1, 2))
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {"mean": float((masked - control).mean()), "lower": float(lower), "upper": float(upper)}


def decide(rows, config):
    arms = {}
    for arm in ("control", "masked"):
        selected = [
            dict(r, arm="candidate" if r["arm"] == arm else "reference")
            for r in rows
            if r["arm"] in (arm, "reference")
        ]
        arms[arm] = analysis.decide(selected, config)
    primary = [r for r in rows if r["suite"] == "classic-peaceful-primary"]
    models = sorted({r["replica"] for r in primary if r["arm"] == "masked"})
    arrays = {}
    for arm in ("control", "masked"):
        groups = [[r for r in primary if r["arm"] == arm and r["replica"] == m] for m in models]
        groups = [sorted(g, key=lambda r: (r["world_seed"], r["agent_seed"])) for g in groups]
        arrays[arm] = [[analysis.metric(r, "elimination") for r in g] for g in groups]
    effect = paired_interval(
        arrays["masked"],
        arrays["control"],
        seed=config["bootstrap"]["seed"],
        samples=config["bootstrap"]["samples"],
    )
    benefit = effect["mean"] >= config["primary_effect_min"] and effect["lower"] > 0
    passed = arms["masked"]["status"] == "exploratory_pass" and benefit
    return {
        "status": "exploratory_pass" if passed else "exploratory_mixed_or_negative",
        "task2_complete": False,
        "arms": arms,
        "masked_minus_control_elimination": effect,
        "treatment_gate": benefit,
        "selected_arm": "masked" if passed else None,
        "selected_replica": arms["masked"]["selected_replica"] if passed else None,
        "next_decision": "review_new_continuation_binding"
        if passed
        else "stop_no_automatic_continuation",
    }


def analyze(root, binding, output):
    config, rows, training, manifest, auth, resources = analysis.load_evidence(
        root.resolve(), binding.resolve(), validator=validate, config_path=CONFIG, issue=147
    )
    result = decide(rows, config)
    require(not output.exists(), "Preserve previous analysis")
    output.mkdir(parents=True)
    with gzip.open(output / "observations.json.gz", "wt", encoding="utf-8") as file:
        json.dump({"evaluation": rows, "training": training}, file)
    result.update(
        authorization=auth,
        resources=resources,
        protocol_sha256=sha256(CONFIG),
        observations_sha256=sha256(output / "observations.json.gz"),
    )
    write_json(output / "result.json", result)
    write_json(output / "source-manifest.json", manifest)
    verify_compact(output)
    return result


def verify_compact(directory):
    result = read_json(directory / "result.json")
    require(result["protocol_sha256"] == sha256(CONFIG), "Wrong compact protocol")
    require(
        result["observations_sha256"] == sha256(directory / "observations.json.gz"),
        "Changed observations",
    )
    with gzip.open(directory / "observations.json.gz", "rt", encoding="utf-8") as file:
        rows = json.load(file)["evaluation"]
    computed = decide(rows, yaml.safe_load(CONFIG.read_text()))
    require(all(result[k] == v for k, v in computed.items()), "Compact decision mismatch")
    return {"verified": True, "status": result["status"]}


def export(root, binding, analysis_dir, output):
    """Retain required evidence and every failure; omit redundant staged source/log copies."""
    verify_compact(analysis_dir)
    require(not output.exists(), "Preserve previous export")
    for name, record in read_json(analysis_dir / "source-manifest.json").items():
        path = campaign.relative_file(root, name)
        require(
            sha256(path) == record["sha256"] and path.stat().st_size == record["size_bytes"],
            "Source evidence changed after analysis",
        )
    names = {
        "authorization.json",
        "resources.json",
        "preflight.json",
        "status.json",
        "resolved_plan.json",
        "metadata.json",
        "framework_stats.json",
        "episodes.csv",
    }
    files = [
        (p, "campaign/" + p.relative_to(root).as_posix())
        for p in root.rglob("*")
        if p.is_file()
        and (
            p.name in names
            or (
                p.suffix == ".pt"
                and bool({"artifacts", "replicas"} & set(p.relative_to(root).parts))
            )
        )
    ]
    files += [
        (p, "binding/" + p.relative_to(binding).as_posix())
        for p in binding.rglob("*")
        if p.is_file()
    ]
    files += [
        (p, "analysis/" + p.relative_to(analysis_dir).as_posix())
        for p in analysis_dir.iterdir()
        if p.is_file()
    ]
    manifest = {name: {"sha256": sha256(p), "size_bytes": p.stat().st_size} for p, name in files}
    require(len(manifest) == len(files), "Duplicate export path")
    with tarfile.open(output, "x:gz") as archive:
        for path, name in files:
            archive.add(path, arcname=name, recursive=False)
    record = {"sha256": sha256(output), "size_bytes": output.stat().st_size, "files": manifest}
    write_json(output.with_suffix(output.suffix + ".manifest.json"), record)
    return {k: record[k] for k in ("sha256", "size_bytes")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("prepare", "dry-run", "run", "analyze", "verify", "export")
    )
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--binding-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--analysis-dir", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--authorize-compute", action="store_true")
    parser.add_argument("--reviewed-commit")
    parser.add_argument("--authorized-by")
    parser.add_argument("--hardware-description")
    parser.add_argument("--available-memory-gib", type=float)
    parser.add_argument("--allocation-hours", type=float)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        value = prepare(args.parent, args.binding_dir)
    elif args.action == "dry-run":
        value = validate(args.binding_dir)[2]
    elif args.action == "run":
        require(
            args.allocation_hours is not None and args.allocation_hours >= 11,
            "At least 11 hours allocation includes 10-hour campaign and export margin",
        )
        campaign.execute(args, validator=validate, issue=147, plan_order=ARMS)
        value = {"completed": True}
    elif args.action == "analyze":
        result = analyze(args.output_root, args.binding_dir, args.analysis_dir)
        value = {k: result[k] for k in ("status", "selected_arm", "selected_replica")}
    elif args.action == "verify":
        value = verify_compact(args.analysis_dir)
    else:
        value = export(args.output_root, args.binding_dir, args.analysis_dir, args.archive)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
