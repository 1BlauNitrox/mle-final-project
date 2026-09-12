"""Issue 150: matched standard/Double DQN, immutable execution and compact evidence."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import tarfile
from pathlib import Path

import torch
import yaml

from scripts.prepare_task3_baseline import prepare as migrate
from training import analyze_task3_campaign as analysis
from training import run_task3_campaign as campaign
from training import task3_mask_campaign as previous
from training.run_issue107_campaign import _seed_values_from_path
from training.run_plan import load_plan

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/2026-09-12-task3-double-dqn/config.yaml"
ARMS = ("reference", "control", "double")
require, sha256, read_json, write_json = (
    previous.require,
    previous.sha256,
    previous.read_json,
    previous.write_json,
)


def payload_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return (
            isinstance(b, dict)
            and a.keys() == b.keys()
            and all(payload_equal(a[k], b[k]) for k in a)
        )
    if isinstance(a, (list, tuple)):
        return (
            type(a) is type(b)
            and len(a) == len(b)
            and all(payload_equal(x, y) for x, y in zip(a, b, strict=True))
        )
    return type(a) is type(b) and a == b


def verify_initialization(directory):
    from agent_code.DagobertDuckDQNTask3.persistence import load_training_checkpoint

    control, double = [load_training_checkpoint(directory / (arm + ".pt")) for arm in ARMS[1:]]
    for loaded in (control, double):
        require(
            loaded.config.action_masking and loaded.config.escape_continuation_features,
            "Wrong inherited modes",
        )
        require(
            loaded.completed_episodes
            == loaded.learner.update_steps
            == len(loaded.replay_buffer)
            == 0,
            "Initialization must be fresh",
        )
    require(not control.config.double_dqn and double.config.double_dqn, "Wrong target intervention")
    left, right = [
        torch.load(directory / (arm + ".pt"), map_location="cpu", weights_only=True)
        for arm in ARMS[1:]
    ]
    left["config"]["double_dqn"] = True
    require(payload_equal(left, right), "Arms differ beyond Double DQN flag")
    original = read_json(directory / "migration/binding.json")
    require(
        original["parent"]["sha256"] == previous.PARENT
        and original["parent"]["source_commit"] == previous.PARENT_SOURCE,
        "Wrong original parent",
    )
    campaign.verify_migration(directory / "migration", original)
    previous.verify_masked_initialization(
        directory / "migration" / original["successor"]["path"], directory / "control.pt"
    )


def prepare(parent, output):
    require(sha256(parent) == previous.PARENT, "Wrong registered parent")
    require(not output.exists(), "Preserve original bindings")
    output.mkdir(parents=True)
    original = migrate(parent, previous.PARENT, output / "migration", previous.PARENT_SOURCE)
    previous.masked_initialization(
        output / "migration" / original["successor"]["path"], output / "control.pt"
    )
    payload = torch.load(output / "control.pt", map_location="cpu", weights_only=True)
    payload["config"]["double_dqn"] = True
    torch.save(payload, output / "double.pt")
    config = yaml.safe_load(CONFIG.read_text())
    binding = {"issue": 150, "task2_complete": False, "artifacts": {}, "plans": {}}
    for arm in ARMS:
        artifact = output / (
            "migration/" + original["parent"]["path"] if arm == "reference" else arm + ".pt"
        )
        binding["artifacts"][arm] = {
            "path": artifact.relative_to(output).as_posix(),
            "sha256": sha256(artifact),
            "size_bytes": artifact.stat().st_size,
        }
        data = yaml.safe_load((ROOT / config["plans"][arm]).read_text())
        for replica in data["replicas"]:
            replica.update(
                parent_artifact=str(artifact.resolve()), parent_artifact_sha256=sha256(artifact)
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
        config["issue"] == 150 and not config["scientific_training_authorized"], "Wrong protocol"
    )
    require(config["logging_policy"] == "warning_only", "Wrong registered logging policy")
    templates = {arm: load_plan(ROOT / config["plans"][arm]) for arm in ARMS}
    comparisons = []
    for arm in ("control", "double"):
        data = templates[arm].to_dict()
        require(
            data["action_masking"] == "framework_legal" and data["escape_continuations"] == "on",
            "Wrong shared modes",
        )
        for key in ("plan_id", "fingerprints"):
            data.pop(key)
        comparisons.append(data)
    require(comparisons[0] == comparisons[1], "Unmatched training arms")
    signatures = []
    for arm, plan in templates.items():
        require(
            [sum(j.rounds for j in plan.jobs if j.kind == k) for k in ("training", "evaluation")]
            == ([0, 320] if arm == "reference" else [50000, 1600]),
            "Wrong episode budget",
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
        config["gates"] == yaml.safe_load(previous.CONFIG.read_text())["gates"],
        "Earlier-task gates changed",
    )
    seeds = {s for p in templates.values() for j in p.jobs for s in (j.world_seed, j.agent_seed)}
    protected = {s for a, b in config["protected_seed_ranges"] for s in range(a, b + 1)}
    require(not seeds & protected, "Reserved seed used")
    own = {CONFIG, *(ROOT / p for p in config["plans"].values())}
    audit = {}
    for path in [
        *(ROOT / "training/run_plans").glob("*.yaml"),
        *(ROOT / "experiments").glob("*/config.yaml"),
    ]:
        if path not in own:
            require(
                not (seeds | protected) & _seed_values_from_path(path), f"Seed collision: {path}"
            )
            audit[path.relative_to(ROOT).as_posix()] = sha256(path)
    plans = templates
    if directory is not None:
        directory = directory.resolve()
        binding = read_json(directory / "binding.json")
        require(binding["issue"] == 150 and binding["task2_complete"] is False, "Wrong binding")
        verify_initialization(directory)
        plans = {}
        for arm, template in templates.items():
            record = binding["artifacts"][arm]
            artifact = campaign.relative_file(directory, record["path"])
            expected_name = "migration/task2-parent.pt" if arm == "reference" else arm + ".pt"
            require(record["path"] == expected_name, "Wrong arm artifact")
            require(
                sha256(artifact) == record["sha256"]
                and artifact.stat().st_size == record["size_bytes"],
                "Artifact changed",
            )
            record = binding["plans"][arm]
            path = campaign.relative_file(directory, record["path"])
            require(sha256(path) == record["sha256"], "Bound plan changed")
            plan = load_plan(path)
            expected = template.to_dict()
            for replica in expected["replicas"]:
                replica.update(
                    parent_artifact=str(artifact), parent_artifact_sha256=sha256(artifact)
                )
            actual = plan.to_dict()
            actual.pop("fingerprints")
            expected.pop("fingerprints")
            require(actual == expected, "Bound matrix changed")
            plans[arm] = plan
    return (
        config,
        plans,
        {
            "issue": 150,
            "training_episodes": 100000,
            "evaluation_episodes": 3520,
            "protocol_sha256": sha256(CONFIG),
            "parent_bound": directory is not None,
            "seed_audit": audit,
            "compute_authorized": False,
        },
    )


def decide(rows, config):
    mapped = [{**r, "arm": "masked" if r["arm"] == "double" else r["arm"]} for r in rows]
    result = previous.decide(mapped, config)
    result["arms"]["double"] = result["arms"].pop("masked")
    result["double_minus_control_elimination"] = result.pop("masked_minus_control_elimination")
    if result["selected_arm"] == "masked":
        result["selected_arm"] = "double"
    return result


def verify(directory):
    result = read_json(directory / "result.json")
    require(result["protocol_sha256"] == sha256(CONFIG), "Wrong evidence protocol")
    require(
        result["observations_sha256"] == sha256(directory / "observations.json.gz"),
        "Observations changed",
    )
    with gzip.open(directory / "observations.json.gz", "rt", encoding="utf-8") as file:
        rows = json.load(file)["evaluation"]
    computed = decide(rows, yaml.safe_load(CONFIG.read_text()))
    require(all(result[k] == v for k, v in computed.items()), "Decision mismatch")
    return {"verified": True, "status": result["status"]}


def analyze(root, binding, output):
    from agent_code.DagobertDuckDQNTask3.persistence import load_training_checkpoint

    config, rows, training, manifest, auth, resources = analysis.load_evidence(
        root.resolve(), binding.resolve(), validator=validate, config_path=CONFIG, issue=150
    )
    # Check the persisted intervention on every trained final, independently of labels.
    for path in (root / "plans").glob("*/status.json"):
        for record in read_json(path)["jobs"].values():
            run = campaign.relative_file(path.parent, record["attempts"][-1]["output"])
            require(
                read_json(run / "metadata.json")["logging_policy"] == "warning_only",
                "Executed logging policy mismatch",
            )
    for arm in ARMS[1:]:
        directory = root / "plans" / ("issue150-" + arm)
        status = read_json(directory / "status.json")
        for job in status["jobs"].values():
            if job["artifact"]["selection"] == "final checkpoint after the exact stage budget":
                loaded = load_training_checkpoint(
                    campaign.relative_file(directory, job["artifact"]["path"])
                )
                require(
                    loaded.completed_episodes == 10000
                    and loaded.config.double_dqn == (arm == "double")
                    and loaded.config.action_masking,
                    "Final intervention/count mismatch",
                )
    require(not output.exists(), "Preserve previous analysis")
    output.mkdir(parents=True)
    with gzip.open(output / "observations.json.gz", "wt", encoding="utf-8") as file:
        json.dump({"evaluation": rows, "training": training}, file)
    result = decide(rows, config)
    result.update(
        authorization=auth,
        resources=resources,
        protocol_sha256=sha256(CONFIG),
        observations_sha256=sha256(output / "observations.json.gz"),
    )
    write_json(output / "result.json", result)
    write_json(output / "source-manifest.json", manifest)
    verify(output)
    return {k: result[k] for k in ("status", "selected_arm", "selected_replica")}


def export(root, binding, analysis_dir, output):
    verify(analysis_dir)
    require(not output.exists(), "Preserve previous export")
    for name, record in read_json(analysis_dir / "source-manifest.json").items():
        path = campaign.relative_file(root, name)
        require(
            sha256(path) == record["sha256"] and path.stat().st_size == record["size_bytes"],
            "Evidence changed after analysis",
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
    files += [(p, "analysis/" + p.name) for p in analysis_dir.iterdir() if p.is_file()]
    manifest = {name: {"sha256": sha256(p), "size_bytes": p.stat().st_size} for p, name in files}
    require(len(manifest) == len(files), "Duplicate export member")
    with tarfile.open(output, "x:gz") as archive:
        for path, name in files:
            archive.add(path, arcname=name, recursive=False)
    record = {"sha256": sha256(output), "size_bytes": output.stat().st_size, "files": manifest}
    write_json(output.with_suffix(output.suffix + ".manifest.json"), record)
    return {k: record[k] for k in ("sha256", "size_bytes")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["prepare", "dry-run", "run", "analyze", "verify", "export"]
    )
    for name in ("parent", "binding-dir", "output-root", "analysis-dir", "archive"):
        parser.add_argument("--" + name, type=Path)
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
            "Reserve campaign plus export allocation",
        )
        os.environ["BOMBERMAN_COMPACT_LOGS"] = "1"
        campaign.execute(args, validator=validate, issue=150, plan_order=ARMS)
        value = {"completed": True}
    elif args.action == "analyze":
        value = analyze(args.output_root, args.binding_dir, args.analysis_dir)
    elif args.action == "verify":
        value = verify(args.analysis_dir)
    else:
        value = export(args.output_root, args.binding_dir, args.analysis_dir, args.archive)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
