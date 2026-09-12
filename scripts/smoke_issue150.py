"""Run a bounded, isolated 26-episode mechanics check; never scientific evidence."""

import argparse
import gzip
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_code.DagobertDuckDQNTask3.persistence import load_training_checkpoint  # noqa: E402
from training import task3_double_campaign as double  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.binding_dir = args.binding_dir.resolve()
    args.output_root = args.output_root.resolve()
    double.require(not args.output_root.exists(), "Preserve earlier smoke outputs")
    config, plans, report = double.validate(args.binding_dir)
    config["resources"] = dict(
        config["resources"], cpu_hours=0.25, wall_hours=0.25, memory_gib=4, max_parallel_training=1
    )
    for arm, plan in plans.items():
        replica = plan.replicas[0]
        selected = []
        for job in plan.jobs:
            if job.replica != replica.replica_id:
                continue
            if job.kind == "training" or job.world_seed in (1501101, 1501201, 1501301, 1501401):
                # Separate mechanics seeds, absent from the scientific matrix.
                selected.append(
                    replace(
                        job,
                        rounds=1,
                        world_seed=job.world_seed + 9000000,
                        agent_seed=job.agent_seed + 9000000,
                    )
                )
        plans[arm] = replace(
            plan, replicas=(replica,), jobs=tuple(selected), max_parallel_training=1
        )
    report.update(
        training_episodes=2, evaluation_episodes=24, evidence_scope="implementation_smoke_only"
    )

    def validator(_):
        return config, plans, report

    args.authorize_compute = True
    args.reviewed_commit = double.campaign.git("rev-parse", "HEAD")
    args.authorized_by = "Julius-authorized-short-setup-smoke"
    args.hardware_description = "Isolated serial mechanics smoke; not scientific evidence"
    args.available_memory_gib = 4
    args.resume = False
    os.environ["BOMBERMAN_COMPACT_LOGS"] = "1"
    double.campaign.execute(args, validator=validator, issue=150, plan_order=double.ARMS)
    _, rows, training, manifest, auth, resources = double.analysis.load_evidence(
        args.output_root,
        args.binding_dir,
        validator=validator,
        config_path=double.CONFIG,
        issue=150,
    )
    assert len(rows) == 24 and len(training) == 2
    for arm in double.ARMS[1:]:
        checkpoint = (
            args.output_root
            / "plans"
            / ("issue150-" + arm)
            / "artifacts/r1/classic-peaceful/checkpoint.pt"
        )
        loaded = load_training_checkpoint(checkpoint)
        assert loaded.completed_episodes == 1
        assert loaded.config.double_dqn == (arm == "double")
    metadata_files = list(args.output_root.glob("plans/*/jobs/*/attempt-*/metadata.json"))
    assert len(metadata_files) == 26
    for metadata in metadata_files:
        assert double.read_json(metadata)["logging_policy"] == "warning_only"
    output = args.output_root / "smoke-analysis"
    output.mkdir()
    with gzip.open(output / "observations.json.gz", "wt", encoding="utf-8") as file:
        json.dump({"evaluation": rows, "training": training}, file)
    try:
        double.decide(rows, config)
    except ValueError as error:
        assert str(error) == "Require all five candidate replicas"
    else:
        raise AssertionError("Incomplete smoke must not produce a scientific decision")
    double.write_json(output / "source-manifest.json", manifest)
    double.write_json(
        output / "mechanics.json",
        {
            "scope": "implementation_smoke_only",
            "episodes": 26,
            "scientific_analyzer_rejects_incomplete_matrix": True,
            "authorization": auth,
            "resources": resources,
        },
    )
    print(
        json.dumps(
            {
                "mechanics_verified": True,
                "episodes": 26,
                "scientific_evidence": False,
                "resources": resources,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
