"""Export reviewable Issue171 tables from the unchanged registered analyzer.

No games, source mutations, model selection, README or AI-log writes occur here.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path
from statistics import mean

from scripts.analyze_task3_learning_rate import analyze, metrics
from scripts.pilot_task3_learning_rate import artifacts, read_json, sha


def write_bytes(path, value):
    if path.exists() and path.read_bytes() != value:
        raise ValueError(f"Refusing to overwrite different result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def save_json(path, value):
    write_bytes(path, (json.dumps(value, indent=2) + "\n").encode())


def save_csv(path, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    write_bytes(path, stream.getvalue().encode())


def arm_summary(analysis):
    result = []
    for suite, models in analysis["summary"].items():
        for arm in ("reference", "control", "treatment"):
            values = [v for k, v in models.items() if k == arm or k.startswith(arm + "-")]
            expected = 1 if arm == "reference" else 3
            if len(values) != expected:
                raise ValueError("Incomplete replica summary")
            result.append(
                {
                    "suite": suite,
                    "arm": arm,
                    "replicas": expected,
                    "primary_episodes": expected * 5,
                    **{key: mean(v[key] for v in values) for key in values[0]},
                }
            )
    return result


def plot_effects(analysis, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    specs = [
        ("coin-heaven", "control", "collection_fraction", "Task 1 vs control", 0.10),
        ("classic-peaceful", "control", "eliminations", "Hunting vs control", 0),
        ("coin-heaven", "reference", "collection_fraction", "Coin heaven vs reference", -0.05),
        ("loot-crate", "reference", "collection_fraction", "Loot crate vs reference", -0.05),
        ("classic-empty", "reference", "collection_fraction", "Classic empty vs reference", -0.05),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), width_ratios=[1, 1])
    for ax, selected in zip(axes, (specs[:2], specs[2:]), strict=True):
        for i, (suite, comp, metric, _label, gate) in enumerate(selected):
            effect = analysis["paired_treatment_minus"][suite][comp][metric]
            value = effect["mean_difference"]
            lo, hi = effect["paired_crossed_bootstrap_95_percent"]
            ax.errorbar(
                value, i, xerr=[[value - lo], [hi - value]], fmt="o", color="#176b87", capsize=5
            )
            ax.plot(gate, i, marker="|", markersize=20, color="#b04b35")
        ax.set_yticks(range(len(selected)), [s[3] for s in selected])
        ax.invert_yaxis()
        ax.axvline(0, color="#777777", linewidth=0.7)
        ax.set_xlim(-0.6, 0.85)
        ax.set_xlabel("Treatment minus comparator (absolute units)")
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Issue 171: learning-rate retention pilot failed", fontsize=14)
    fig.text(
        0.5,
        0.02,
        "Points: paired mean effects; lines: descriptive 95% crossed bootstrap intervals.\n"
        "Red ticks: registered minimum effects. Three replicas, five common worlds; "
        "no checkpoint selected.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160, metadata={"Software": "matplotlib"})
    plt.close(fig)
    write_bytes(path, buffer.getvalue())


def export(root, output, archive):
    verified = analyze(root)
    if verified != read_json(root / "analysis.json"):
        raise ValueError("Registered reanalysis differs from saved analysis")
    archive_meta = read_json(archive.with_suffix(archive.suffix + ".manifest.json"))
    if (
        sha(archive) != archive_meta["sha256"]
        or archive.stat().st_size != archive_meta["size_bytes"]
    ):
        raise ValueError("Archive checksum or size differs")
    cfg = read_json(root / "config.json")
    states = {
        stage: read_json(root / f"{stage}-state.json") for stage in ("training", "evaluation")
    }
    records = {"training": [], "evaluation": []}
    runs = []
    for stage, state in states.items():
        for job, relative in state["completed"].items():
            directory = root / relative
            with gzip.open(directory / "episodes.json.gz", "rt") as file:
                rows = json.load(file)
            records[stage].extend({"job": job, **row} for row in rows)
            runs.append(
                {
                    "stage": stage,
                    "job": job,
                    "relative_path": relative,
                    **read_json(directory / "result.json"),
                }
            )
    if len(records["training"]) != 300 or len(records["evaluation"]) != 280:
        raise ValueError("Wrong retained episode counts")
    save_json(output / "analysis.json", verified)
    write_bytes(
        output / "observations.json.gz",
        gzip.compress(
            json.dumps(
                {"schema_version": 1, **records, "runs": runs},
                sort_keys=True,
                separators=(",", ":"),
            ).encode(),
            mtime=0,
        ),
    )
    save_csv(output / "summary.csv", arm_summary(verified))
    save_csv(
        output / "per-replica.csv",
        [
            {"suite": suite, "artifact": key, **value}
            for suite, models in verified["summary"].items()
            for key, value in models.items()
        ],
    )
    save_csv(
        output / "contrasts.csv",
        [
            {
                "suite": suite,
                "contrast": "treatment-minus-" + comp,
                "metric": metric,
                "mean_difference": value["mean_difference"],
                "descriptive_95_lower": value["paired_crossed_bootstrap_95_percent"][0],
                "descriptive_95_upper": value["paired_crossed_bootstrap_95_percent"][1],
            }
            for suite, comparisons in verified["paired_treatment_minus"].items()
            for comp, effects in comparisons.items()
            for metric, value in effects.items()
        ],
    )
    save_csv(
        output / "gates.csv",
        [
            {"scope": "overall", "gate": key, "passed": value}
            for key, value in verified["gates"].items()
        ]
        + [
            {"scope": suite, "gate": key, "passed": value}
            for suite, gates in verified["retention_suites"].items()
            for key, value in gates.items()
        ],
    )
    save_csv(
        output / "training-summary.csv",
        [{"artifact": key, **value} for key, value in verified["training_summary"].items()],
    )
    save_csv(
        output / "episodes.csv",
        [
            {
                "stage": stage,
                "job": row["job"],
                "world_seed": row["world_seed"],
                "agent_seed": row["agent_seed"],
                "repeat": row.get("repeat"),
                "behavior_epsilon": row["behavior_epsilon"],
                **metrics(row),
            }
            for stage, rows in records.items()
            for row in rows
        ],
    )
    save_json(
        output / "invalid-actions.json",
        [
            {
                "artifact": row["artifact"],
                "suite": row["suite"],
                "world_seed": row["world_seed"],
                "repeat": row["repeat"],
                "invalid_actions": metrics(row)["invalid_actions"],
                "cause": "not established by aggregate native observations; no new replay run",
            }
            for row in records["evaluation"]
            if metrics(row)["invalid_actions"]
        ],
    )
    save_csv(
        output / "checkpoints.csv",
        [
            {
                "artifact": key,
                "path_in_archive": path.relative_to(root).as_posix(),
                "sha256": sha(path),
                "size_bytes": path.stat().st_size,
                "selected": False,
                "learning_rate": None
                if key == "reference"
                else cfg["learning_rates"][key.split("-")[0]],
            }
            for key, path in artifacts(root).items()
        ],
    )
    members = []
    with tarfile.open(archive) as file:
        for member in file:
            if not member.isfile():
                raise ValueError("Unexpected non-file archive member")
            with file.extractfile(member) as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if sha(root / member.name) != digest:
                raise ValueError("Archive member differs from verified extraction")
            members.append({"path": member.name, "size_bytes": member.size, "sha256": digest})
    save_json(output / "archive-members.json", members)
    save_json(
        output / "execution-verification.json",
        {
            "reanalysis_matches_saved": True,
            "analysis_sha256": sha(root / "analysis.json"),
            "training_episodes": 300,
            "evaluation_episodes": 280,
            "behavioral_repeat_pairs": 140,
            "training_checkpoint_snapshots": len(
                list((root / "training").rglob("checkpoint-*.pt"))
            ),
            "runtime_source": cfg["runtime_source"],
            "binding": read_json(root / "binding.json"),
            "initialization": read_json(root / "initialization-verification.json"),
            "states": states,
            "resource_gates": {
                stage: {
                    "cpu": state["cpu_seconds"] <= cfg[f"{stage}_limits"]["cpu_seconds"],
                    "wall": state["wall_seconds"] <= cfg[f"{stage}_limits"]["wall_seconds"],
                    "memory": state["peak_memory_bytes"] <= cfg[f"{stage}_limits"]["memory_bytes"],
                }
                for stage, state in states.items()
            },
            "archive_members": len(members),
            "new_games_for_analysis": 0,
            "publication_status": "pending durable release; locally verified evidence only",
        },
    )
    save_json(
        output / "evidence-manifest.json",
        {
            **archive_meta,
            "archive_name": archive.name,
            "publication_status": "not published",
            "durable_url": None,
            "planned_release": "issue171-pilot-evidence-v1",
            "contents": "Immutable runtime archive and manifest; protocol/binding; "
            "initial and frozen inputs; "
            "all 300 episode checkpoints plus six final checkpoints; training/evaluation attempts, "
            "native observations, timings and registered analysis.",
            "schema": "Issue171 runner at "
            + archive_meta["tool_source"]
            + "; JSON native episodes, restricted PyTorch checkpoints",
            "retrieval_after_publication": (
                "gh release download issue171-pilot-evidence-v1 "
                "--repo 1BlauNitrox/mle-final-project --dir inputs-171"
            ),
            "verification_command": "python -m scripts.pilot_task3_learning_rate import "
            "--root training_outputs/issue171-review --archive inputs-171/"
            + archive.name
            + " --sha256 "
            + archive_meta["sha256"],
            "analysis_command": (
                "python -m scripts.pilot_task3_learning_rate analyze "
                "--root training_outputs/issue171-review "
                "--output training_outputs/issue171-review/recomputed-analysis.json"
            ),
            "required_tool_commit": archive_meta["tool_source"],
        },
    )
    plot_effects(verified, output / "figures/paired-effects.png")
    print(
        json.dumps(
            {
                "output": str(output),
                "pilot_screen_passed": verified["pilot_screen_passed"],
                "archive_sha256": archive_meta["sha256"],
                "members_verified": len(members),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    export(args.root.resolve(), args.output.resolve(), args.archive.resolve())
