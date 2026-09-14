"""Recompute issue168 exposure summaries from compact retained episode evidence."""

from __future__ import annotations

import argparse
import gzip
import json
import random
import tarfile
import tempfile
from pathlib import Path
from statistics import mean

from scripts.probe_task3_exploration import decide, sha, write

METRICS = (
    "survival_steps",
    "coins",
    "crates",
    "eliminations",
    "self_kills",
    "survived",
    "invalid_actions",
    "attack_steps",
    "safe_attack_steps",
    "available_safe_attack_steps",
    "safe_cratefree_placements",
)


def interval(values):
    rng = random.Random(1682026)
    boot = sorted(mean(rng.choices(values, k=len(values))) for _ in range(10000))
    return [boot[249], boot[9749]]


def analyze(root):
    status = json.loads((root / "status.json").read_text())
    if status["status"] != "completed":
        raise ValueError("Cannot analyze incomplete matrix")
    jobs = status["jobs"]
    phase_c = status.get("phase") == "C"
    if phase_c:
        from scripts.task3_episode_exploration import decide as decide_c

        decision = decide_c(jobs)
    else:
        decision = decide(jobs)
    for job in jobs:
        episode = root / job["output"]
        for name, field in (
            ("observations.json.gz", "observations_sha256"),
            ("framework_stats.json", "framework_stats_sha256"),
        ):
            if sha(episode / name) != job[field]:
                raise ValueError("Episode evidence checksum mismatch")
        with gzip.open(episode / "observations.json.gz", "rt") as file:
            rows = json.load(file)
        native = json.loads((episode / "framework_stats.json").read_text())
        rounds = list(native["by_round"].values())
        if len(rounds) != 1:
            raise ValueError("Expected one native episode")
        result = rounds[0]["agents"]["DagobertDuckDQNTask3"]
        mapping = {
            "survival_steps": "survival_steps",
            "coins": "coins",
            "crates": "crates_destroyed",
            "eliminations": "kills",
            "self_kills": "self_kills",
            "invalid_actions": "invalid",
            "score": "score",
        }
        if any(job[key] != result[value] for key, value in mapping.items()):
            raise ValueError("Native metric mismatch")
        if len(rows) != job["survival_steps"]:
            raise ValueError("Observation length mismatch")
        mapping = {
            "attack_steps": "attack",
            "safe_attack_steps": "safe_attack",
            "available_safe_attack_steps": "available_safe_attack",
            "safe_cratefree_placements": "safe_cratefree_placement",
        }
        if any(job[key] != sum(row[value] for row in rows) for key, value in mapping.items()):
            raise ValueError("Instrumented metric mismatch")
    if phase_c:
        groups = {
            "stepwise": sorted(
                (j for j in jobs if j["epsilon"] == 0.2), key=lambda j: j["world_seed"]
            ),
            "episode_mixture": sorted(
                (j for j in jobs if j["epsilon"] != 0.2), key=lambda j: j["world_seed"]
            ),
        }
        reference, treatment = groups["stepwise"], groups["episode_mixture"]
    else:
        groups = {
            str(e): sorted((j for j in jobs if j["epsilon"] == e), key=lambda j: j["world_seed"])
            for e in (1.0, 0.2, 0.0)
        }
        reference, treatment = groups["1.0"], groups["0.2"]
    summary = {
        eps: {
            metric: {"mean": mean(j[metric] for j in rows), "total": sum(j[metric] for j in rows)}
            for metric in METRICS
        }
        for eps, rows in groups.items()
    }
    paired = {}
    for metric in METRICS:
        differences = [b[metric] - a[metric] for a, b in zip(reference, treatment, strict=True)]
        paired[metric] = {
            "mean_difference": mean(differences),
            "paired_bootstrap_95_percent": interval(differences),
        }
    return {
        "decision": decision,
        "summary": summary,
        ("paired_mixture_minus_stepwise" if phase_c else "paired_low_minus_control"): paired,
        "uncertainty": (
            "Descriptive paired-world bootstrap; 10000 draws, seed1682026. "
            "Not a Task3 efficacy gate."
        ),
        "cpu_seconds_including_failed_attempts": status["cpu_seconds"],
        "wall_seconds_including_failed_attempts": status["wall_seconds"],
        "peak_memory_bytes": status["peak_memory_bytes"],
    }


def history(path):
    with gzip.open(path, "rt") as file:
        data = json.load(file)
    control = [row for row in data["training"] if row["arm"] == "control"]
    if len(control) != 25000:
        raise ValueError("Expected all 25000 original control episodes")
    grouped = []
    for replica in range(1, 6):
        rows = sorted(
            (r for r in control if r["replica"] == f"r{replica}"), key=lambda r: r["round"]
        )
        if len(rows) != 5000 or {r["round"] for r in rows} != set(range(1, 5001)):
            raise ValueError("Missing/duplicate replica episodes")
        for offset in range(0, 5000, 1000):
            block = rows[offset : offset + 1000]
            grouped.append(
                {
                    "replica": f"r{replica}",
                    "episode_start": offset + 1,
                    "episode_end": offset + 1000,
                    "episodes": len(block),
                    **{
                        key: {
                            "total": sum(r[key] for r in block),
                            "mean": mean(r[key] for r in block),
                        }
                        for key in (
                            "survival_steps",
                            "coins_collected",
                            "crates_destroyed",
                            "opponents_eliminated",
                            "self_kills",
                            "survived",
                            "epsilon",
                            "shaped_reward",
                        )
                    },
                }
            )
    return {
        "source_observations_sha256": sha(path),
        "control_episodes": len(control),
        "opponents_eliminated": sum(r["opponents_eliminated"] for r in control),
        "blocks": grouped,
        "scope": "Retrospective motivation, not independent confirmation of the next treatment",
    }


def export(root, output):
    analyze(root)
    status = json.loads((root / "status.json").read_text())
    paths = [
        (root / name, name) for name in ("status.json", "registration.json", "source-manifest.json")
    ]
    for job in status["jobs"]:
        for name in ("observations.json.gz", "framework_stats.json", "result.json"):
            paths.append((root / job["output"] / name, job["output"] + "/" + name))
    previous = status.get("previous_attempt")
    if previous:
        path = Path(previous["path"])
        if sha(path) != previous["sha256"]:
            raise ValueError("Failed attempt changed")
        paths.append((path, "failed-attempt/status.json"))
        for native in sorted(path.parent.glob("*/framework_stats.json")):
            paths.append((native, "failed-attempt/" + native.parent.name + "/framework_stats.json"))
    with (
        output.open("xb") as target,
        gzip.GzipFile(filename="", fileobj=target, mode="wb", mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for path, name in paths:
            info = archive.gettarinfo(path, arcname=name)
            info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ""
            with path.open("rb") as file:
                archive.addfile(info, file)
    return {
        "sha256": sha(output),
        "size_bytes": output.stat().st_size,
        "files": len(paths),
        "contents": (
            f"{len(status['jobs'])} native episodes and compact step observations, "
            "registration/source hashes and failed-attempt record; no checkpoints or raw logs"
        ),
    }


def analyze_input(path):
    if path.is_dir():
        return analyze(path)
    with tempfile.TemporaryDirectory(prefix="issue168-analysis-") as directory:
        with tarfile.open(path) as archive:
            archive.extractall(directory, filter="data")
        return analyze(Path(directory))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["analyze", "history", "export"])
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.mode == "export":
        print(json.dumps(export(args.input, args.output), indent=2))
    else:
        write(
            args.output,
            analyze_input(args.input) if args.mode == "analyze" else history(args.input),
        )


if __name__ == "__main__":
    main()
