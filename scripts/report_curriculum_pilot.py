"""Verify retained issue217 observations and report all stopped pilot checkpoints.

This is a descriptive audit of the registered pilot, not a new selection screen.
No games or gradient updates are run. Read outputs in the newly created directory.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.analyze_hunting_curriculum import analyze, diagnostic_summary, metric, safety_gate
from scripts.curriculum_io import read, require, write


def original_bytes(content, checksum):
    """Undo Git newline conversion only when the original hash proves exact bytes."""
    lf = content.replace(b"\r\n", b"\n")
    for candidate in (content, lf, lf.replace(b"\n", b"\r\n")):
        if hashlib.sha256(candidate).hexdigest() == checksum:
            return candidate
    raise ValueError("Evidence checksum mismatch")


def unpack(source, destination):
    index = read(source / "evidence-index.json")
    for name, record in index.items():
        stored = (source / record["stored"]).resolve()
        target = (destination / name).resolve()
        require(stored.is_relative_to(source.resolve()), "Unsafe stored path")
        require(target.is_relative_to(destination.resolve()), "Unsafe output path")
        packed = original_bytes(stored.read_bytes(), record["stored_sha256"])
        content = gzip.decompress(packed) if stored.suffix == ".gz" else packed
        require(len(content) == record["bytes"], "Evidence size mismatch")
        require(hashlib.sha256(content).hexdigest() == record["sha256"], "Original hash mismatch")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return destination


def suite_summary(path, setting, reference_sha=None):
    evidence = read(path)
    if reference_sha:
        require(evidence["checkpoint_sha256"] == reference_sha, "Reference binding mismatch")
    rows = evidence["rows"]
    require([r["world_seed"] for r in rows] == list(range(
        setting["seeds"][0], setting["seeds"][1] + 1)), "Wrong pilot seeds")
    require(all(r["epsilon"] == 0 and r["opponents"] == setting["opponents"]
                and r["scenario"] == setting["scenario"]
                and r["slot"] == i % (len(setting["opponents"]) + 1)
                for i, r in enumerate(rows)), "Wrong pilot conditions")
    return {
        "games": len(rows),
        "checkpoint_sha256": evidence["checkpoint_sha256"],
        "metrics": {key: {
            "mean": float(np.mean([metric(r, key) for r in rows])),
            "std_across_worlds": float(np.std([metric(r, key) for r in rows], ddof=1)),
        } for key in ("score", "kills", "survived", "survival_steps", "self_kills",
                      "coins", "crates_destroyed", "invalid", "collection_fraction")},
        "diagnostics": diagnostic_summary(rows),
    }


def report(sources, output):
    require(not output.exists(), "Use a new output directory; preserve previous reports")
    output.mkdir(parents=True)
    roots = [unpack(source, output / source.name) for source in sources]
    combined = analyze(roots)
    require(not combined["complete"], "This report is only for the stopped pilot")
    evaluations, gates = {}, {}
    for root in roots:
        cfg = read(root / "config.json")
        for pair in sorted((root / "pairs").iterdir()):
            reference = pair / "reference/pilot"
            for suite, setting in cfg["evaluation"]["pilot"].items():
                evaluations[f"{pair.name}/reference/{suite}"] = suite_summary(
                    reference / f"{suite}.json", setting, cfg["reference_sha256"])
            for gate_path in sorted(pair.glob("gate-*.json")):
                label = gate_path.stem.removeprefix("gate-")
                saved = read(gate_path)
                for arm in cfg["arms"]:
                    directory = pair / arm / "snapshots" / label / "evaluation"
                    recomputed = safety_gate(directory, reference, cfg["pilot"])
                    require(
                        recomputed == saved["arms"][arm], "Saved gate differs from recomputation"
                    )
                    gates[f"{pair.name}/{arm}/{label}"] = recomputed
                    for suite, setting in cfg["evaluation"]["pilot"].items():
                        evaluations[f"{pair.name}/{arm}/{label}/{suite}"] = suite_summary(
                            directory / f"{suite}.json", setting)
    write(output / "registered-analysis.json", combined)
    write(output / "pilot-evaluations.json", {
        "interpretation": "Descriptive repeated pilot worlds; no new selection or causal claim. "
        "Reference worlds repeat across replicas, not independent replication. "
        "Final and confirmation suites were not reached.",
        "gates_recomputed": gates, "evaluations": evaluations,
    })
    return evaluations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    values = report(args.evidence, args.output)
    for key, value in values.items():
        if key.endswith("/classic"):
            print(key, json.dumps({k: v["mean"] for k, v in value["metrics"].items()}))


if __name__ == "__main__":
    main()
