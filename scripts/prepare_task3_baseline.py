"""Prepare an explicitly exploratory Task 3 baseline and matched plan templates.

This does not authorize scientific training or replace the committed fixture.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def prepare(parent: Path, parent_sha256: str, output: Path, source_commit: str) -> dict:
    from agent_code.DagobertDuckDQNTask2.persistence import load_evaluation_checkpoint
    from scripts.migrate_task3_dqn_successor import sha256_file

    parent, output = parent.resolve(), output.resolve()
    if sha256_file(parent) != parent_sha256:
        raise ValueError("Parent checksum mismatch")
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("Parent source must be a full commit SHA")
    if output.exists():
        raise ValueError("Output directory already exists; preserve the previous baseline")
    loaded = load_evaluation_checkpoint(parent)
    output.mkdir(parents=True)
    frozen_parent = output / "task2-parent.pt"
    successor = output / "task3-baseline.pt"
    shutil.copyfile(parent, frozen_parent)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/migrate_task3_dqn_successor.py"),
            "--parent",
            str(frozen_parent),
            "--parent-sha256",
            parent_sha256,
            "--output",
            str(successor),
        ],
        cwd=ROOT,
        check=True,
    )
    modes = {
        "action_masking": "framework_legal" if loaded.config.action_masking else "none",
        "escape_continuations": "on" if loaded.config.escape_continuation_features else "off",
    }
    plans = {}
    for name, source, checkpoint in (
        ("candidate", "issue109-task3-vs-peaceful.yaml", successor),
        ("reference", "issue109-task2-predecessor-vs-peaceful.yaml", frozen_parent),
        ("smoke", "issue108-dqn-task3-smoke.yaml", successor),
    ):
        data = yaml.safe_load((ROOT / "training/run_plans" / source).read_text())
        data.update(modes)
        data["plan_id"] = "issue125-exploratory-" + name
        for replica in data["replicas"]:
            replica["parent_artifact"] = str(checkpoint)
        path = output / (name + ".yaml")
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8", newline="\n")
        plans[name] = {"path": path.name, "sha256": sha256_file(path)}
    binding = {
        "schema_version": 1,
        "status": "exploratory_baseline_pending_issue124",
        "scientific_training_authorized": False,
        "task2_complete": False,
        "parent": {
            "path": frozen_parent.name,
            "sha256": parent_sha256,
            "size_bytes": frozen_parent.stat().st_size,
            "source_commit": source_commit,
        },
        "successor": {
            "path": successor.name,
            "sha256": sha256_file(successor),
            "size_bytes": successor.stat().st_size,
            "input_dim": 39,
        },
        "modes": modes,
        "preparation_source": {
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "files": {
                path.relative_to(ROOT).as_posix(): sha256_file(path)
                for path in [
                    ROOT / "scripts/migrate_task3_dqn_successor.py",
                    *sorted((ROOT / "agent_code/DagobertDuckDQNTask3").rglob("*.py")),
                ]
            },
        },
        "plans": plans,
        "warning": ("Historical exploratory baseline only; "
                    "do not substitute for Issue 124 selection"),
    }
    (output / "binding.json").write_text(json.dumps(binding, indent=2) + "\n", encoding="utf-8")
    return binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--parent-source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.parent, args.parent_sha256, args.output_dir, args.parent_source_commit),
            indent=2,
        )
    )


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
