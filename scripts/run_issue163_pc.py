"""Pinned serial PC campaign, complete analysis/export, retained resume and whole-allocation cap."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import sleep

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training import task3_attack_campaign as study  # noqa: E402
from training.run_issue107_campaign import CampaignLimits, CampaignResourceMonitor  # noqa: E402


def command(action, root, *extra):
    return [
        sys.executable,
        "-m",
        "training.task3_attack_campaign",
        action,
        "--binding-dir",
        str(root / "binding"),
        "--output-root",
        str(root / "runs"),
        *extra,
    ]


def next_output(root, stem, suffix=""):
    index = 1
    while (root / f"{stem}-{index:03d}{suffix}").exists():
        index += 1
    return root / f"{stem}-{index:03d}{suffix}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--authorize-compute", action="store_true")
    parser.add_argument("--review-note")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    root = args.root.resolve()
    study.require(study.campaign.git("rev-parse", "HEAD") == args.source, "Pinned source mismatch")
    study.require(not study.campaign.git("status", "--porcelain"), "Source must be clean")
    root.mkdir(parents=True, exist_ok=True)
    if not (root / "binding").exists():
        study.require(args.parent is not None and not args.resume, "Original parent required")
        study.prepare(args.parent, root / "binding")
    report = study.validate(root / "binding")[2]
    study.write_json(root / "preflight.json", report)
    if args.prepare_only:
        print(json.dumps(report, indent=2))
        return
    study.require(
        args.authorize_compute and args.review_note, "Explicit run and budget decision required"
    )
    import psutil

    study.require(psutil.virtual_memory().available >= 2 * 1024**3, "Need 2 GiB available RAM")
    if not (root / "run-completed").exists():
        study.preflight(study.validate(root / "binding")[1], root / "runs", ROOT)
    ownership = root / ".supervisor.lock"
    with ownership.open("x", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    identity = {
        "source": args.source,
        "protocol_sha256": study.sha256(study.CONFIG),
        "review_note": args.review_note,
        "allocation_hours": 12,
        "total_cpu_hours": 12,
        "memory_gib": 2,
        "binding_sha256": study.sha256(root / "binding/binding.json"),
    }
    monitor = None
    try:
        auth_path = root / "supervisor-authorization.json"
        if auth_path.exists():
            auth = study.read_json(auth_path)
            study.require(args.resume and auth["identity"] == identity, "Resume identity mismatch")
        else:
            study.require(not args.resume, "No authorization to resume")
            auth = {
                "identity": identity,
                "authorized_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }
            study.write_json(auth_path, auth)
        monitor = CampaignResourceMonitor(
            state_path=root / "supervisor-resources.json",
            authorized_at=auth["authorized_at"],
            limits=CampaignLimits(12 * 3600, 12 * 3600, 2 * 1024**3),
            campaign_metadata={"issue": 163, "scope": "training_evaluation_analysis_export"},
        )
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)

        def execute(cmd):
            monitor.check()
            print("Executing:", subprocess.list2cmdline(cmd), flush=True)
            process = subprocess.Popen(
                cmd, cwd=ROOT, env={**os.environ, "BOMBERMAN_COMPACT_LOGS": "1"}
            )
            registered = False
            try:
                monitor.register(process.pid)
                registered = True
                while process.poll() is None:
                    monitor.check()
                    sleep(0.2)
                study.require(
                    process.returncode == 0,
                    f"Stage failed with exit {process.returncode}; preserve outputs",
                )
            finally:
                # Resource breaches terminate only this registered process tree.
                if registered:
                    monitor.unregister(process.pid)
            monitor.check()

        if not (root / "run-completed").exists():
            execute(
                command(
                    "run",
                    root,
                    "--reviewed-commit",
                    args.source,
                    "--authorize-compute",
                    "--authorized-by",
                    "Julius",
                    "--hardware-description",
                    args.review_note + "; same Windows PC, serial, 2GiB cap",
                    "--available-memory-gib",
                    "2",
                    "--allocation-hours",
                    "12",
                    *(
                        ["--resume"]
                        if args.resume and (root / "runs/authorization.json").exists()
                        else []
                    ),
                )
            )
            (root / "run-completed").write_text(args.source, encoding="utf-8")
        study.require(
            (root / "run-completed").read_text() == args.source, "Completion source differs"
        )
        if (root / "analysis-completed.json").exists():
            analysis = root / study.read_json(root / "analysis-completed.json")["path"]
        else:
            analysis = next_output(root, "analysis")
            execute(command("analyze", root, "--analysis-dir", str(analysis)))
            execute(command("verify", root, "--analysis-dir", str(analysis)))
            study.write_json(root / "analysis-completed.json", {"path": analysis.name})
        execute(command("verify", root, "--analysis-dir", str(analysis)))
        archive = next_output(root, "issue163-evidence", ".tar.gz")
        execute(command("export", root, "--analysis-dir", str(analysis), "--archive", str(archive)))
        study.write_json(
            root / "completed.json",
            {
                "source": args.source,
                "analysis": analysis.name,
                "archive": archive.name,
                "sha256": study.sha256(archive),
                "decision": study.read_json(analysis / "result.json")["status"],
            },
        )
        print(json.dumps(study.read_json(root / "completed.json"), indent=2), flush=True)
    finally:
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        ownership.unlink()


if __name__ == "__main__":
    main()
