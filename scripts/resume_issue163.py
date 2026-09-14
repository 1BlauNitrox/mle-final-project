"""Source-pinned #163 resume with an explicit, separately recorded wall-only amendment."""

from __future__ import annotations

import argparse
import ctypes
import gzip
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
from contextlib import suppress
from pathlib import Path

SOURCE = "1e18b9c659041b4044d04e9ec6563b98c8c471dc"
NOTE = "https://github.com/1BlauNitrox/mle-final-project/issues/163#issuecomment-5652175907"
GIB = 1024**3


def sha(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, data):
    with Path(path).open("x", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.flush()
        os.fsync(file.fileno())


def require(value, message):
    if not value:
        raise ValueError(message)


def amended_record(original, original_wall, cpu_floor=0):
    require(
        original["limits"]
        == {"cpu_seconds": 43200, "wall_seconds": original_wall, "memory_bytes": 2 * GIB},
        "Unexpected original limits",
    )
    require(
        not original["active_root_pids"] and original["limit_reached"] is None,
        "Original record has active roots or a reached limit",
    )
    result = {**original, "limits": {**original["limits"], "wall_seconds": original_wall + 7200}}
    result["cpu_seconds_consumed"] = max(original["cpu_seconds_consumed"], cpu_floor)
    require(0 <= result["cpu_seconds_consumed"] < 43200, "Retained CPU budget exhausted")
    return result


def cumulative_monitor(original):
    class CumulativeMonitor(original):
        def _sample_locked(self):
            # Retain each descendant's observed CPU after it exits; do not use the
            # maximum of a shrinking live tree as cumulative pipeline CPU.
            for record in self._active.values():
                seen = record.setdefault("seen_cpu", {})
                for process in self._tree(record["process"]):
                    try:
                        key = (process.pid, process.create_time())
                        times = process.cpu_times()
                        seen[key] = max(seen.get(key, 0), times.user + times.system)
                    except (ProcessLookupError, self.process_error):
                        continue
                record["cpu_seconds"] = sum(seen.values())

    import psutil

    CumulativeMonitor.process_error = psutil.Error
    return CumulativeMonitor


def validate(root, repo):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()

    require(
        git("rev-parse", "HEAD") == SOURCE and not git("status", "--porcelain"),
        "Keep exact clean executed source",
    )
    amendment = read(root / "wall-amendment166.json")
    require(
        amendment["source"] == SOURCE and amendment["owner_note"] == NOTE,
        "Wrong amendment identity",
    )
    require(amendment["helper_sha256"] == sha(__file__), "Resume helper changed")
    for name, digest in amendment["original_files"].items():
        require(sha(root / name) == digest, f"Original record changed: {name}")
    sys.path.insert(0, str(repo))
    from training import task3_attack_campaign as study

    require(study.ROOT.resolve() == repo, "Wrong imported execution root")
    _, plans, _ = study.validate(root / "binding")
    auth = read(root / "runs/authorization.json")
    require(auth["identity"]["reviewed_commit"] == SOURCE, "Wrong authorization source")
    require(
        auth["identity"]["plans"] == {n: study.campaign.portable_plan(p) for n, p in plans.items()},
        "Original authorized matrix changed",
    )
    require(
        auth["identity"]["binding_sha256"] == sha(root / "binding/binding.json"), "Binding changed"
    )
    for plan in plans.values():
        status = read(root / "runs/plans" / plan.plan_id / "status.json")["jobs"]
        for job in plan.jobs:
            if job.kind == "training":
                require(status[job.run_id]["status"] == "completed", "Refuse retraining")
    return study, plans, auth, amendment


def resource_decision(original, amended):
    limits = amended["limits"]
    require(
        limits == {**original["limits"], "wall_seconds": original["limits"]["wall_seconds"] + 7200},
        "Changed amendment limits",
    )
    require(
        not amended["active_root_pids"] and not amended["limit_reached"], "Amended run incomplete"
    )
    require(amended["authorized_at"] == original["authorized_at"], "Original epoch changed")
    require(amended["cpu_seconds_consumed"] >= original["cpu_seconds_consumed"], "CPU was reset")
    require(
        amended["wall_seconds_elapsed"] <= limits["wall_seconds"]
        and amended["cpu_seconds_consumed"] <= limits["cpu_seconds"]
        and amended["peak_memory_bytes"] < limits["memory_bytes"],
        "Amended cap exceeded",
    )
    return {
        "original_wall_gate_pass": amended["wall_seconds_elapsed"]
        <= original["limits"]["wall_seconds"],
        "approved_wall_gate_pass": True,
        "cpu_gate_pass": True,
        "memory_gate_pass": True,
    }


def worker(root, repo, stage):
    study, plans, auth, amendment = validate(root, repo)
    if stage == "run":
        original = study.StorageMonitor

        class AmendedStorage(original):
            def __init__(self, **kwargs):
                require(
                    Path(kwargs["state_path"]) == root / "runs/resources.json", "Wrong state path"
                )
                from training.run_issue107_campaign import CampaignLimits

                kwargs["state_path"] = root / "runs/resources-amended166.json"
                kwargs["limits"] = CampaignLimits(43200, 43200, 2 * GIB)
                super().__init__(**kwargs)

        study.StorageMonitor = AmendedStorage
        identity = auth["identity"]
        sys.argv = [
            "task3_attack_campaign",
            "run",
            "--resume",
            "--authorize-compute",
            "--binding-dir",
            str(root / "binding"),
            "--output-root",
            str(root / "runs"),
            "--reviewed-commit",
            SOURCE,
            "--authorized-by",
            identity["authorized_by"],
            "--hardware-description",
            identity["hardware_description"],
            "--available-memory-gib",
            "2",
            "--allocation-hours",
            "12",
        ]
        study.main()
        write_new(root / "evaluation-completed166.json", {"source": SOURCE})
    else:
        original = read(root / "runs/resources.json")
        amended = read(root / "runs/resources-amended166.json")
        gates = resource_decision(original, amended)
        analysis = root / "analysis-amended166"
        # Reuse the unchanged strict raw/CSV/source/checkpoint/repeat checks and
        # registered metrics. Its resource check covers the preserved INITIAL
        # segment only; separately verified cumulative amended records govern
        # completion below. Never present its conditional selection as promotion.
        study.analyze(root / "runs", root / "binding", analysis)
        (analysis / "result.json").rename(analysis / "performance-decision.json")
        performance = read(analysis / "performance-decision.json")
        final = {
            "status": "completed_with_authorized_wall_extension",
            "selected_arm": None,
            "selected_replica": None,
            "next_decision": "stop_no_automatic_continuation",
            "performance_status": performance["status"],
            "resource_gates": gates,
            "resource_scope": "original and amended cumulative records are both retained",
            "original_resources": original,
            "amended_resources": amended,
            "wall_amendment": amendment,
            "protocol_sha256": performance["protocol_sha256"],
            "observations_sha256": performance["observations_sha256"],
            "performance_decision_sha256": sha(analysis / "performance-decision.json"),
        }
        write_new(analysis / "result.json", final)
        with gzip.open(analysis / "observations.json.gz", "rt", encoding="utf-8") as file:
            rows = json.load(file)["evaluation"]
        computed = study.decide(rows, study.validate(root / "binding")[0])
        require(
            all(performance[k] == value for k, value in computed.items()),
            "Metric decision mismatch",
        )
        archive = root / "issue163-evidence-amended166.tar.gz"
        exported = study.export_files(root / "runs", root / "binding", analysis, archive)
        write_new(
            root / "analysis-export-completed166.json",
            {"archive": archive.name, **exported, "decision": final["status"]},
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--stage", choices=["run", "analyze-export"])
    args = parser.parse_args()
    root, repo = args.root.resolve(), args.repo.resolve()
    if args.prepare:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "issue166_offline_storage", Path(__file__).with_name("recover_snapshot_links.py")
        )
        storage = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(storage)
        inventory = storage.inventory

        report = inventory(root)
        require(not any(o["rotate"] for o in report["objects"]), "Storage recovery missing")
        require(not report["incomplete_snapshots"], "Incomplete input snapshot still blocks resume")
        original_files = {
            name: sha(root / name)
            for name in [
                "runs/authorization.json",
                "runs/resources.json",
                "supervisor-authorization.json",
                "supervisor-resources.json",
                "binding/binding.json",
                "hardware.txt",
            ]
        }
        inner, outer = read(root / "runs/resources.json"), read(root / "supervisor-resources.json")
        # Sum is a conservative upper bound because the original outer sampler
        # undercounted exited descendants; double-counting is deliberately safe.
        cpu_floor = inner["cpu_seconds_consumed"] + outer["cpu_seconds_consumed"]
        write_new(
            root / "wall-amendment166.json",
            {
                "source": SOURCE,
                "owner_note": NOTE,
                "helper_sha256": sha(__file__),
                "original_files": original_files,
                "wall_extension_seconds": 7200,
                "original_wall_gate_must_be_reported": True,
                "outer_cpu_initialization": "sum original inner and outer CPU (conservative)",
                "pending_evaluations_at_preparation": report["pending_evaluations"],
            },
        )
        write_new(root / "runs/resources-amended166.json", amended_record(inner, 36000))
        write_new(
            root / "supervisor-resources-amended166.json", amended_record(outer, 43200, cpu_floor)
        )
        validate(root, repo)
        print("Amendment prepared; no scientific jobs launched", flush=True)
        return
    if args.stage:
        worker(root, repo, args.stage)
        return
    study, _, _, _ = validate(root, repo)
    require(args.execute, "Execution requires explicit --execute")
    import psutil

    require(psutil.virtual_memory().available >= 2 * GIB, "Need 2 GiB available RAM")
    from scripts.run_issue163_pc import stop_owned_process
    from training.run_issue107_campaign import CampaignLimits, CampaignResourceMonitor

    lock = root / ".supervisor.lock"
    with lock.open("x", encoding="utf-8") as file:
        file.write(str(os.getpid()))
    try:
        monitor = cumulative_monitor(CampaignResourceMonitor)(
            state_path=root / "supervisor-resources-amended166.json",
            authorized_at=read(root / "supervisor-resources.json")["authorized_at"],
            limits=CampaignLimits(43200, 50400, 2 * GIB),
        )
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        for stage, marker in [
            ("run", "evaluation-completed166.json"),
            ("analyze-export", "analysis-export-completed166.json"),
        ]:
            if (root / marker).exists():
                continue
            monitor.check()
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--root",
                str(root),
                "--repo",
                str(repo),
                "--stage",
                stage,
            ]
            print("Starting", stage, flush=True)
            process = subprocess.Popen(
                cmd, cwd=repo, env={**os.environ, "BOMBERMAN_COMPACT_LOGS": "1"}
            )
            try:
                monitor.register(process.pid)
                while process.poll() is None:
                    monitor.check()
                    time.sleep(0.2)
                require(
                    process.returncode == 0,
                    f"{stage} exited {process.returncode}; preserve evidence",
                )
            finally:
                stop_owned_process(process)
                monitor.unregister(process.pid)
        monitor.check()
        # Package the amendment and storage-recovery trail separately; the compact
        # campaign export intentionally does not include arbitrary local files.
        paths = list((root / "recovery-166-001").rglob("*"))
        paths += [
            root / name
            for name in [
                "wall-amendment166.json",
                "runs/resources-amended166.json",
                "supervisor-resources-amended166.json",
                "supervisor-resources.json",
                "supervisor-authorization.json",
            ]
        ]
        paths += [Path(__file__).resolve()]
        manifest = {}
        with tarfile.open(root / "issue163-recovery-amended166.tar.gz", "x:gz") as archive:
            for path in paths:
                if not path.is_file():
                    continue
                name = (
                    path.relative_to(root).as_posix()
                    if path.is_relative_to(root)
                    else "resume_issue163.py"
                )
                monitor.check()
                archive.add(path, arcname=name, recursive=False)
                manifest[name] = {"sha256": sha(path), "size_bytes": path.stat().st_size}
        write_new(root / "issue163-recovery-amended166.manifest.json", manifest)
        monitor.check()
        write_new(
            root / "completed-amended166.json", read(root / "analysis-export-completed166.json")
        )
        print("Completed amended evaluation, analysis and evidence export", flush=True)
    finally:
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        with suppress(FileNotFoundError):
            lock.unlink()


if __name__ == "__main__":
    main()
