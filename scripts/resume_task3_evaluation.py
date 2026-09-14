"""Resume failed evaluation through unchanged registered CLI and original deadline."""

import json
import os
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE / "training_outputs/campaign-v1"
RECOVERY = ROOT / "recovery-02"


def main():
    if "--worker" in sys.argv:
        from scripts import pilot_task3_frozen as pilot
        from scripts.task3_progress_retry import with_permission_retry

        pilot.write = with_permission_retry(pilot.write, RECOVERY / "write-retries.jsonl")
        pilot.supervise(ROOT, "evaluation", resume=True)
        sys.exit(0)
    original = json.loads((RECOVERY / "chain-state.json").read_text())
    deadline = original["deadline_unix"]
    record = {
        "scope": "operational resume; scientific source, seeds, inputs and budgets unchanged",
        "original_deadline_unix": deadline,
        "started_unix": time.time(),
        "status": "running",
        "completed": [],
        "active_pid": None,
    }

    def save():
        # This separate operational record is written once per stage, never read live.
        (RECOVERY / "resume-state.json").write_text(json.dumps(record, indent=2))

    try:
        for mode in ("evaluate", "results"):
            remaining = deadline - time.time()
            if remaining <= 0:
                raise RuntimeError("Original chain deadline exhausted")
            command = [
                sys.executable,
                "-m",
                "scripts.pilot_task3_frozen",
                mode,
                "--root",
                str(ROOT),
            ]
            if mode == "evaluate":
                command = [sys.executable, "-m", "scripts.resume_task3_evaluation", "--worker"]
            else:
                command += ["--output", str(ROOT / "issue173-pilot-evidence.tar.gz")]
            record["stage"] = mode
            with (RECOVERY / (mode + ".log")).open("x") as log:
                child = subprocess.Popen(
                    command,
                    cwd=BASE,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env={
                        **os.environ,
                        "TASK173_PILOT_AUTHORIZED": "yes",
                        "OMP_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                    },
                )
                record["active_pid"] = child.pid
                save()
                try:
                    # Existing evaluation supervisor retains cumulative CPU/wall/RAM limits.
                    # Export has a conservative five-minute ceiling inside the original reserve.
                    status = child.wait(
                        timeout=min(remaining, 300) if mode == "results" else remaining
                    )
                except subprocess.TimeoutExpired:
                    import psutil

                    process = psutil.Process(child.pid)
                    for owned in process.children(recursive=True):
                        with suppress(psutil.NoSuchProcess):
                            owned.kill()
                    process.kill()
                    child.wait()
                    raise RuntimeError("Original allocation or export reserve exhausted") from None
                if status:
                    raise RuntimeError(mode + " failed; inspect retained recovery log")
            record["completed"].append(mode)
            record["active_pid"] = None
            save()
        import hashlib
        import tarfile

        output = ROOT / "issue173-recovery.tar.gz"
        with tarfile.open(output, "x:gz") as archive:
            for folder in (ROOT / "recovery-01", ROOT / "recovery-02"):
                for path in folder.rglob("*"):
                    if path.is_file():
                        archive.add(path, arcname=path.relative_to(ROOT), recursive=False)
            for path in (
                Path(__file__),
                BASE / "scripts/task3_progress_retry.py",
                BASE / "training_outputs/resume_evaluation_01.py",
            ):
                archive.add(path, arcname="tools/" + path.name, recursive=False)
        with output.open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        (ROOT / "issue173-recovery.manifest.json").write_text(
            json.dumps({"sha256": digest, "size_bytes": output.stat().st_size}, indent=2)
        )
        record["status"] = "completed"
    except BaseException as error:
        record.update(status="failed", error=str(error))
        raise
    finally:
        record["active_pid"] = None
        record["finished_unix"] = time.time()
        save()


if __name__ == "__main__":
    main()
