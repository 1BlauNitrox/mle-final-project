"""Build the tournament submission directory and zip from a selected checkpoint.

The trained agent lives in ``agent_code/DagobertDuckDQNTask3`` because every
Task 4 tool pins that name. The tournament ships a differently named copy, so
packaging copies the code verbatim and installs one selected checkpoint. Code is
never edited here: a drift between the trained and the shipped agent would make
the submitted artifact unexplainable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

# Only what the framework loads when it plays with train=False. The framework
# imports callbacks.py and nothing else unless it is training, and nothing that
# callbacks.py pulls in reaches train.py, rewards.py or migration.py.
CODE_FILES = (
    "callbacks.py",
    "config.py",
    "legality.py",
    "model.py",
    "persistence.py",
    "replay.py",
    "requirements.txt",
)
CODE_DIRS = ("features",)

REPO_ROOT = Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def build_directory(source: Path, target: Path, checkpoint: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    for name in CODE_FILES:
        shutil.copy2(source / name, target / name)
    for name in CODE_DIRS:
        shutil.copytree(source / name, target / name, ignore=shutil.ignore_patterns("__pycache__"))

    shutil.copy2(checkpoint, target / "checkpoint.pt")


def write_artifact(target: Path, source: Path, checkpoint: Path, selection_basis: str) -> dict:
    artifact = {
        "schema_version": 1,
        "agent_name": target.name,
        "packaged_from": str(source.relative_to(REPO_ROOT)).replace("\\", "/"),
        "packaging_commit": git_rev(),
        "checkpoint": {
            "installed_as": "checkpoint.pt",
            "source_path": str(checkpoint),
            "sha256": sha256_file(target / "checkpoint.pt"),
            "size_bytes": (target / "checkpoint.pt").stat().st_size,
        },
        "selection_basis": selection_basis,
        "code_provenance": (
            "Code copied verbatim from the trained agent directory; packaging "
            "never edits agent code."
        ),
    }
    path = target / "artifact.json"
    path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8", newline="\n")
    return artifact


def check_matches_trained_code(target: Path, source: Path, commit: str) -> list[str]:
    """The shipped policy must run the code it was trained with.

    Training imports the agent from the runtime pinned at `commit`, not from the
    working tree, so packaging from a drifted checkout would ship an agent whose
    behaviour was never the one measured.
    """
    mismatches = []
    for name in (
        *CODE_FILES,
        *(f"{d}/{p.name}" for d in CODE_DIRS for p in (source / d).glob("*.py")),
    ):
        if name == "requirements.txt":
            continue  # declares the grader's install, not the policy
        pinned = subprocess.run(
            ["git", "show", f"{commit}:{source.relative_to(REPO_ROOT).as_posix()}/{name}"],
            cwd=REPO_ROOT,
            capture_output=True,
        )
        if pinned.returncode != 0:
            mismatches.append(f"{name}: absent from pinned runtime {commit[:12]}")
            continue
        shipped = (target / name).read_bytes()
        if hashlib.sha256(pinned.stdout).hexdigest() != hashlib.sha256(shipped).hexdigest():
            mismatches.append(f"{name}: differs from pinned runtime {commit[:12]}")
    return mismatches


def check_no_absolute_paths(target: Path) -> list[str]:
    offenders = []
    needles = (":\\", ":/", "/home/", "/Users/", "C:\\")
    for path in target.rglob("*"):
        if path.suffix not in {".py", ".txt", ".yaml", ".yml"} or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_no, line in enumerate(text.splitlines(), 1):
            if any(n in line for n in needles) and "http" not in line:
                offenders.append(f"{path.relative_to(target)}:{line_no}: {line.strip()[:90]}")
    return offenders


def smoke_run(agent_name: str, rounds: int) -> tuple[bool, str]:
    proc = subprocess.run(
        [
            sys.executable,
            "main.py",
            "play",
            "--agents",
            agent_name,
            "random_agent",
            "random_agent",
            "random_agent",
            "--n-rounds",
            str(rounds),
            "--no-gui",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    combined = (proc.stdout + proc.stderr)[-3000:]
    failed = proc.returncode != 0 or "Traceback" in combined or "Error" in combined
    return (not failed), combined


def shipped_names(target: Path) -> list[str]:
    """Exactly what goes into the zip, independent of whatever else sits in the folder.

    Playing writes logs/ and __pycache__/ into the agent directory, and a running
    milestone evaluation stages checkpoints there, so the zip is built from this
    list and never from the folder's contents.
    """
    features = sorted(f"{d}/{p.name}" for d in CODE_DIRS for p in (target / d).glob("*.py"))
    return sorted([*CODE_FILES, *features, "checkpoint.pt"])


def fresh_framework_check(out_zip: Path, agent_name: str) -> tuple[bool, str]:
    """Play one game from the zip alone, the way the graders do.

    A fresh copy of the framework gets no copy of the agent except the one in
    the zip, so a file the agent needs but the zip lacks fails here instead of
    being quietly found in our own agent_code folder.
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        archive = tmp / "framework.tar"
        subprocess.run(
            ["git", "archive", "--format=tar", f"--output={archive}", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
        )
        root = tmp / "framework"
        root.mkdir()
        with tarfile.open(archive) as tar:
            tar.extractall(root, filter="data")
        shutil.rmtree(root / "agent_code" / agent_name, ignore_errors=True)
        with zipfile.ZipFile(out_zip) as zf:
            zf.extractall(root / "agent_code")
        proc = subprocess.run(
            [
                sys.executable,
                "main.py",
                "play",
                "--agents",
                agent_name,
                "random_agent",
                "random_agent",
                "random_agent",
                "--n-rounds",
                "1",
                "--no-gui",
            ],
            cwd=root,
            capture_output=True,
            text=True,
        )
        output = (proc.stdout + proc.stderr)[-2000:]
        return proc.returncode == 0 and "Traceback" not in output, output


def build_zip(target: Path, out_zip: Path) -> None:
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    if out_zip.exists():
        out_zip.unlink()
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in shipped_names(target):
            zf.write(target / name, f"{target.name}/{name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="Bomb-omb")
    parser.add_argument("--source", default="agent_code/DagobertDuckDQNTask3")
    parser.add_argument(
        "--checkpoint",
        default="agent_code/DagobertDuckDQNTask3/checkpoint.pt",
        help="Checkpoint installed as the shipped policy.",
    )
    parser.add_argument(
        "--selection-basis",
        default="UNSELECTED provisional compatibility fixture; not a trained policy.",
    )
    parser.add_argument("--zip", default=None, help="Output zip path.")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument(
        "--trained-runtime",
        default="c4ddfa4efadf0b3ec6d4380a4239b9cb3a097113",
        help="Commit the training runtime is pinned to; shipped code must match it.",
    )
    args = parser.parse_args()

    source = (REPO_ROOT / args.source).resolve()
    target = (REPO_ROOT / "agent_code" / args.name).resolve()
    checkpoint = (REPO_ROOT / args.checkpoint).resolve()

    if not source.is_dir():
        print(f"FAIL: source {source} missing", file=sys.stderr)
        return 1
    if not checkpoint.is_file():
        print(f"FAIL: checkpoint {checkpoint} missing", file=sys.stderr)
        return 1

    print(f"source     : {source}")
    print(f"target     : {target}")
    print(f"checkpoint : {checkpoint}  ({sha256_file(checkpoint)[:16]}...)")

    build_directory(source, target, checkpoint)
    artifact = write_artifact(target, source, checkpoint, args.selection_basis)
    print(f"installed  : sha256 {artifact['checkpoint']['sha256'][:16]}...")

    drift = check_matches_trained_code(target, source, args.trained_runtime)
    if drift:
        print("FAIL: shipped code does not match the trained runtime:")
        for line in drift:
            print("   ", line)
        return 1
    print(f"check      : code matches trained runtime {args.trained_runtime[:12]}")

    offenders = check_no_absolute_paths(target)
    if offenders:
        print("FAIL: absolute paths found:")
        for line in offenders:
            print("   ", line)
        return 1
    print("check      : no absolute paths")

    if not args.skip_smoke:
        ok, output = smoke_run(args.name, args.rounds)
        if not ok:
            print("FAIL: evaluation smoke run errored:")
            print(output)
            return 1
        print(f"check      : {args.rounds} evaluation rounds vs 3 random_agents, no errors")

    out_zip = Path(args.zip) if args.zip else REPO_ROOT / "final-project-agent-code.zip"
    build_zip(target, out_zip)
    with zipfile.ZipFile(out_zip) as zf:
        names = zf.namelist()
        cb_dirs = sorted({n.rsplit("/", 1)[0] for n in names if n.endswith("callbacks.py")})
    print(f"zip        : {out_zip}  ({out_zip.stat().st_size} bytes, {len(names)} entries)")
    print(f"callbacks  : {cb_dirs}")
    if len(cb_dirs) != 1:
        print("FAIL: zip must contain exactly one directory with callbacks.py")
        return 1
    expected = sorted(f"{args.name}/{n}" for n in shipped_names(target))
    if sorted(names) != expected:
        print(
            "FAIL: zip contents differ from the runtime file list:",
            sorted(set(names) ^ set(expected)),
        )
        return 1
    if not args.skip_smoke:
        ok, output = fresh_framework_check(out_zip, args.name)
        if not ok:
            print("FAIL: the zip alone does not play in a fresh framework copy:")
            print(output)
            return 1
        print("check      : zip alone plays one game in a fresh framework copy")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
