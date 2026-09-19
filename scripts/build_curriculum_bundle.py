"""Build or verify the portable issue217 source/runtime/input archive."""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.curriculum_io import read, require, sha, write  # noqa: E402
from scripts.run_hunting_curriculum import CONFIG, sources  # noqa: E402


def verify(path, destination=None):
    with zipfile.ZipFile(path) as stream:
        manifest = json.loads(stream.read("manifest.json"))
        require(
            set(stream.namelist()) == set(manifest["files"]) | {"manifest.json"},
            "Unexpected archive members",
        )
        require(len(stream.namelist()) == len(set(stream.namelist())), "Duplicate archive member")
        for name, record in manifest["files"].items():
            require(
                not Path(name).is_absolute() and ".." not in Path(name).parts,
                "Unsafe archive member",
            )
            content = stream.read(name)
            require(
                len(content) == record["bytes"]
                and hashlib.sha256(content).hexdigest() == record["sha256"],
                f"Archive verification failed: {name}",
            )
        if destination:
            require(not destination.exists(), "Extraction destination must be new")
            destination.mkdir(parents=True)
            stream.extractall(destination)
    return manifest


def build(reference, output):
    cfg = read(CONFIG)
    require(sha(reference) == cfg["reference_sha256"], "Wrong reference checkpoint")
    require(not output.exists(), "Never overwrite a published bundle")
    names = list(sources()) + [
        "scripts/build_curriculum_bundle.py",
        "scripts/start_hunting_curriculum.ps1",
        "training/issue217-commands.md",
        "requirements.txt",
        "requirements-dev.txt",
    ]
    dirty = subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=ROOT, text=True)
    tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    require(
        not set(dirty.splitlines()).intersection(names) and set(names) <= set(tracked),
        "Commit technical sources before bundling",
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        runtime = Path(temporary) / "runtime.tar"
        subprocess.run(
            ["git", "archive", "--format=tar", f"--output={runtime}", cfg["runtime_commit"]],
            cwd=ROOT,
            check=True,
        )
        metadata = {
            "runtime_commit": cfg["runtime_commit"],
            "sha256": sha(runtime),
            "bytes": runtime.stat().st_size,
        }
        content = {name: (ROOT / name).read_bytes() for name in names}
        content["inputs/reference.pt"] = reference.read_bytes()
        content["inputs/runtime.tar"] = runtime.read_bytes()
        content["inputs/runtime.json"] = json.dumps(metadata, indent=2).encode()
        manifest = {
            "source_commit": commit,
            "issue": 217,
            "files": {
                name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
                for name, data in content.items()
            },
        }
        with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED) as stream:
            for name, data in content.items():
                stream.writestr(name, data)
            stream.writestr("manifest.json", json.dumps(manifest, indent=2))
    verify(output)
    write(
        output.with_suffix(".json"),
        {"sha256": sha(output), "bytes": output.stat().st_size, "source_commit": commit},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--extract", type=Path)
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify(args.verify, args.extract), indent=2))
    else:
        require(args.reference is not None and args.output is not None, "Reference/output required")
        build(args.reference, args.output)


if __name__ == "__main__":
    main()
