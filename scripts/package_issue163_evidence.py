"""Export or verify compact #163 evidence with exact executed source bytes."""

import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

SOURCE = "1e18b9c659041b4044d04e9ec6563b98c8c471dc"


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def record(path):
    return {"sha256": sha(path), "size_bytes": path.stat().st_size}


def verify(archive_path, extract=None):
    manifest = json.loads(Path(str(archive_path) + ".manifest.json").read_text(encoding="utf-8"))
    if record(archive_path) != {k: manifest[k] for k in ("sha256", "size_bytes")}:
        raise ValueError("Archive hash/size mismatch")
    if extract is not None:
        extract.mkdir(parents=True, exist_ok=False)
    seen = set()
    with tarfile.open(archive_path) as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if (
                not member.isfile()
                or name.is_absolute()
                or ".." in name.parts
                or "\\" in member.name
                or ":" in member.name
                or member.name in seen
            ):
                raise ValueError("Unsafe/duplicate archive member")
            seen.add(member.name)
            expected = manifest["files"].get(member.name)
            if expected is None:
                raise ValueError("Unlisted archive member")
            with archive.extractfile(member) as source:
                data = source.read()
            if (
                hashlib.sha256(data).hexdigest() != expected["sha256"]
                or len(data) != expected["size_bytes"]
            ):
                raise ValueError("Member bytes differ")
            if extract is not None:
                path = extract / member.name
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as file:
                    file.write(data)
    if seen != set(manifest["files"]):
        raise ValueError("Missing archive member")
    return {
        "verified_files": len(seen),
        "sha256": manifest["sha256"],
        "size_bytes": manifest["size_bytes"],
    }


def export(root, source, analysis, archive_path):
    if archive_path.exists() or Path(str(archive_path) + ".manifest.json").exists():
        raise FileExistsError("Preserve previous export")
    if (root / ".supervisor.lock").exists() or (root / "runs/.task3-campaign.lock").exists():
        raise ValueError("Campaign still active")
    if (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != SOURCE
    ):
        raise ValueError("Wrong source checkout")
    names = {
        "metadata.json",
        "framework_stats.json",
        "episodes.csv",
        "status.json",
        "resolved_plan.json",
        "authorization.json",
        "preflight.json",
        "resources.json",
        "resources-amended166.json",
    }
    files = {}
    for path in (root / "runs").rglob("*"):
        if path.is_file() and (
            path.name in names
            or (path.suffix == ".pt" and {"artifacts", "replicas"} & set(path.parts))
            or any(part.endswith(".failed-storage") for part in path.parts)
        ):
            files["campaign-root/" + path.relative_to(root).as_posix()] = path
    for folder in ("binding", "recovery-166-001"):
        for path in (root / folder).rglob("*"):
            if path.is_file():
                files["campaign-root/" + path.relative_to(root).as_posix()] = path
    for path in root.iterdir():
        if path.is_file() and path.suffix in {".json", ".py", ".txt", ".log"}:
            files["campaign-root/" + path.name] = path
    for path in analysis.iterdir():
        if path.is_file():
            files["analysis/" + path.name] = path
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=source).decode().split("\0")
    source_records = {}
    for name in filter(None, tracked):
        path = source / name
        if path.is_symlink():
            raise ValueError("Source symlink unsupported")
        files["source/" + name] = path
        source_records[name] = record(path)
    source_record = archive_path.with_name(archive_path.name + ".source-record.json")
    with source_record.open("x", encoding="utf-8") as file:
        json.dump({"execution_source": SOURCE, "files": source_records}, file, indent=2)
    files["source/source-record.json"] = source_record
    manifest = {name: record(path) for name, path in sorted(files.items())}
    with tarfile.open(archive_path, "x:gz", dereference=True) as archive:
        for name, path in sorted(files.items()):
            archive.add(path, arcname=name, recursive=False)
    with Path(str(archive_path) + ".manifest.json").open("x", encoding="utf-8") as file:
        json.dump(
            {
                **record(archive_path),
                "files": manifest,
                "scope": "completed_jobs_failed_scientific_gates_human_narrative_pending",
            },
            file,
            indent=2,
        )
    return verify(archive_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--extract", type=Path)
    args = parser.parse_args()
    if args.root:
        if args.source is None or args.analysis is None:
            parser.error("Export needs --source and --analysis")
        result = export(args.root, args.source, args.analysis, args.archive)
    else:
        result = verify(args.archive, args.extract)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
