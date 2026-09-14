"""Retrieve only the checksum-bound initial/reference inputs for stability pilots."""

from __future__ import annotations

import argparse
import hashlib
import tarfile
import urllib.request
from pathlib import Path

ARCHIVE_SHA = "791257ca81e075e2e8b394f71c2e295d0ba06c35ab7fb61624a6da482131cedd"
INPUTS = {
    "initial.pt": "8076ea7ddcf2c934a9a9ef21d3dc87833ededf860cf8fd8bf486d1e222c2ccf2",
    "reference.pt": "99144d1688f66dcc6369d3efc02b2b9d5755cc8d21e6f2c1e1ffef466d0a7113",
}
URL = (
    "https://github.com/1BlauNitrox/mle-final-project/releases/download/"
    "issue175-evidence-v1/issue175-pilot-evidence.tar.gz"
)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def extract_inputs(archive, output):
    if digest(archive) != ARCHIVE_SHA:
        raise ValueError("Evidence archive checksum mismatch")
    if output.exists():
        if all((output / k).is_file() and digest(output / k) == v for k, v in INPUTS.items()):
            return
        raise ValueError("Existing inputs differ; preserve and inspect them")
    contents = {}
    with tarfile.open(archive) as bundle:
        for name, expected in INPUTS.items():
            member = bundle.getmember(name)
            if not member.isfile() or member.size > 100000:
                raise ValueError("Unexpected input member")
            contents[name] = bundle.extractfile(member).read()
            if hashlib.sha256(contents[name]).hexdigest() != expected:
                raise ValueError("Input checkpoint checksum mismatch")
    output.mkdir(parents=True)
    for name, content in contents.items():
        (output / name).write_bytes(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if not args.archive.exists() and args.download:
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        partial = args.archive.with_suffix(args.archive.suffix + ".partial")
        with partial.open("xb") as target, urllib.request.urlopen(URL, timeout=60) as response:
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
        if digest(partial) != ARCHIVE_SHA:
            raise ValueError("Downloaded checksum mismatch; partial retained")
        partial.rename(args.archive)
    extract_inputs(args.archive, args.output)
    print("Verified initial.pt and reference.pt:", args.output.resolve())


if __name__ == "__main__":
    main()
