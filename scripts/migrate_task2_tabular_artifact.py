"""Copy and verify the frozen Task 1 artifact for the tabular Task 2 successor."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

SOURCE_MODEL = REPOSITORY_ROOT / "agent_code" / "DerKleineVermoegensumverteiler" / "model.npz"

TARGET_MODEL = REPOSITORY_ROOT / "agent_code" / "DerKleineSprengstoffkapitalist" / "model.npz"

EXPECTED_SHA256 = "4e1da63a819ef8f51b112ffaf422ab251b853915375fe486538be8595b988307"

EXPECTED_SIZE_BYTES = 6845


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 checksum of a file."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def verify_artifact(path: Path) -> None:
    """Verify the expected frozen Task 1 artifact."""

    if not path.is_file():
        raise FileNotFoundError(f"Artifact does not exist: {path}")

    actual_size = path.stat().st_size
    actual_sha256 = sha256_file(path)

    if actual_size != EXPECTED_SIZE_BYTES:
        raise RuntimeError(
            f"Unexpected artifact size for {path}: "
            f"expected {EXPECTED_SIZE_BYTES}, got {actual_size}"
        )

    if actual_sha256 != EXPECTED_SHA256:
        raise RuntimeError(
            f"Unexpected artifact checksum for {path}: "
            f"expected {EXPECTED_SHA256}, got {actual_sha256}"
        )


def main() -> None:
    """Verify the parent and create or verify the successor artifact."""

    verify_artifact(SOURCE_MODEL)

    if TARGET_MODEL.exists():
        verify_artifact(TARGET_MODEL)
        print(f"Successor artifact already exists and is valid: {TARGET_MODEL}")
        return

    TARGET_MODEL.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE_MODEL, TARGET_MODEL)
    verify_artifact(TARGET_MODEL)

    if SOURCE_MODEL.read_bytes() != TARGET_MODEL.read_bytes():
        raise RuntimeError("Copied artifact is not byte-identical to its parent")

    print(f"Copied and verified artifact: {TARGET_MODEL}")
    print(f"SHA-256: {EXPECTED_SHA256}")
    print(f"Size: {EXPECTED_SIZE_BYTES} bytes")


if __name__ == "__main__":
    main()
