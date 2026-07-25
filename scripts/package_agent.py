"""Validate and package one self-contained agent directory for submission."""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

REQUIRED_FILES = ("callbacks.py", "train.py", "README.md")
EXCLUDED_PARTS = {"__pycache__", "logs"}
AGENT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def validate_agent(agent_name: str, repository_root: Path) -> Path:
    """Return a validated agent path or raise a descriptive error."""
    if agent_name.startswith("_"):
        raise ValueError("Template/private directories cannot be packaged")
    if not AGENT_NAME_PATTERN.fullmatch(agent_name):
        raise ValueError("Agent name must contain only letters, digits, underscores, or hyphens")

    agent_root = repository_root / "agent_code" / agent_name
    if not agent_root.is_dir():
        raise FileNotFoundError(f"Agent directory does not exist: {agent_root}")

    missing = [name for name in REQUIRED_FILES if not (agent_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Agent is missing required files: {', '.join(missing)}")

    return agent_root


def package_agent(agent_name: str, repository_root: Path, output_path: Path) -> Path:
    """Create a deterministic-enough zip containing only the selected agent."""
    agent_root = validate_agent(agent_name, repository_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(path for path in agent_root.rglob("*") if path.is_file())
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            relative = path.relative_to(agent_root)
            if EXCLUDED_PARTS.intersection(relative.parts) or path.suffix == ".log":
                continue
            archive.write(path, Path(agent_name) / relative)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("agent_name", help="Directory name below agent_code/")
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[1]
    output_path = repository_root / "dist" / "final-project-agent-code.zip"
    result = package_agent(args.agent_name, repository_root, output_path)
    print(f"Created {result}")


if __name__ == "__main__":
    main()
