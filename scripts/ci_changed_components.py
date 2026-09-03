"""Classify changed files for the optional CI smoke-test jobs.

The required quality and framework smoke jobs always run. This module only
decides whether the more expensive tabular-agent, DQN-agent, and official
Docker compatibility jobs are relevant to a change.
"""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class AffectedComponents:
    """Optional CI components affected by a set of changed paths."""

    tabular: bool = False
    dqn: bool = False
    docker: bool = False

    def union(self, other: AffectedComponents) -> AffectedComponents:
        """Return the component-wise union of two classifications."""
        return AffectedComponents(
            tabular=self.tabular or other.tabular,
            dqn=self.dqn or other.dqn,
            docker=self.docker or other.docker,
        )


ALL_COMPONENTS = AffectedComponents(tabular=True, dqn=True, docker=True)
NO_COMPONENTS = AffectedComponents()
TABULAR_COMPONENT = AffectedComponents(tabular=True)
DQN_COMPONENTS = AffectedComponents(dqn=True, docker=True)
DOCKER_COMPONENT = AffectedComponents(docker=True)
PYTHON_AGENT_COMPONENTS = AffectedComponents(tabular=True, dqn=True)

DOCUMENTATION_ROOTS = {
    "docs",
    "experiments",
}
ROOT_DOCUMENTATION_FILES = {
    "AGENTS.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "README.md",
}
FRAMEWORK_RUNTIME_FILES = {
    "agents.py",
    "environment.py",
    "events.py",
    "items.py",
    "main.py",
    "settings.py",
    "world.py",
}


def classify_path(raw_path: str) -> AffectedComponents:
    """Return the optional CI jobs that can be affected by one path.

    The fallback is deliberately conservative: an unfamiliar non-documentation
    path runs every optional check. It is safer to spend runner time than to
    miss a submission compatibility regression.
    """
    normalized = raw_path.replace("\\", "/").removeprefix("./")
    path = PurePosixPath(normalized)
    parts = path.parts

    if not parts:
        return NO_COMPONENTS

    if normalized == ".github/workflows/ci.yml":
        return ALL_COMPONENTS

    if (
        parts[0] in DOCUMENTATION_ROOTS
        or normalized in ROOT_DOCUMENTATION_FILES
        or path.suffix.lower() == ".md"
    ):
        return NO_COMPONENTS

    if parts[0] == "tests":
        # The required quality job executes the complete test suite.
        return NO_COMPONENTS

    if parts[0] == "agent_code" and len(parts) >= 2:
        if parts[1] == "DagobertDuckDQN":
            return DQN_COMPONENTS
        if parts[1] == "DerKleineVermoegensumverteiler":
            return TABULAR_COMPONENT
        if parts[1] == "_team_agent_template":
            return NO_COMPONENTS

    if parts[0] == "training":
        return ALL_COMPONENTS

    if normalized == "scripts/package_agent.py":
        return ALL_COMPONENTS

    if normalized in FRAMEWORK_RUNTIME_FILES:
        return ALL_COMPONENTS

    if normalized in {"Dockerfile", ".dockerignore"}:
        return DOCKER_COMPONENT

    if normalized in {"requirements.txt", "requirements-dev.txt"}:
        return PYTHON_AGENT_COMPONENTS

    if normalized == ".github/workflows/package-agent.yml":
        # This workflow is exercised manually and does not change CI runtime.
        return NO_COMPONENTS

    return ALL_COMPONENTS


def classify_paths(paths: Iterable[str]) -> AffectedComponents:
    """Combine classifications for all changed paths."""
    affected = NO_COMPONENTS
    for path in paths:
        affected = affected.union(classify_path(path))
    return affected


def changed_paths(revision: str) -> list[str]:
    """Read NUL-delimited changed paths for one Git revision range."""
    result = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            "--diff-filter=ACDMRTUXB",
            "-z",
            revision,
        ],
        check=True,
        capture_output=True,
    )
    return [
        item.decode("utf-8", errors="surrogateescape")
        for item in result.stdout.split(b"\0")
        if item
    ]


def write_github_output(path: Path, affected: AffectedComponents) -> None:
    """Publish lower-case booleans for GitHub Actions job conditions."""
    with path.open("a", encoding="utf-8", newline="\n") as output:
        for name in ("tabular", "dqn", "docker"):
            value = "true" if getattr(affected, name) else "false"
            output.write(f"{name}={value}\n")


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--revision",
        required=True,
        help="Git revision range to inspect, for example BASE...HEAD",
    )
    parser.add_argument(
        "--github-output",
        type=Path,
        required=True,
        help="Path supplied by the GitHub Actions GITHUB_OUTPUT variable",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    paths = changed_paths(arguments.revision)
    affected = classify_paths(paths)
    write_github_output(arguments.github_output, affected)
    print(
        "Changed paths: "
        f"{len(paths)}; optional jobs: "
        f"tabular={affected.tabular}, dqn={affected.dqn}, "
        f"docker={affected.docker}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
