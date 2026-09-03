"""Tests for path-aware optional CI jobs."""

from pathlib import Path

import pytest

from scripts.ci_changed_components import (
    ALL_COMPONENTS,
    DOCKER_COMPONENT,
    DQN_COMPONENTS,
    NO_COMPONENTS,
    PYTHON_AGENT_COMPONENTS,
    TABULAR_COMPONENT,
    AffectedComponents,
    classify_path,
    classify_paths,
    write_github_output,
)


@pytest.mark.parametrize(
    "path",
    [
        "README.md",
        "docs/0003-development-workflow.md",
        "experiments/2026-09-02-example/config.yaml",
        "agent_code/DagobertDuckDQN/README.md",
        "tests/test_example.py",
        ".github/workflows/package-agent.yml",
    ],
)
def test_documentation_and_quality_only_paths_skip_optional_jobs(path: str) -> None:
    assert classify_path(path) == NO_COMPONENTS


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("agent_code/DagobertDuckDQN/callbacks.py", DQN_COMPONENTS),
        (
            "agent_code/DerKleineVermoegensumverteiler/train.py",
            TABULAR_COMPONENT,
        ),
        ("Dockerfile", DOCKER_COMPONENT),
        (".dockerignore", DOCKER_COMPONENT),
        ("requirements.txt", PYTHON_AGENT_COMPONENTS),
        ("requirements-dev.txt", PYTHON_AGENT_COMPONENTS),
        ("training/run_experiment.py", ALL_COMPONENTS),
        ("scripts/package_agent.py", ALL_COMPONENTS),
        ("environment.py", ALL_COMPONENTS),
        (".github/workflows/ci.yml", ALL_COMPONENTS),
        ("new-runtime-file.py", ALL_COMPONENTS),
    ],
)
def test_runtime_paths_trigger_every_relevant_optional_job(
    path: str,
    expected: AffectedComponents,
) -> None:
    assert classify_path(path) == expected


def test_path_sets_are_combined() -> None:
    affected = classify_paths(
        [
            "docs/0004-experimentation-protocol.md",
            "agent_code/DerKleineVermoegensumverteiler/config.py",
            "Dockerfile",
        ]
    )

    assert affected == AffectedComponents(tabular=True, docker=True)


def test_github_output_contains_lower_case_booleans(tmp_path: Path) -> None:
    output = tmp_path / "github-output"

    write_github_output(
        output,
        AffectedComponents(tabular=False, dqn=True, docker=True),
    )

    assert output.read_text(encoding="utf-8") == (
        "tabular=false\n"
        "dqn=true\n"
        "docker=true\n"
    )
