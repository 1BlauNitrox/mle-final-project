"""Repository-level contract tests for documentation and agent structure."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "agent_code" / "_team_agent_template"
UPSTREAM_TRAINING_EXAMPLES = {"tpl_agent"}


class RepositoryContractTests(unittest.TestCase):
    def test_required_project_files_exist(self) -> None:
        required = (
            "AGENTS.md",
            "README.md",
            "CONTRIBUTING.md",
            "requirements.txt",
            "requirements-dev.txt",
            "docs/0001-project-requirements.md",
            "docs/0002-repository-architecture.md",
            "docs/0003-development-workflow.md",
            "docs/0004-experimentation-protocol.md",
            "docs/0005-definition-of-ready-and-done.md",
            ".github/pull_request_template.md",
            ".github/workflows/ci.yml",
        )
        missing = [path for path in required if not (ROOT / path).is_file()]
        self.assertEqual([], missing, f"Missing repository files: {missing}")

    def test_template_exposes_framework_callbacks(self) -> None:
        expected = {
            "callbacks.py": {"setup", "act"},
            "train.py": {"setup_training", "game_events_occurred", "end_of_round"},
        }
        for filename, required_functions in expected.items():
            source = (TEMPLATE / filename).read_text(encoding="utf-8")
            tree = ast.parse(source, filename=filename)
            functions = {
                node.name for node in tree.body if isinstance(node, ast.FunctionDef)
            }
            self.assertTrue(
                required_functions <= functions,
                f"{filename} is missing {required_functions - functions}",
            )

    def test_every_team_agent_has_an_agent_card(self) -> None:
        team_agents = [
            path
            for path in (ROOT / "agent_code").iterdir()
            if path.is_dir()
            and (path / "train.py").is_file()
            and path.name not in UPSTREAM_TRAINING_EXAMPLES
        ]
        missing = [path.name for path in team_agents if not (path / "README.md").is_file()]
        self.assertEqual([], missing, f"Agents without README.md: {missing}")

    def test_report_pdf_is_ignored(self) -> None:
        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("final-report*.pdf", gitignore)


if __name__ == "__main__":
    unittest.main()
