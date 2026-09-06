"""Repository-level contract tests for documentation and agent structure."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "agent_code" / "_team_agent_template"
UPSTREAM_TRAINING_EXAMPLES = {"tpl_agent"}
CPU_TORCH_INDEX = "--extra-index-url https://download.pytorch.org/whl/cpu"
CPU_TORCH_REQUIREMENTS = {
    'torch==2.13.0+cpu; sys_platform != "darwin"',
    'torch==2.13.0; sys_platform == "darwin"',
}


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
            "docs/0006-ai-usage.md",
            "docs/0007-task-1-baseline-contract.md",
            ".github/pull_request_template.md",
            ".github/workflows/ci.yml",
        )
        missing = [path for path in required if not (ROOT / path).is_file()]
        self.assertEqual([], missing, f"Missing repository files: {missing}")

    def test_readme_links_task_1_baseline_contract(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("docs/0007-task-1-baseline-contract.md", readme)

    def test_pull_request_template_exposes_conditional_evidence_scopes(self) -> None:
        template = (ROOT / ".github" / "pull_request_template.md").read_text(
            encoding="utf-8"
        )
        expected_scopes = (
            "Prospective experiment protocol",
            "Completed experiment or training result",
            "Frozen or released model",
            "Partial or incomplete result record",
            "Refs #...",
        )
        for scope in expected_scopes:
            self.assertIn(scope, template)

        dod = (ROOT / "docs" / "0005-definition-of-ready-and-done.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("Results are not required", dod)
        self.assertIn("checksum without", dod)
        self.assertIn("retrievable bytes is not reviewable evidence", dod)

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

    def test_pytorch_requirements_select_cpu_distributions(self) -> None:
        requirement_files = (
            ROOT / "requirements.txt",
            ROOT / "agent_code" / "DagobertDuckDQN" / "requirements.txt",
        )
        for requirement_file in requirement_files:
            lines = {
                line.strip()
                for line in requirement_file.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            }
            self.assertIn(CPU_TORCH_INDEX, lines, requirement_file)
            self.assertTrue(lines >= CPU_TORCH_REQUIREMENTS, requirement_file)


if __name__ == "__main__":
    unittest.main()
