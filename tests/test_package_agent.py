"""Tests for the submission packager."""

from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.package_agent import package_agent, validate_agent


class PackageAgentTests(unittest.TestCase):
    def test_rejects_template_directory(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        with self.assertRaisesRegex(ValueError, "Template/private"):
            validate_agent("_team_agent_template", repository_root)

    def test_packages_only_selected_agent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            agent = root / "agent_code" / "learned_agent"
            agent.mkdir(parents=True)
            for name in ("callbacks.py", "train.py", "README.md", "model.npz"):
                (agent / name).write_text(name, encoding="utf-8")
            (agent / "logs").mkdir()
            (agent / "logs" / "agent.log").write_text("ignore", encoding="utf-8")

            output = root / "dist" / "submission.zip"
            package_agent("learned_agent", root, output)

            with zipfile.ZipFile(output) as archive:
                names = set(archive.namelist())

            self.assertIn("learned_agent/callbacks.py", names)
            self.assertIn("learned_agent/model.npz", names)
            self.assertNotIn("learned_agent/logs/agent.log", names)


if __name__ == "__main__":
    unittest.main()
