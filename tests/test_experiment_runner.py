"""Tests for the reproducible experiment runner."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace
from unittest.mock import patch

import training.run_experiment as runner
from training.plot_run import plot_run


class ExperimentCommandTests(unittest.TestCase):
    def test_evaluation_command_does_not_enable_training(self) -> None:
        command = runner._build_game_command(
            agent="random_agent",
            mode="evaluation",
            scenario="coin-heaven",
            rounds=5,
            world_seed=1,
            opponents=[],
            framework_statistics_path=Path(
                "/tmp/example output/framework_stats.json"
            ),
        )

        self.assertNotIn("--train", command)
        self.assertIn("--seed", command)
        self.assertIn("1", command)
        self.assertIn(
            str(Path("/tmp/example output/framework_stats.json")),
            command,
        )

    def test_training_command_trains_only_first_agent(self) -> None:
        command = runner._build_game_command(
            agent="_team_agent_template",
            mode="training",
            scenario="coin-heaven",
            rounds=5,
            world_seed=2,
            opponents=["random_agent"],
            framework_statistics_path=Path(
                "/tmp/framework_stats.json"
            ),
        )

        agents_index = command.index("--agents")
        self.assertEqual(
            "_team_agent_template",
            command[agents_index + 1],
        )
        self.assertEqual(
            "random_agent",
            command[agents_index + 2],
        )

        train_index = command.index("--train")
        self.assertEqual("1", command[train_index + 1])

    def test_command_omits_missing_world_seed(self) -> None:
        command = runner._build_game_command(
            agent="random_agent",
            mode="evaluation",
            scenario="coin-heaven",
            rounds=1,
            world_seed=None,
            opponents=[],
            framework_statistics_path=Path(
                "/tmp/framework_stats.json"
            ),
        )

        self.assertNotIn("--seed", command)

    def test_duplicate_agent_uses_framework_instance_name(self) -> None:
        self.assertEqual(
            "random_agent_0",
            runner._framework_agent_name(
                "random_agent",
                ["random_agent"],
            ),
        )


class AgentConfigurationReferenceTests(unittest.TestCase):
    def test_fingerprint_changes_with_agent_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            agent_directory = root / "agent_code" / "test_agent"
            agent_directory.mkdir(parents=True)
            configuration = agent_directory / "config.py"
            configuration.write_text("EPSILON = 0.5\n", encoding="utf-8")
            snapshot_directory = root / "run" / "agent_snapshot"

            with patch.object(runner, "REPOSITORY_ROOT", root):
                first = runner._agent_configuration_reference(
                    "test_agent",
                    snapshot_directory=snapshot_directory,
                )
                configuration.write_text(
                    "EPSILON = 0.2\n",
                    encoding="utf-8",
                )
                second = runner._agent_configuration_reference("test_agent")
                snapshot_content = (
                    snapshot_directory / "config.py"
                ).read_text(encoding="utf-8")

        self.assertEqual("agent_code/test_agent", first["path"])
        self.assertEqual(64, len(first["sha256"] or ""))
        self.assertEqual("agent_snapshot", first["snapshot_path"])
        self.assertEqual(
            "EPSILON = 0.5\n",
            snapshot_content,
        )
        self.assertNotEqual(first["sha256"], second["sha256"])


class ExperimentRunnerTests(unittest.TestCase):
    def test_training_agent_metrics_reach_framework_csv_and_summary(
        self,
    ) -> None:
        agent_root = runner.REPOSITORY_ROOT / "agent_code"

        with (
            tempfile.TemporaryDirectory(
                dir=agent_root,
                prefix="test_learning_metrics_",
            ) as agent_directory_name,
            tempfile.TemporaryDirectory() as output_directory_name,
        ):
            agent_directory = Path(agent_directory_name)
            agent_name = agent_directory.name

            (agent_directory / "callbacks.py").write_text(
                dedent(
                    """
                    def setup(self):
                        pass


                    def act(self, game_state):
                        return "BOMB" if game_state["step"] == 1 else "WAIT"
                    """
                ).lstrip(),
                encoding="utf-8",
            )
            (agent_directory / "train.py").write_text(
                dedent(
                    """
                    def setup_training(self):
                        pass


                    def game_events_occurred(
                        self,
                        old_game_state,
                        self_action,
                        new_game_state,
                        events,
                    ):
                        pass


                    def end_of_round(
                        self,
                        last_game_state,
                        last_action,
                        events,
                    ):
                        return {
                            "shaped_reward": 7.5,
                            "epsilon": 0.25,
                            "q_table_size": 12,
                            "mean_abs_td_error": 0.125,
                        }
                    """
                ).lstrip(),
                encoding="utf-8",
            )

            run_directory = runner.run_experiment(
                agent=agent_name,
                mode="training",
                scenario="coin-heaven",
                rounds=1,
                world_seed=1,
                agent_seed=2,
                opponents=[],
                output_root=Path(output_directory_name),
            )

            framework_statistics = json.loads(
                (run_directory / "framework_stats.json").read_text(
                    encoding="utf-8"
                )
            )
            with (run_directory / "episodes.csv").open(
                encoding="utf-8",
                newline="",
            ) as episodes_file:
                episode_rows = list(csv.DictReader(episodes_file))
            summary = json.loads(
                (run_directory / "summary.json").read_text(
                    encoding="utf-8"
                )
            )
            metadata = json.loads(
                (run_directory / "metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            snapshot_created = (
                run_directory / "agent_snapshot" / "callbacks.py"
            ).is_file()
            figure_paths = plot_run(run_directory)
            figures_created = all(path.is_file() for path in figure_paths)

        round_statistics = next(
            iter(framework_statistics["by_round"].values())
        )
        self.assertEqual(
            {
                "epsilon": 0.25,
                "mean_abs_td_error": 0.125,
                "q_table_size": 12,
                "shaped_reward": 7.5,
            },
            round_statistics["agents"][agent_name]["learning_metrics"],
        )
        self.assertEqual("7.5", episode_rows[0]["shaped_reward"])
        self.assertEqual("0.25", episode_rows[0]["epsilon"])
        self.assertEqual("12", episode_rows[0]["q_table_size"])
        self.assertEqual(
            7.5,
            summary["overall"]["learning_metrics"][
                "mean_shaped_reward"
            ],
        )
        self.assertEqual(
            12,
            summary["overall"]["learning_metrics"][
                "maximum_q_table_size"
            ],
        )
        self.assertEqual(3, len(figure_paths))
        self.assertTrue(figures_created)
        self.assertEqual(
            "agent_snapshot",
            metadata["agent_configuration"]["snapshot_path"],
        )
        self.assertTrue(snapshot_created)

    def test_successful_run_writes_completed_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)

            def fake_subprocess_run(
                command: list[str],
                **_: object,
            ) -> SimpleNamespace:
                statistics_index = command.index("--save-stats") + 1
                statistics_path = Path(command[statistics_index])
                statistics_path.write_text(
                    '{"by_round": {}}',
                    encoding="utf-8",
                )
                return SimpleNamespace(returncode=0)

            with (
                patch.object(
                    runner,
                    "_git_commit",
                    return_value="a" * 40,
                ),
                patch.object(
                    runner,
                    "_git_is_dirty",
                    return_value=False,
                ),
                patch.object(
                    runner.subprocess,
                    "run",
                    side_effect=fake_subprocess_run,
                ),
                patch.object(
                    runner,
                    "normalize_framework_statistics",
                    return_value=[{"round": 1}],
                ) as normalize,
                patch.object(
                    runner,
                    "aggregate_episodes_csv",
                    return_value={"episode_rows": 1},
                ) as aggregate,
            ):
                run_directory = runner.run_experiment(
                    agent="random_agent",
                    mode="evaluation",
                    scenario="coin-heaven",
                    rounds=1,
                    world_seed=1,
                    agent_seed=None,
                    opponents=[],
                    output_root=output_root,
                )

            metadata = json.loads(
                (run_directory / "metadata.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual("completed", metadata["status"])
        self.assertEqual(0, metadata["return_code"])
        self.assertIsNone(metadata["error"])
        self.assertEqual("a" * 40, metadata["git_commit"])
        self.assertFalse(metadata["git_dirty"])
        self.assertEqual("random_agent", metadata["observed_agent"])
        self.assertEqual(
            "agent_code/random_agent",
            metadata["agent_configuration"]["path"],
        )
        self.assertEqual(
            64,
            len(metadata["agent_configuration"]["sha256"]),
        )
        self.assertIsNone(
            metadata["agent_configuration"]["snapshot_path"]
        )
        self.assertIsInstance(
            metadata["duration_seconds"],
            float,
        )
        normalize.assert_called_once()
        aggregate.assert_called_once_with(
            input_path=run_directory / "episodes.csv",
            output_path=run_directory / "summary.json",
            observed_agent="random_agent",
        )

    def test_process_failure_writes_failed_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)

            with (
                patch.object(
                    runner,
                    "_git_commit",
                    return_value="b" * 40,
                ),
                patch.object(
                    runner,
                    "_git_is_dirty",
                    return_value=True,
                ),
                patch.object(
                    runner.subprocess,
                    "run",
                    return_value=SimpleNamespace(returncode=7),
                ),self.assertRaisesRegex(
                RuntimeError,
                "return code 7",
            )
            ):
                runner.run_experiment(
                    agent="does_not_exist",
                    mode="evaluation",
                    scenario="coin-heaven",
                    rounds=1,
                    world_seed=1,
                    agent_seed=None,
                    opponents=[],
                    output_root=output_root,
                )

            run_directories = list(output_root.iterdir())
            self.assertEqual(1, len(run_directories))

            run_directory = run_directories[0]
            metadata = json.loads(
                (run_directory / "metadata.json").read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual("failed", metadata["status"])
            self.assertEqual(7, metadata["return_code"])
            self.assertIn("RuntimeError", metadata["error"])
            self.assertGreaterEqual(
                metadata["duration_seconds"],
                0,
            )
            self.assertFalse(
                (run_directory / "episodes.csv").exists()
            )
            self.assertFalse(
                (run_directory / "summary.json").exists()
            )

    def test_missing_framework_statistics_marks_run_failed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)

            with (
                patch.object(
                    runner,
                    "_git_commit",
                    return_value="c" * 40,
                ),
                patch.object(
                    runner,
                    "_git_is_dirty",
                    return_value=False,
                ),
                patch.object(
                    runner.subprocess,
                    "run",
                    return_value=SimpleNamespace(returncode=0),
                ),self.assertRaisesRegex(
                FileNotFoundError,
                "without producing framework statistics",
            )
            ):
                runner.run_experiment(
                    agent="random_agent",
                    mode="evaluation",
                    scenario="coin-heaven",
                    rounds=1,
                    world_seed=1,
                    agent_seed=None,
                    opponents=[],
                    output_root=output_root,
                )

            run_directory = next(output_root.iterdir())
            metadata = json.loads(
                (run_directory / "metadata.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual("failed", metadata["status"])
        self.assertEqual(0, metadata["return_code"])
        self.assertIn("FileNotFoundError", metadata["error"])

    def test_postprocessing_failure_marks_run_failed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)

            def fake_subprocess_run(
                command: list[str],
                **_: object,
            ) -> SimpleNamespace:
                statistics_index = command.index("--save-stats") + 1
                statistics_path = Path(command[statistics_index])
                statistics_path.write_text(
                    '{"by_round": {}}',
                    encoding="utf-8",
                )
                return SimpleNamespace(returncode=0)

            with (
                patch.object(
                    runner,
                    "_git_commit",
                    return_value="d" * 40,
                ),
                patch.object(
                    runner,
                    "_git_is_dirty",
                    return_value=False,
                ),
                patch.object(
                    runner.subprocess,
                    "run",
                    side_effect=fake_subprocess_run,
                ),
                patch.object(
                    runner,
                    "normalize_framework_statistics",
                    side_effect=ValueError("invalid statistics"),
                ),self.assertRaisesRegex(
                ValueError,
                "invalid statistics",
            )
            ):
                runner.run_experiment(
                    agent="random_agent",
                    mode="evaluation",
                    scenario="coin-heaven",
                    rounds=1,
                    world_seed=1,
                    agent_seed=None,
                    opponents=[],
                    output_root=output_root,
                )

            run_directory = next(output_root.iterdir())
            metadata = json.loads(
                (run_directory / "metadata.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual("failed", metadata["status"])
        self.assertEqual(0, metadata["return_code"])
        self.assertEqual(
            "ValueError: invalid statistics",
            metadata["error"],
        )

    def test_agent_seed_is_exposed_through_environment(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            captured_environment: dict[str, str] = {}

            def fake_subprocess_run(
                command: list[str],
                **kwargs: object,
            ) -> SimpleNamespace:
                environment = kwargs["env"]
                assert isinstance(environment, dict)
                captured_environment.update(environment)

                statistics_index = command.index("--save-stats") + 1
                statistics_path = Path(command[statistics_index])
                statistics_path.write_text(
                    '{"by_round": {}}',
                    encoding="utf-8",
                )
                return SimpleNamespace(returncode=0)

            with (
                patch.object(
                    runner,
                    "_git_commit",
                    return_value="e" * 40,
                ),
                patch.object(
                    runner,
                    "_git_is_dirty",
                    return_value=False,
                ),
                patch.object(
                    runner.subprocess,
                    "run",
                    side_effect=fake_subprocess_run,
                ),
                patch.object(
                    runner,
                    "normalize_framework_statistics",
                    return_value=[{"round": 1}],
                ),
                patch.object(
                    runner,
                    "aggregate_episodes_csv",
                    return_value={"episode_rows": 1},
                ),
            ):
                runner.run_experiment(
                    agent="random_agent",
                    mode="evaluation",
                    scenario="coin-heaven",
                    rounds=1,
                    world_seed=1,
                    agent_seed=42,
                    opponents=[],
                    output_root=output_root,
                )

        self.assertEqual(
            "42",
            captured_environment["BOMBERMAN_AGENT_SEED"],
        )

    def test_rejects_more_than_four_agents(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(
                ValueError,
                "at most four agents",
            ):
                runner.run_experiment(
                    agent="first_agent",
                    mode="evaluation",
                    scenario="classic",
                    rounds=1,
                    world_seed=1,
                    agent_seed=None,
                    opponents=[
                        "second_agent",
                        "third_agent",
                        "fourth_agent",
                        "fifth_agent",
                    ],
                    output_root=Path(temporary_directory),
                )

            self.assertEqual(
                [],
                list(Path(temporary_directory).iterdir()),
            )


if __name__ == "__main__":
    unittest.main()
