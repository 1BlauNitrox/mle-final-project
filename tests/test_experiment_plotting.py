"""Tests for experiment plot generation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from training.metrics import write_episodes_csv
from training.plot_run import _rolling_mean, plot_run

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def make_episode_row(
    **overrides: object,
) -> dict[str, object]:
    """Return one valid normalized episode row."""
    row: dict[str, object] = {
        "schema_version": 1,
        "round": 1,
        "agent": "random_agent",
        "mode": "evaluation",
        "episode_steps": 10,
        "score": 2,
        "coins_collected": 2,
        "invalid_actions": 1,
        "attempted_actions": 10,
        "invalid_action_rate": 0.1,
        "survived": True,
        "termination_reason": "step_limit",
        "action_up": 2,
        "action_right": 2,
        "action_down": 2,
        "action_left": 2,
        "action_wait": 1,
        "action_bomb": 1,
        "decision_time_median_ms": 0.2,
        "decision_time_p95_ms": 0.5,
        "decision_time_max_ms": 0.8,
        "shaped_reward": None,
        "epsilon": None,
        "q_table_size": None,
        "mean_abs_td_error": None,
    }
    row.update(overrides)
    return row


def create_run_directory(
    parent: Path,
    rows: list[dict[str, object]],
) -> Path:
    """Create a temporary run directory containing episodes.csv."""
    run_directory = parent / "test-run"
    run_directory.mkdir()

    write_episodes_csv(
        rows,
        run_directory / "episodes.csv",
    )

    return run_directory


class RollingMeanTests(unittest.TestCase):
    def test_rolling_mean_uses_shorter_initial_windows(self) -> None:
        result = _rolling_mean(
            [2, 4, 9],
            window=2,
        )

        self.assertEqual(
            [2.0, 3.0, 6.5],
            result,
        )

    def test_rolling_mean_handles_one_value(self) -> None:
        result = _rolling_mean(
            [5],
            window=10,
        )

        self.assertEqual([5.0], result)

    def test_rolling_mean_rejects_zero_window(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Rolling window must be positive",
        ):
            _rolling_mean(
                [1, 2, 3],
                window=0,
            )


class ExperimentPlottingTests(unittest.TestCase):
    def test_one_episode_creates_all_png_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_directory = create_run_directory(
                root,
                [make_episode_row()],
            )

            output_paths = plot_run(
                run_directory,
                rolling_window=10,
            )

            self.assertEqual(3, len(output_paths))

            expected_names = {
                "learning_curve.png",
                "task_metrics.png",
                "behavior_diagnostics.png",
            }
            self.assertEqual(
                expected_names,
                {path.name for path in output_paths},
            )

            for output_path in output_paths:
                self.assertTrue(output_path.is_file())
                self.assertGreater(
                    output_path.stat().st_size,
                    len(PNG_SIGNATURE),
                )
                self.assertEqual(
                    PNG_SIGNATURE,
                    output_path.read_bytes()[
                        : len(PNG_SIGNATURE)
                    ],
                )

    def test_missing_optional_metrics_do_not_break_plotting(
        self,
    ) -> None:
        row = make_episode_row(
            shaped_reward=None,
            epsilon=None,
            q_table_size=None,
            mean_abs_td_error=None,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_directory = create_run_directory(
                root,
                [row],
            )

            output_paths = plot_run(run_directory)

            diagnostics_path = (
                run_directory
                / "figures"
                / "behavior_diagnostics.png"
            ).resolve()

            self.assertIn(
                diagnostics_path,
                output_paths,
            )
            self.assertTrue(diagnostics_path.is_file())
            self.assertGreater(
                diagnostics_path.stat().st_size,
                0,
            )

    def test_available_learning_metrics_do_not_break_plotting(
        self,
    ) -> None:
        rows = [
            make_episode_row(
                round=1,
                epsilon=0.8,
                q_table_size=10,
                mean_abs_td_error=0.5,
            ),
            make_episode_row(
                round=2,
                epsilon=0.6,
                q_table_size=25,
                mean_abs_td_error=0.25,
            ),
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_directory = create_run_directory(
                root,
                rows,
            )

            output_paths = plot_run(run_directory)

            for output_path in output_paths:
                self.assertTrue(output_path.is_file())
                self.assertGreater(
                    output_path.stat().st_size,
                    0,
                )

    def test_multiple_agents_are_supported(self) -> None:
        rows = [
            make_episode_row(
                agent="first_agent",
                score=2,
                coins_collected=2,
            ),
            make_episode_row(
                agent="second_agent",
                score=3,
                coins_collected=3,
            ),
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_directory = create_run_directory(
                root,
                rows,
            )

            output_paths = plot_run(run_directory)

            self.assertEqual(3, len(output_paths))
            self.assertTrue(
                all(path.is_file() for path in output_paths)
            )

    def test_missing_invalid_action_rate_is_supported(self) -> None:
        row = make_episode_row(
            episode_steps=0,
            invalid_actions=0,
            attempted_actions=0,
            invalid_action_rate=None,
            action_up=0,
            action_right=0,
            action_down=0,
            action_left=0,
            action_wait=0,
            action_bomb=0,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_directory = create_run_directory(
                root,
                [row],
            )

            output_paths = plot_run(run_directory)

            self.assertEqual(3, len(output_paths))
            self.assertTrue(
                all(path.is_file() for path in output_paths)
            )

    def test_replotting_preserves_source_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_directory = create_run_directory(
                root,
                [make_episode_row()],
            )
            episodes_path = run_directory / "episodes.csv"
            original_episodes = episodes_path.read_bytes()

            plot_run(run_directory)
            first_output_paths = plot_run(run_directory)

            self.assertEqual(
                original_episodes,
                episodes_path.read_bytes(),
            )
            self.assertEqual(
                3,
                len(first_output_paths),
            )
            self.assertTrue(
                all(
                    path.is_file()
                    for path in first_output_paths
                )
            )

    def test_zero_rolling_window_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_directory = create_run_directory(
                root,
                [make_episode_row()],
            )

            with self.assertRaisesRegex(
                ValueError,
                "Rolling window must be positive",
            ):
                plot_run(
                    run_directory,
                    rolling_window=0,
                )

            self.assertFalse(
                (run_directory / "figures").exists()
            )

    def test_missing_episodes_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_directory = (
                Path(temporary_directory) / "missing-run"
            )
            run_directory.mkdir()

            with self.assertRaisesRegex(
                FileNotFoundError,
                "Episode metrics do not exist",
            ):
                plot_run(run_directory)


if __name__ == "__main__":
    unittest.main()