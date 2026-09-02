"""Tests for episode-level experiment metric normalization."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from agents import _normalize_learning_metrics
from environment import GenericWorld
from training.metrics import (
    normalize_episode_rows,
    normalize_framework_statistics,
    write_episodes_csv,
)


def make_agent_statistics(
    **overrides: object,
) -> dict[str, object]:
    """Return valid framework statistics for one agent episode."""
    statistics: dict[str, object] = {
        "score": 2,
        "coins": 2,
        "episode_steps": 8,
        "survival_steps": 6,
        "invalid": 1,
        "attempted_actions": 6,
        "action_up": 1,
        "action_right": 1,
        "action_down": 1,
        "action_left": 1,
        "action_wait": 1,
        "action_bomb": 1,
        "action_unknown": 0,
        "survived": True,
        "termination_reason": "step_limit",
        "decision_time_median_ms": 0.2,
        "decision_time_p95_ms": 0.5,
        "decision_time_max_ms": 0.7,
    }
    statistics.update(overrides)
    return statistics


def make_framework_statistics(
    round_key: object = "Round 01 (2026-08-19 16-27-48)",
    *,
    agent_statistics: dict[str, object] | None = None,
) -> dict[str, object]:
    """Return a valid framework statistics document."""
    if agent_statistics is None:
        agent_statistics = make_agent_statistics()

    return {
        "by_agent": {},
        "by_round": {
            round_key: {
                "steps": agent_statistics["episode_steps"],
                "agents": {
                    "random_agent": agent_statistics,
                },
            }
        },
    }


class EpisodeMetricNormalizationTests(unittest.TestCase):
    def test_normalizes_one_agent_episode(self) -> None:
        rows = normalize_episode_rows(
            make_framework_statistics(),
            mode="evaluation",
        )

        self.assertEqual(1, len(rows))

        row = rows[0]
        self.assertEqual(1, row["schema_version"])
        self.assertEqual(1, row["round"])
        self.assertEqual("random_agent", row["agent"])
        self.assertEqual("evaluation", row["mode"])
        self.assertEqual(8, row["episode_steps"])
        self.assertEqual(6, row["survival_steps"])
        self.assertEqual(2, row["score"])
        self.assertEqual(2, row["coins_collected"])
        self.assertEqual(1, row["invalid_actions"])
        self.assertEqual(6, row["attempted_actions"])
        self.assertEqual(0, row["action_unknown"])
        self.assertAlmostEqual(1 / 6, row["invalid_action_rate"])
        self.assertTrue(row["survived"])
        self.assertEqual("step_limit", row["termination_reason"])

    def test_preserves_unknown_action_count(self) -> None:
        rows = normalize_episode_rows(
            make_framework_statistics(
                agent_statistics=make_agent_statistics(
                    attempted_actions=7,
                    action_unknown=1,
                )
            ),
            mode="evaluation",
        )

        self.assertEqual(1, rows[0]["action_unknown"])

    def test_accepts_framework_round_identifier(self) -> None:
        statistics = make_framework_statistics(
            "Round 07 (2026-08-19 16-27-48)"
        )

        rows = normalize_episode_rows(
            statistics,
            mode="evaluation",
        )

        self.assertEqual(7, rows[0]["round"])

    def test_accepts_numeric_string_round_identifier(self) -> None:
        rows = normalize_episode_rows(
            make_framework_statistics("3"),
            mode="evaluation",
        )

        self.assertEqual(3, rows[0]["round"])

    def test_sorts_rounds_numerically(self) -> None:
        first_agent = make_agent_statistics(score=1)
        second_agent = make_agent_statistics(score=2)

        statistics = {
            "by_round": {
                "Round 10 (2026-08-19 16-30-00)": {
                    "steps": 8,
                    "agents": {
                        "random_agent": second_agent,
                    },
                },
                "Round 02 (2026-08-19 16-20-00)": {
                    "steps": 8,
                    "agents": {
                        "random_agent": first_agent,
                    },
                },
            }
        }

        rows = normalize_episode_rows(
            statistics,
            mode="evaluation",
        )

        self.assertEqual([2, 10], [row["round"] for row in rows])
        self.assertEqual([1, 2], [row["score"] for row in rows])

    def test_zero_attempted_actions_produces_missing_rate(self) -> None:
        agent_statistics = make_agent_statistics(
            episode_steps=0,
            survival_steps=0,
            invalid=0,
            attempted_actions=0,
            action_up=0,
            action_right=0,
            action_down=0,
            action_left=0,
            action_wait=0,
            action_bomb=0,
        )

        rows = normalize_episode_rows(
            make_framework_statistics(
                agent_statistics=agent_statistics
            ),
            mode="evaluation",
        )

        self.assertIsNone(rows[0]["invalid_action_rate"])

    def test_survival_steps_cannot_exceed_episode_steps(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Survival steps exceed episode steps",
        ):
            normalize_episode_rows(
                make_framework_statistics(
                    agent_statistics=make_agent_statistics(
                        episode_steps=4,
                        survival_steps=5,
                    )
                ),
                mode="evaluation",
            )

    def test_missing_optional_metrics_are_allowed(self) -> None:
        rows = normalize_episode_rows(
            make_framework_statistics(),
            mode="evaluation",
        )

        row = rows[0]
        self.assertIsNone(row["shaped_reward"])
        self.assertIsNone(row["epsilon"])
        self.assertIsNone(row["q_table_size"])
        self.assertIsNone(row["mean_abs_td_error"])

    def test_optional_metrics_are_preserved(self) -> None:
        agent_statistics = make_agent_statistics(
            learning_metrics={
                "shaped_reward": -2.5,
                "epsilon": 0.25,
                "q_table_size": 42,
                "mean_abs_td_error": 0.125,
            },
        )

        rows = normalize_episode_rows(
            make_framework_statistics(
                agent_statistics=agent_statistics
            ),
            mode="training",
        )

        row = rows[0]
        self.assertEqual("training", row["mode"])
        self.assertEqual(-2.5, row["shaped_reward"])
        self.assertEqual(0.25, row["epsilon"])
        self.assertEqual(42, row["q_table_size"])
        self.assertEqual(0.125, row["mean_abs_td_error"])

    def test_learning_metrics_must_be_namespaced_object(self) -> None:
        agent_statistics = make_agent_statistics(
            learning_metrics=0.25,
        )

        with self.assertRaisesRegex(
            ValueError,
            "Learning metrics.*must be an object",
        ):
            normalize_episode_rows(
                make_framework_statistics(
                    agent_statistics=agent_statistics
                ),
                mode="training",
            )

    def test_unknown_additional_fields_are_tolerated(self) -> None:
        agent_statistics = make_agent_statistics(
            future_metric=123,
        )

        rows = normalize_episode_rows(
            make_framework_statistics(
                agent_statistics=agent_statistics
            ),
            mode="evaluation",
        )

        self.assertEqual(1, len(rows))

    def test_missing_required_field_is_rejected(self) -> None:
        agent_statistics = make_agent_statistics()
        del agent_statistics["attempted_actions"]

        with self.assertRaisesRegex(
            ValueError,
            "attempted_actions",
        ):
            normalize_episode_rows(
                make_framework_statistics(
                    agent_statistics=agent_statistics
                ),
                mode="evaluation",
            )

    def test_action_count_mismatch_is_rejected(self) -> None:
        agent_statistics = make_agent_statistics(
            attempted_actions=7,
        )

        with self.assertRaisesRegex(
            ValueError,
            "Action-count mismatch",
        ):
            normalize_episode_rows(
                make_framework_statistics(
                    agent_statistics=agent_statistics
                ),
                mode="evaluation",
            )

    def test_invalid_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Unsupported mode",
        ):
            normalize_episode_rows(
                make_framework_statistics(),
                mode="testing",
            )

    def test_invalid_round_identifier_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Invalid round number",
        ):
            normalize_episode_rows(
                make_framework_statistics("not-a-round"),
                mode="evaluation",
            )

    def test_csv_uses_empty_fields_for_missing_values(self) -> None:
        rows = normalize_episode_rows(
            make_framework_statistics(),
            mode="evaluation",
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = (
                Path(temporary_directory) / "episodes.csv"
            )
            write_episodes_csv(rows, output_path)

            with output_path.open(
                encoding="utf-8",
                newline="",
            ) as file:
                written_rows = list(csv.DictReader(file))

        self.assertEqual(1, len(written_rows))
        self.assertEqual("", written_rows[0]["epsilon"])
        self.assertEqual("", written_rows[0]["q_table_size"])
        self.assertEqual("0", written_rows[0]["action_unknown"])
        self.assertEqual(
            "evaluation",
            written_rows[0]["mode"],
        )

    def test_complete_file_normalization(self) -> None:
        framework_statistics = make_framework_statistics()

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            input_path = directory / "framework_stats.json"
            output_path = directory / "episodes.csv"

            input_path.write_text(
                json.dumps(framework_statistics),
                encoding="utf-8",
            )

            rows = normalize_framework_statistics(
                input_path=input_path,
                output_path=output_path,
                mode="evaluation",
            )

            self.assertEqual(1, len(rows))
            self.assertTrue(output_path.is_file())

            with output_path.open(
                encoding="utf-8",
                newline="",
            ) as file:
                written_rows = list(csv.DictReader(file))

        self.assertEqual("1", written_rows[0]["round"])
        self.assertEqual(
            "random_agent",
            written_rows[0]["agent"],
        )


class FrameworkStepMetricTests(unittest.TestCase):
    def test_episode_survival_and_action_steps_remain_distinct(self) -> None:
        statistics: defaultdict[str, int] = defaultdict(int)
        statistics["attempted_actions"] = 2

        def note_stat(name: str, value: int = 1) -> None:
            statistics[name] += value

        agent = SimpleNamespace(
            name="slow_agent",
            score=0,
            dead=False,
            decision_times=[],
            statistics=statistics,
            learning_metrics={},
            survival_steps=5,
            note_stat=note_stat,
        )
        world = object.__new__(GenericWorld)
        world.running = True
        world.step = 7
        world.agents = [agent]
        world.round_id = "Round 01"
        world.round_statistics = {}

        GenericWorld.end_round(world)

        recorded = world.round_statistics["Round 01"]["agents"][
            "slow_agent"
        ]
        self.assertEqual(7, recorded["episode_steps"])
        self.assertEqual(5, recorded["survival_steps"])
        self.assertEqual(2, recorded["attempted_actions"])


class AgentLearningMetricInterfaceTests(unittest.TestCase):
    def test_accepts_numeric_metrics_and_missing_values(self) -> None:
        self.assertEqual(
            {
                "epsilon": 0.25,
                "q_table_size": 42,
                "td_error": None,
            },
            _normalize_learning_metrics(
                {
                    "epsilon": 0.25,
                    "q_table_size": 42,
                    "td_error": None,
                }
            ),
        )
        self.assertEqual({}, _normalize_learning_metrics(None))

    def test_rejects_non_numeric_metric_values(self) -> None:
        with self.assertRaisesRegex(TypeError, "must be numeric"):
            _normalize_learning_metrics({"epsilon": "0.25"})

    def test_rejects_non_finite_metric_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be finite"):
            _normalize_learning_metrics({"td_error": float("inf")})


if __name__ == "__main__":
    unittest.main()
