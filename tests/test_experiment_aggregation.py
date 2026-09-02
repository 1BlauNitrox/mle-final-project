"""Tests for experiment metric aggregation."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from training.aggregate import (
    aggregate_episode_rows,
    aggregate_episodes_csv,
    read_episodes_csv,
)
from training.metrics import write_episodes_csv


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
        "survival_steps": 10,
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
        "action_unknown": 0,
        "decision_time_median_ms": 0.2,
        "decision_time_p95_ms": 0.5,
        "decision_time_max_ms": 0.8,
        "shaped_reward": None,
        "epsilon": None,
        "q_table_size": None,
        "replay_size": None,
        "update_count": None,
        "mean_loss": None,
        "mean_abs_td_error": None,
        "target_synchronizations": None,
        "episode_target_synchronizations": None,
    }
    row.update(overrides)
    return row


class ExperimentAggregationTests(unittest.TestCase):
    def assert_csv_row_rejected(
        self,
        message: str,
        **overrides: object,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "episodes.csv"
            write_episodes_csv([make_episode_row(**overrides)], path)

            with self.assertRaisesRegex(ValueError, message):
                read_episodes_csv(path)

    def test_aggregates_multiple_episodes(self) -> None:
        rows = [
            make_episode_row(
                round=1,
                score=2,
                coins_collected=2,
                episode_steps=10,
            ),
            make_episode_row(
                round=2,
                score=4,
                coins_collected=4,
                episode_steps=20,
            ),
        ]

        summary = aggregate_episode_rows(rows)
        overall = summary["overall"]

        self.assertEqual(1, summary["schema_version"])
        self.assertEqual(2, summary["episode_rows"])
        self.assertEqual(["random_agent"], summary["agents"])
        self.assertEqual(2, overall["episode_count"])
        self.assertEqual(6, overall["coins"]["total"])
        self.assertEqual(3.0, overall["coins"]["mean"])
        self.assertEqual(6, overall["score"]["total"])
        self.assertEqual(3.0, overall["score"]["mean"])
        self.assertEqual(30, overall["episode_steps"]["total"])
        self.assertEqual(15.0, overall["episode_steps"]["mean"])

    def test_invalid_action_rate_uses_aggregate_counts(self) -> None:
        first = make_episode_row(
            invalid_actions=1,
            attempted_actions=2,
            invalid_action_rate=0.5,
            action_up=1,
            action_right=1,
            action_down=0,
            action_left=0,
            action_wait=0,
            action_bomb=0,
        )
        second = make_episode_row(
            round=2,
            invalid_actions=1,
            attempted_actions=8,
            invalid_action_rate=0.125,
            action_up=2,
            action_right=2,
            action_down=1,
            action_left=1,
            action_wait=1,
            action_bomb=1,
        )

        summary = aggregate_episode_rows([first, second])
        invalid = summary["overall"]["invalid_actions"]

        self.assertEqual(2, invalid["total"])
        self.assertEqual(10, invalid["attempted_actions"])
        self.assertAlmostEqual(0.2, invalid["rate"])

    def test_zero_attempted_actions_produces_missing_rate(self) -> None:
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

        summary = aggregate_episode_rows([row])
        overall = summary["overall"]

        self.assertIsNone(
            overall["invalid_actions"]["rate"]
        )

        for value in overall["actions"]["distribution"].values():
            self.assertIsNone(value)

    def test_aggregates_action_counts_and_distribution(self) -> None:
        row = make_episode_row(
            action_bomb=0,
            action_unknown=1,
        )

        summary = aggregate_episode_rows([row])
        actions = summary["overall"]["actions"]

        self.assertEqual(2, actions["totals"]["action_up"])
        self.assertEqual(1, actions["totals"]["action_wait"])
        self.assertEqual(1, actions["totals"]["action_unknown"])
        self.assertAlmostEqual(
            0.2,
            actions["distribution"]["action_up"],
        )
        self.assertAlmostEqual(
            0.1,
            actions["distribution"]["action_wait"],
        )
        self.assertAlmostEqual(
            1.0,
            sum(actions["distribution"].values()),
        )

    def test_aggregates_survival_and_termination_reasons(self) -> None:
        rows = [
            make_episode_row(
                round=1,
                survived=True,
                termination_reason="step_limit",
            ),
            make_episode_row(
                round=2,
                survived=False,
                termination_reason="killed",
            ),
        ]

        summary = aggregate_episode_rows(rows)
        survival = summary["overall"]["survival"]

        self.assertEqual(1, survival["survived_episodes"])
        self.assertEqual(0.5, survival["rate"])
        self.assertEqual(
            {
                "killed": 1,
                "step_limit": 1,
            },
            survival["termination_counts"],
        )

    def test_aggregates_agents_separately(self) -> None:
        rows = [
            make_episode_row(
                agent="first_agent",
                score=2,
            ),
            make_episode_row(
                agent="second_agent",
                score=5,
            ),
        ]

        summary = aggregate_episode_rows(
            rows,
            observed_agent="first_agent",
        )

        self.assertEqual(
            ["first_agent", "second_agent"],
            summary["agents"],
        )
        self.assertEqual(
            2.0,
            summary["by_agent"]["first_agent"]["score"]["mean"],
        )
        self.assertEqual(
            5.0,
            summary["by_agent"]["second_agent"]["score"]["mean"],
        )
        self.assertEqual("first_agent", summary["observed_agent"])
        self.assertEqual(2.0, summary["overall"]["score"]["mean"])

    def test_multiple_agents_require_an_observed_agent(self) -> None:
        rows = [
            make_episode_row(agent="first_agent"),
            make_episode_row(agent="second_agent"),
        ]

        with self.assertRaisesRegex(ValueError, "Observed agent is required"):
            aggregate_episode_rows(rows)

    def test_aggregates_decision_times(self) -> None:
        rows = [
            make_episode_row(
                decision_time_median_ms=0.2,
                decision_time_p95_ms=0.5,
                decision_time_max_ms=0.8,
            ),
            make_episode_row(
                round=2,
                decision_time_median_ms=0.4,
                decision_time_p95_ms=0.7,
                decision_time_max_ms=1.2,
            ),
        ]

        summary = aggregate_episode_rows(rows)
        decision_time = summary["overall"]["decision_time_ms"]

        self.assertAlmostEqual(
            0.3,
            decision_time["mean_episode_median"],
        )
        self.assertAlmostEqual(
            0.6,
            decision_time["mean_episode_p95"],
        )
        self.assertEqual(1.2, decision_time["maximum"])

    def test_missing_optional_metrics_are_allowed(self) -> None:
        row = make_episode_row(
            decision_time_median_ms=None,
            decision_time_p95_ms=None,
            decision_time_max_ms=None,
            shaped_reward=None,
            epsilon=None,
            q_table_size=None,
            mean_abs_td_error=None,
        )

        summary = aggregate_episode_rows([row])
        overall = summary["overall"]

        self.assertIsNone(
            overall["decision_time_ms"]["mean_episode_median"]
        )
        self.assertIsNone(
            overall["decision_time_ms"]["mean_episode_p95"]
        )
        self.assertIsNone(
            overall["decision_time_ms"]["maximum"]
        )
        self.assertIsNone(
            overall["learning_metrics"]["mean_shaped_reward"]
        )
        self.assertIsNone(
            overall["learning_metrics"]["mean_epsilon"]
        )
        self.assertIsNone(
            overall["learning_metrics"]["maximum_q_table_size"]
        )
        self.assertIsNone(
            overall["learning_metrics"]["mean_abs_td_error"]
        )
        # Zero is a valid measurement and must not stand in for missing data.
        self.assertIsNone(
            overall["learning_metrics"]["total_episode_target_synchronizations"]
        )

    def test_aggregates_available_learning_metrics(self) -> None:
        rows = [
            make_episode_row(
                shaped_reward=2.0,
                epsilon=0.5,
                q_table_size=10,
                replay_size=100,
                update_count=20,
                mean_loss=0.8,
                mean_abs_td_error=0.4,
                target_synchronizations=1,
                episode_target_synchronizations=1,
            ),
            make_episode_row(
                round=2,
                shaped_reward=4.0,
                epsilon=0.3,
                q_table_size=25,
                replay_size=200,
                update_count=40,
                mean_loss=0.4,
                mean_abs_td_error=0.2,
                target_synchronizations=2,
                episode_target_synchronizations=1,
            ),
        ]

        summary = aggregate_episode_rows(rows)
        learning = summary["overall"]["learning_metrics"]

        self.assertEqual(3.0, learning["mean_shaped_reward"])
        self.assertEqual(0.4, learning["mean_epsilon"])
        self.assertEqual(25, learning["maximum_q_table_size"])
        self.assertEqual(200, learning["maximum_replay_size"])
        self.assertEqual(40, learning["maximum_update_count"])
        self.assertAlmostEqual(0.6, learning["mean_loss"])
        self.assertAlmostEqual(
            0.3,
            learning["mean_abs_td_error"],
        )
        self.assertEqual(2, learning["maximum_target_synchronizations"])
        self.assertEqual(
            2,
            learning["total_episode_target_synchronizations"],
        )

    def test_coin_efficiency_uses_ratios_of_totals(self) -> None:
        rows = [
            make_episode_row(
                episode_steps=20,
                survival_steps=20,
                coins_collected=0,
            ),
            make_episode_row(
                round=2,
                episode_steps=20,
                survival_steps=20,
                coins_collected=4,
            ),
        ]

        summary = aggregate_episode_rows(rows)
        overall = summary["overall"]

        # The zero-coin episode contributes its 20 steps to the cost,
        # but it does not introduce a fictitious collected coin.
        self.assertEqual(
            10.0,
            overall["steps_per_coin"]["ratio_of_totals"],
        )
        self.assertEqual(
            10.0,
            overall["coins_per_100_steps"]["ratio_of_totals"],
        )
        self.assertEqual(
            1,
            overall["coins"]["zero_coin_episodes"],
        )
        self.assertEqual(0.5, overall["coins"]["zero_coin_rate"])

    def test_all_zero_coin_episodes_have_no_steps_per_coin(self) -> None:
        summary = aggregate_episode_rows(
            [make_episode_row(coins_collected=0)]
        )
        overall = summary["overall"]

        self.assertIsNone(
            overall["steps_per_coin"]["ratio_of_totals"]
        )
        self.assertEqual(
            0.0,
            overall["coins_per_100_steps"]["ratio_of_totals"],
        )
        self.assertEqual(1.0, overall["coins"]["zero_coin_rate"])

    def test_empty_episode_collection_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "empty episode collection",
        ):
            aggregate_episode_rows([])

    def test_reads_and_aggregates_csv_file(self) -> None:
        rows = [
            make_episode_row(
                action_bomb=0,
                action_unknown=1,
            ),
            make_episode_row(
                round=2,
                score=4,
                coins_collected=4,
            ),
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            input_path = directory / "episodes.csv"
            output_path = directory / "summary.json"

            write_episodes_csv(rows, input_path)

            parsed_rows = read_episodes_csv(input_path)
            summary = aggregate_episodes_csv(
                input_path=input_path,
                output_path=output_path,
            )

            self.assertEqual(2, len(parsed_rows))
            self.assertEqual(1, parsed_rows[0]["action_unknown"])
            self.assertTrue(output_path.is_file())

        self.assertEqual(2, summary["episode_rows"])
        self.assertEqual(
            6,
            summary["overall"]["coins"]["total"],
        )

    def test_empty_csv_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "episodes.csv"
            write_episodes_csv([], path)

            with self.assertRaisesRegex(
                ValueError,
                "does not contain any episode rows",
            ):
                read_episodes_csv(path)

    def test_missing_required_csv_column_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "episodes.csv"

            with path.open(
                "w",
                encoding="utf-8",
                newline="",
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["schema_version", "round"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "schema_version": 1,
                        "round": 1,
                    }
                )

            with self.assertRaisesRegex(
                ValueError,
                "missing required columns",
            ):
                read_episodes_csv(path)

    def test_invalid_numeric_csv_value_is_rejected(self) -> None:
        rows = [make_episode_row()]

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "episodes.csv"
            write_episodes_csv(rows, path)

            contents = path.read_text(encoding="utf-8")
            contents = contents.replace(
                ",10,2,2,",
                ",not-a-number,2,2,",
                1,
            )
            path.write_text(contents, encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                "invalid integer",
            ):
                read_episodes_csv(path)

    def test_survival_steps_cannot_exceed_episode_steps_in_csv(self) -> None:
        rows = [make_episode_row(episode_steps=5, survival_steps=6)]

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "episodes.csv"
            write_episodes_csv(rows, path)

            with self.assertRaisesRegex(
                ValueError,
                "more survival steps than episode steps",
            ):
                read_episodes_csv(path)

    def test_negative_csv_counters_are_rejected(self) -> None:
        columns = (
            "episode_steps",
            "survival_steps",
            "coins_collected",
            "invalid_actions",
            "attempted_actions",
            "action_up",
            "action_unknown",
            "q_table_size",
        )

        for column in columns:
            with self.subTest(column=column):
                self.assert_csv_row_rejected(
                    "negative",
                    **{column: -1},
                )

    def test_invalid_actions_cannot_exceed_attempts_in_csv(self) -> None:
        self.assert_csv_row_rejected(
            "more invalid actions than attempted actions",
            invalid_actions=11,
        )

    def test_action_counts_must_match_attempts_in_csv(self) -> None:
        self.assert_csv_row_rejected(
            "action counts totaling",
            action_unknown=1,
        )

    def test_non_finite_csv_numbers_are_rejected(self) -> None:
        cases = (
            ("decision_time_max_ms", float("inf")),
            ("epsilon", float("-inf")),
            ("mean_abs_td_error", float("nan")),
        )

        for column, value in cases:
            with self.subTest(column=column, value=value):
                self.assert_csv_row_rejected(
                    "non-finite number",
                    **{column: value},
                )


if __name__ == "__main__":
    unittest.main()
