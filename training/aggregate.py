from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from statistics import fmean
from typing import Any

from training.metrics import SCHEMA_VERSION

REQUIRED_COLUMNS = {
    "schema_version",
    "round",
    "agent",
    "mode",
    "episode_steps",
    "score",
    "coins_collected",
    "invalid_actions",
    "attempted_actions",
    "invalid_action_rate",
    "survived",
    "termination_reason",
    "action_up",
    "action_right",
    "action_down",
    "action_left",
    "action_wait",
    "action_bomb",
    "decision_time_median_ms",
    "decision_time_p95_ms",
    "decision_time_max_ms",
}

ACTION_COLUMNS = (
    "action_up",
    "action_right",
    "action_down",
    "action_left",
    "action_wait",
    "action_bomb",
)

INTEGER_COLUMNS = (
    "schema_version",
    "round",
    "episode_steps",
    "score",
    "coins_collected",
    "invalid_actions",
    "attempted_actions",
    *ACTION_COLUMNS,
)

OPTIONAL_FLOAT_COLUMNS = (
    "invalid_action_rate",
    "decision_time_median_ms",
    "decision_time_p95_ms",
    "decision_time_max_ms",
    "shaped_reward",
    "epsilon",
    "mean_abs_td_error",
)

OPTIONAL_INTEGER_COLUMNS = (
    "q_table_size",
)


def read_episodes_csv(path: Path) -> list[dict[str, Any]]:
    """Read and validate normalized episode rows from CSV."""
    if not path.is_file():
        raise FileNotFoundError(f"Episode metrics do not exist: {path}")

    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError("Episode CSV does not contain a header")

        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(
                f"Episode CSV is missing required columns: {missing}"
            )

        rows = [
            _parse_csv_row(raw_row, row_number)
            for row_number, raw_row in enumerate(reader, start=2)
        ]

    if not rows:
        raise ValueError("Episode CSV does not contain any episode rows")

    return rows


def aggregate_episode_rows(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate episode rows overall and separately for each agent."""
    materialized_rows = [dict(row) for row in rows]

    if not materialized_rows:
        raise ValueError("Cannot aggregate an empty episode collection")

    rows_by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in materialized_rows:
        agent = row.get("agent")
        if not isinstance(agent, str) or not agent:
            raise ValueError(
                f"Episode row contains an invalid agent name: {agent!r}"
            )
        rows_by_agent[agent].append(row)

    return {
        "schema_version": SCHEMA_VERSION,
        "episode_rows": len(materialized_rows),
        "agents": sorted(rows_by_agent),
        "overall": _aggregate_group(materialized_rows),
        "by_agent": {
            agent: _aggregate_group(agent_rows)
            for agent, agent_rows in sorted(rows_by_agent.items())
        },
    }


def write_summary_json(
    summary: Mapping[str, Any],
    output_path: Path,
) -> None:
    """Write an aggregated experiment summary as formatted JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")


def aggregate_episodes_csv(
    input_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Read episodes.csv, aggregate it, and write summary.json."""
    rows = read_episodes_csv(input_path)
    summary = aggregate_episode_rows(rows)
    write_summary_json(summary, output_path)
    return summary


def _aggregate_group(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate one collection of agent-episode rows."""
    episode_count = len(rows)

    total_coins = sum(row["coins_collected"] for row in rows)
    total_invalid_actions = sum(
        row["invalid_actions"] for row in rows
    )
    total_attempted_actions = sum(
        row["attempted_actions"] for row in rows
    )
    survived_episodes = sum(
        1 for row in rows if row["survived"]
    )

    invalid_action_rate = (
        total_invalid_actions / total_attempted_actions
        if total_attempted_actions > 0
        else None
    )

    action_totals = {
        action: sum(row[action] for row in rows)
        for action in ACTION_COLUMNS
    }

    action_distribution = {
        action: (
            count / total_attempted_actions
            if total_attempted_actions > 0
            else None
        )
        for action, count in action_totals.items()
    }

    termination_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        termination_counts[row["termination_reason"]] += 1

    steps_per_coin = [
        row["episode_steps"] / max(row["coins_collected"], 1)
        for row in rows
    ]

    return {
        "episode_count": episode_count,
        "coins": {
            "total": total_coins,
            "mean": fmean(
                row["coins_collected"] for row in rows
            ),
            "zero_coin_episodes": sum(
                1 for row in rows
                if row["coins_collected"] == 0
            ),
        },
        "score": {
            "total": sum(row["score"] for row in rows),
            "mean": fmean(row["score"] for row in rows),
        },
        "episode_steps": {
            "total": sum(row["episode_steps"] for row in rows),
            "mean": fmean(
                row["episode_steps"] for row in rows
            ),
        },
        "steps_per_coin": {
            "mean": fmean(steps_per_coin),
        },
        "invalid_actions": {
            "total": total_invalid_actions,
            "attempted_actions": total_attempted_actions,
            "rate": invalid_action_rate,
        },
        "survival": {
            "survived_episodes": survived_episodes,
            "rate": survived_episodes / episode_count,
            "termination_counts": dict(
                sorted(termination_counts.items())
            ),
        },
        "actions": {
            "totals": action_totals,
            "distribution": action_distribution,
        },
        "decision_time_ms": {
            "mean_episode_median": _mean_available(
                rows,
                "decision_time_median_ms",
            ),
            "mean_episode_p95": _mean_available(
                rows,
                "decision_time_p95_ms",
            ),
            "maximum": _maximum_available(
                rows,
                "decision_time_max_ms",
            ),
        },
        "learning_metrics": {
            "mean_shaped_reward": _mean_available(
                rows,
                "shaped_reward",
            ),
            "mean_epsilon": _mean_available(
                rows,
                "epsilon",
            ),
            "maximum_q_table_size": _maximum_available(
                rows,
                "q_table_size",
            ),
            "mean_abs_td_error": _mean_available(
                rows,
                "mean_abs_td_error",
            ),
        },
    }


def _parse_csv_row(
    raw_row: Mapping[str, str | None],
    row_number: int,
) -> dict[str, Any]:
    """Parse one CSV row into typed values."""
    parsed: dict[str, Any] = dict(raw_row)

    for column in INTEGER_COLUMNS:
        parsed[column] = _parse_integer(
            raw_row.get(column),
            column=column,
            row_number=row_number,
        )

    for column in OPTIONAL_INTEGER_COLUMNS:
        parsed[column] = _parse_optional_integer(
            raw_row.get(column),
            column=column,
            row_number=row_number,
        )

    for column in OPTIONAL_FLOAT_COLUMNS:
        parsed[column] = _parse_optional_float(
            raw_row.get(column),
            column=column,
            row_number=row_number,
        )

    parsed["survived"] = _parse_boolean(
        raw_row.get("survived"),
        column="survived",
        row_number=row_number,
    )

    for column in ("agent", "mode", "termination_reason"):
        value = raw_row.get(column)
        if value is None or not value.strip():
            raise ValueError(
                f"CSV row {row_number} contains an empty "
                f"{column!r} field"
            )
        parsed[column] = value

    if parsed["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"CSV row {row_number} uses unsupported schema version "
            f"{parsed['schema_version']!r}"
        )

    return parsed


def _parse_integer(
    value: str | None,
    *,
    column: str,
    row_number: int,
) -> int:
    """Parse one required integer CSV field."""
    if value is None or value == "":
        raise ValueError(
            f"CSV row {row_number} contains an empty "
            f"{column!r} field"
        )

    try:
        return int(value)
    except ValueError as error:
        raise ValueError(
            f"CSV row {row_number} contains invalid integer "
            f"{column!r}: {value!r}"
        ) from error


def _parse_optional_integer(
    value: str | None,
    *,
    column: str,
    row_number: int,
) -> int | None:
    """Parse one optional integer CSV field."""
    if value is None or value == "":
        return None

    return _parse_integer(
        value,
        column=column,
        row_number=row_number,
    )


def _parse_optional_float(
    value: str | None,
    *,
    column: str,
    row_number: int,
) -> float | None:
    """Parse one optional floating-point CSV field."""
    if value is None or value == "":
        return None

    try:
        return float(value)
    except ValueError as error:
        raise ValueError(
            f"CSV row {row_number} contains invalid number "
            f"{column!r}: {value!r}"
        ) from error


def _parse_boolean(
    value: str | None,
    *,
    column: str,
    row_number: int,
) -> bool:
    """Parse a CSV boolean written by csv.DictWriter."""
    if value == "True":
        return True
    if value == "False":
        return False

    raise ValueError(
        f"CSV row {row_number} contains invalid boolean "
        f"{column!r}: {value!r}"
    )


def _mean_available(
    rows: Iterable[Mapping[str, Any]],
    field: str,
) -> float | None:
    """Return the mean of available values or null if all are missing."""
    values = [
        row[field]
        for row in rows
        if row.get(field) is not None
    ]
    return fmean(values) if values else None


def _maximum_available(
    rows: Iterable[Mapping[str, Any]],
    field: str,
) -> float | int | None:
    """Return the maximum available value or null if all are missing."""
    values = [
        row[field]
        for row in rows
        if row.get(field) is not None
    ]
    return max(values) if values else None