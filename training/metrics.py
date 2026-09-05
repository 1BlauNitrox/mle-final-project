from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

VALID_MODES = {"training", "evaluation"}

CSV_COLUMNS = (
    "schema_version",
    "round",
    "agent",
    "mode",
    "episode_steps",
    "survival_steps",
    "score",
    "coins_collected",
    "initially_available_coins",
    "coins_found",
    "crates_destroyed",
    "bombs_dropped",
    "self_kills",
    "invalid_actions",
    "attempted_actions",
    "invalid_action_rate",
    "survived",
    "termination_reason",
    "executed_action_sequence_sha256",
    "action_up",
    "action_right",
    "action_down",
    "action_left",
    "action_wait",
    "action_bomb",
    "action_unknown",
    "decision_time_median_ms",
    "decision_time_p95_ms",
    "decision_time_max_ms",
    "shaped_reward",
    "epsilon",
    "q_table_size",
    "replay_size",
    "update_count",
    "mean_loss",
    "mean_abs_td_error",
    "target_synchronizations",
    "episode_target_synchronizations",
)

REQUIRED_AGENT_FIELDS = (
    "score",
    "coins",
    "episode_steps",
    "survival_steps",
    "invalid",
    "attempted_actions",
    "action_up",
    "action_right",
    "action_down",
    "action_left",
    "action_wait",
    "action_bomb",
    "survived",
    "termination_reason",
    "decision_time_median_ms",
    "decision_time_p95_ms",
    "decision_time_max_ms",
)

ACTION_FIELDS = (
    "action_up",
    "action_right",
    "action_down",
    "action_left",
    "action_wait",
    "action_bomb",
)

ROUND_KEY_PATTERN = re.compile(r"^Round\s+(?P<number>\d+)(?:\s+\([^)]*\))?$")


def load_framework_statistics(path: Path) -> dict[str, Any]:
    "Load and validate the top-level framework statistics document."
    if not path.is_file():
        raise FileNotFoundError(f"Framework statistics do not exist: {path}")

    with path.open(encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("Framework statistics must contain a JSON object")

    if "by_round" not in data:
        raise ValueError("Framework statistics are missing 'by_round'")

    if not isinstance(data["by_round"], dict):
        raise ValueError("'by_round' must contain a JSON object")

    return data


def normalize_episode_rows(
    framework_statistics: Mapping[str, Any],
    mode: str,
) -> list[dict[str, Any]]:
    "Convert nested framework statistics into one row per agent and round."
    if mode not in VALID_MODES:
        raise ValueError(
            f"Unsupported mode {mode!r}; expected one of {sorted(VALID_MODES)}"
        )

    by_round = framework_statistics.get("by_round")
    if not isinstance(by_round, Mapping):
        raise ValueError("Framework statistics are missing a valid 'by_round' object")

    rows: list[dict[str, Any]] = []

    for raw_round, round_statistics in sorted(
        by_round.items(),
        key=lambda item: _parse_round_number(item[0]),
    ):
        round_number = _parse_round_number(raw_round)

        if not isinstance(round_statistics, Mapping):
            raise ValueError(
                f"Statistics for round {round_number} must be an object"
            )

        agents = round_statistics.get("agents")
        if not isinstance(agents, Mapping):
            raise ValueError(
                f"Statistics for round {round_number} are missing 'agents'"
            )

        for agent_name, agent_statistics in sorted(agents.items()):
            if not isinstance(agent_name, str):
                raise ValueError(
                    f"Round {round_number} contains a non-string agent name"
                )

            if not isinstance(agent_statistics, Mapping):
                raise ValueError(
                    f"Statistics for agent {agent_name!r} in round "
                    f"{round_number} must be an object"
                )

            rows.append(
                _normalize_agent_episode(
                    round_number=round_number,
                    agent_name=agent_name,
                    mode=mode,
                    statistics=agent_statistics,
                )
            )

    return rows


def write_episodes_csv(
    rows: Iterable[Mapping[str, Any]],
    output_path: Path,
) -> None:
    "Write normalized episode rows using the stable version-one schema."
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    column: _csv_value(row.get(column))
                    for column in CSV_COLUMNS
                }
            )


def normalize_framework_statistics(
    input_path: Path,
    output_path: Path,
    mode: str,
) -> list[dict[str, Any]]:
    "Load framework statistics, normalize them, and write episodes.csv."
    statistics = load_framework_statistics(input_path)
    rows = normalize_episode_rows(statistics, mode)
    write_episodes_csv(rows, output_path)
    return rows


def _normalize_agent_episode(
    *,
    round_number: int,
    agent_name: str,
    mode: str,
    statistics: Mapping[str, Any],
) -> dict[str, Any]:
    "Normalize and validate one agent's statistics for one episode."
    missing = [
        field
        for field in REQUIRED_AGENT_FIELDS
        if field not in statistics
    ]
    if missing:
        raise ValueError(
            f"Agent {agent_name!r} in round {round_number} is missing "
            f"required fields: {', '.join(missing)}"
        )

    episode_steps = _non_negative_int(
        statistics["episode_steps"],
        field="episode_steps",
        round_number=round_number,
        agent_name=agent_name,
    )
    survival_steps = _non_negative_int(
        statistics["survival_steps"],
        field="survival_steps",
        round_number=round_number,
        agent_name=agent_name,
    )
    if survival_steps > episode_steps:
        raise ValueError(
            f"Survival steps exceed episode steps for agent "
            f"{agent_name!r} in round {round_number}"
        )
    attempted_actions = _non_negative_int(
        statistics["attempted_actions"],
        field="attempted_actions",
        round_number=round_number,
        agent_name=agent_name,
    )
    invalid_actions = _non_negative_int(
        statistics["invalid"],
        field="invalid",
        round_number=round_number,
        agent_name=agent_name,
    )

    action_counts = {
        field: _non_negative_int(
            statistics[field],
            field=field,
            round_number=round_number,
            agent_name=agent_name,
        )
        for field in ACTION_FIELDS
    }

    unknown_actions = _non_negative_int(
        statistics.get("action_unknown", 0),
        field="action_unknown",
        round_number=round_number,
        agent_name=agent_name,
    )

    counted_actions = sum(action_counts.values()) + unknown_actions
    if counted_actions != attempted_actions:
        raise ValueError(
            f"Action-count mismatch for agent {agent_name!r} in round "
            f"{round_number}: counted {counted_actions}, "
            f"attempted {attempted_actions}"
        )

    invalid_action_rate = (
        invalid_actions / attempted_actions
        if attempted_actions > 0
        else None
    )

    learning_metrics = statistics.get("learning_metrics", {})
    if not isinstance(learning_metrics, Mapping):
        raise ValueError(
            f"Learning metrics for agent {agent_name!r} in round "
            f"{round_number} must be an object"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "round": round_number,
        "agent": agent_name,
        "mode": mode,
        "episode_steps": episode_steps,
        "survival_steps": survival_steps,
        "score": _integer(
            statistics["score"],
            field="score",
            round_number=round_number,
            agent_name=agent_name,
        ),
        "coins_collected": _non_negative_int(
            statistics["coins"],
            field="coins",
            round_number=round_number,
            agent_name=agent_name,
        ),
        "initially_available_coins": _optional_non_negative_int(
            statistics.get("initially_available_coins"),
            field="initially_available_coins",
            round_number=round_number,
            agent_name=agent_name,
        ),
        "coins_found": _non_negative_int(
            statistics.get("coins_found", 0),
            field="coins_found",
            round_number=round_number,
            agent_name=agent_name,
        ),
        "crates_destroyed": _non_negative_int(
            statistics.get("crates_destroyed", statistics.get("crates", 0)),
            field="crates_destroyed",
            round_number=round_number,
            agent_name=agent_name,
        ),
        "bombs_dropped": _non_negative_int(
            statistics.get("bombs_dropped", statistics.get("bombs", 0)),
            field="bombs_dropped",
            round_number=round_number,
            agent_name=agent_name,
        ),
        "self_kills": _non_negative_int(
            statistics.get("self_kills", statistics.get("suicides", 0)),
            field="self_kills",
            round_number=round_number,
            agent_name=agent_name,
        ),
        "invalid_actions": invalid_actions,
        "attempted_actions": attempted_actions,
        "invalid_action_rate": invalid_action_rate,
        "survived": _boolean(
            statistics["survived"],
            field="survived",
            round_number=round_number,
            agent_name=agent_name,
        ),
        "termination_reason": _string(
            statistics["termination_reason"],
            field="termination_reason",
            round_number=round_number,
            agent_name=agent_name,
        ),
        # Optional so that statistics recorded before this field existed remain
        # readable; an empty value means the sequence was not instrumented.
        "executed_action_sequence_sha256": _optional_string(
            statistics.get("executed_action_sequence_sha256"),
            field="executed_action_sequence_sha256",
            round_number=round_number,
            agent_name=agent_name,
        ),
        **action_counts,
        "action_unknown": unknown_actions,
        "decision_time_median_ms": _optional_number(
            statistics["decision_time_median_ms"],
            field="decision_time_median_ms",
            round_number=round_number,
            agent_name=agent_name,
        ),
        "decision_time_p95_ms": _optional_number(
            statistics["decision_time_p95_ms"],
            field="decision_time_p95_ms",
            round_number=round_number,
            agent_name=agent_name,
        ),
        "decision_time_max_ms": _optional_number(
            statistics["decision_time_max_ms"],
            field="decision_time_max_ms",
            round_number=round_number,
            agent_name=agent_name,
        ),
        "shaped_reward": _optional_number(
            learning_metrics.get("shaped_reward"),
            field="shaped_reward",
            round_number=round_number,
            agent_name=agent_name,
        ),
        "epsilon": _optional_number(
            learning_metrics.get("epsilon"),
            field="epsilon",
            round_number=round_number,
            agent_name=agent_name,
        ),
        "q_table_size": _optional_non_negative_int(
            learning_metrics.get("q_table_size"),
            field="q_table_size",
            round_number=round_number,
            agent_name=agent_name,
        ),
        "replay_size": _optional_non_negative_int(
            learning_metrics.get("replay_size"),
            field="replay_size",
            round_number=round_number,
            agent_name=agent_name,
        ),
        "update_count": _optional_non_negative_int(
            learning_metrics.get("update_count"),
            field="update_count",
            round_number=round_number,
            agent_name=agent_name,
        ),
        "mean_loss": _optional_number(
            learning_metrics.get("mean_loss"),
            field="mean_loss",
            round_number=round_number,
            agent_name=agent_name,
        ),
        "mean_abs_td_error": _optional_number(
            learning_metrics.get("mean_abs_td_error"),
            field="mean_abs_td_error",
            round_number=round_number,
            agent_name=agent_name,
        ),
        "target_synchronizations": _optional_non_negative_int(
            learning_metrics.get("target_synchronizations"),
            field="target_synchronizations",
            round_number=round_number,
            agent_name=agent_name,
        ),
        "episode_target_synchronizations": _optional_non_negative_int(
            learning_metrics.get("episode_target_synchronizations"),
            field="episode_target_synchronizations",
            round_number=round_number,
            agent_name=agent_name,
        ),
    }


def _parse_round_number(value: object) -> int:
    "Convert a JSON round key into a positive integer."
    if isinstance(value, bool):
        raise ValueError(f"Invalid round number: {value!r}")

    if isinstance(value, int):
        round_number = value
    elif isinstance(value, str):
        stripped_value = value.strip()

        if stripped_value.isdigit():
            round_number = int(stripped_value)
        else:
            match = ROUND_KEY_PATTERN.fullmatch(stripped_value)
            if match is None:
                raise ValueError(f"Invalid round number: {value!r}")

            round_number = int(match.group("number"))
    else:
        raise ValueError(f"Invalid round number: {value!r}")

    if round_number < 1:
        raise ValueError(
            f"Round number must be positive: {round_number}"
        )
    return round_number



def _integer(
    value: object,
    *,
    field: str,
    round_number: int,
    agent_name: str,
) -> int:
    "Validate an integer field without accepting booleans."
    if isinstance(value, bool) or not isinstance(value, int):
        raise _field_error(
            field,
            value,
            round_number,
            agent_name,
            expected="an integer",
        )
    return value


def _non_negative_int(
    value: object,
    *,
    field: str,
    round_number: int,
    agent_name: str,
) -> int:
    "Validate a non-negative integer field."
    result = _integer(
        value,
        field=field,
        round_number=round_number,
        agent_name=agent_name,
    )
    if result < 0:
        raise _field_error(
            field,
            value,
            round_number,
            agent_name,
            expected="a non-negative integer",
        )
    return result


def _optional_non_negative_int(
    value: object,
    *,
    field: str,
    round_number: int,
    agent_name: str,
) -> int | None:
    "Validate an optional non-negative integer field."
    if value is None:
        return None

    return _non_negative_int(
        value,
        field=field,
        round_number=round_number,
        agent_name=agent_name,
    )


def _optional_number(
    value: object,
    *,
    field: str,
    round_number: int,
    agent_name: str,
) -> float | None:
    "Validate an optional numeric field."
    if value is None:
        return None

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _field_error(
            field,
            value,
            round_number,
            agent_name,
            expected="a number or null",
        )

    return float(value)


def _boolean(
    value: object,
    *,
    field: str,
    round_number: int,
    agent_name: str,
) -> bool:
    "Validate a boolean field."
    if not isinstance(value, bool):
        raise _field_error(
            field,
            value,
            round_number,
            agent_name,
            expected="a boolean",
        )
    return value


def _optional_string(
    value: object,
    *,
    field: str,
    round_number: int,
    agent_name: str,
) -> str | None:
    "Validate an optional non-empty string field."
    if value is None:
        return None

    return _string(
        value,
        field=field,
        round_number=round_number,
        agent_name=agent_name,
    )


def _string(
    value: object,
    *,
    field: str,
    round_number: int,
    agent_name: str,
) -> str:
    "Validate a non-empty string field."
    if not isinstance(value, str) or not value:
        raise _field_error(
            field,
            value,
            round_number,
            agent_name,
            expected="a non-empty string",
        )
    return value


def _field_error(
    field: str,
    value: object,
    round_number: int,
    agent_name: str,
    *,
    expected: str,
) -> ValueError:
    "Create a consistent validation error."
    return ValueError(
        f"Invalid {field!r} for agent {agent_name!r} in round "
        f"{round_number}: expected {expected}, got {value!r}"
    )


def _csv_value(value: Any) -> Any:
    "Represent unavailable optional values as empty CSV fields."
    return "" if value is None else value
