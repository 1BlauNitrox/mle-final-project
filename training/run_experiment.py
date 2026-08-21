"""Run reproducible Bomberman training or evaluation jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any

from training.aggregate import aggregate_episodes_csv
from training.metrics import SCHEMA_VERSION, normalize_framework_statistics

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "training_outputs"
VALID_MODES = ("training", "evaluation")


def run_experiment(
    *,
    agent: str,
    mode: str,
    scenario: str,
    rounds: int,
    world_seed: int | None,
    agent_seed: int | None,
    opponents: list[str],
    output_root: Path,
) -> Path:
    """Run one game job and create a self-contained experiment directory."""
    _validate_arguments(
        agent=agent,
        mode=mode,
        scenario=scenario,
        rounds=rounds,
        opponents=opponents,
    )

    started_at = datetime.now(timezone.utc)
    run_id = _create_run_id(
        started_at=started_at,
        agent=agent,
        scenario=scenario,
    )
    run_directory = output_root.resolve() / run_id
    run_directory.mkdir(parents=True, exist_ok=False)

    metadata_path = run_directory / "metadata.json"
    framework_statistics_path = (
        run_directory / "framework_stats.json"
    )
    episodes_path = run_directory / "episodes.csv"
    summary_path = run_directory / "summary.json"
    observed_agent = _framework_agent_name(agent, opponents)

    command = _build_game_command(
        agent=agent,
        mode=mode,
        scenario=scenario,
        rounds=rounds,
        world_seed=world_seed,
        opponents=opponents,
        framework_statistics_path=framework_statistics_path,
    )

    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "agent": agent,
        "observed_agent": observed_agent,
        "mode": mode,
        "scenario": scenario,
        "opponents": opponents,
        "rounds": rounds,
        "world_seed": world_seed,
        "agent_seed": agent_seed,
        "agent_configuration": _agent_configuration_reference(agent),
        "command": command,
        "git_commit": _git_commit(),
        "git_dirty": _git_is_dirty(),
        "started_at": _format_timestamp(started_at),
        "duration_seconds": None,
        "python_version": platform.python_version(),
        "status": "running",
        "return_code": None,
        "error": None,
    }
    _write_json(metadata_path, metadata)

    environment = os.environ.copy()
    if agent_seed is not None:
        environment["BOMBERMAN_AGENT_SEED"] = str(agent_seed)

    start_time = monotonic()

    try:
        result = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=False,
        )

        metadata["return_code"] = result.returncode

        if result.returncode != 0:
            raise RuntimeError(
                "Bomberman process failed with return code "
                f"{result.returncode}"
            )

        if not framework_statistics_path.is_file():
            raise FileNotFoundError(
                "Bomberman completed without producing framework "
                f"statistics: {framework_statistics_path}"
            )

        rows = normalize_framework_statistics(
            input_path=framework_statistics_path,
            output_path=episodes_path,
            mode=mode,
        )

        if not rows:
            raise ValueError(
                "Bomberman completed without producing episode rows"
            )

        aggregate_episodes_csv(
            input_path=episodes_path,
            output_path=summary_path,
            observed_agent=observed_agent,
        )

        metadata["status"] = "completed"
    except Exception as error:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        metadata["duration_seconds"] = monotonic() - start_time
        _write_json(metadata_path, metadata)

    return run_directory


def _build_game_command(
    *,
    agent: str,
    mode: str,
    scenario: str,
    rounds: int,
    world_seed: int | None,
    opponents: list[str],
    framework_statistics_path: Path,
) -> list[str]:
    """Build the Bomberman subprocess command without shell quoting."""
    command = [
        sys.executable,
        "main.py",
        "play",
        "--agents",
        agent,
        *opponents,
        "--no-gui",
        "--n-rounds",
        str(rounds),
        "--scenario",
        scenario,
        "--save-stats",
        str(framework_statistics_path),
    ]

    if mode == "training":
        command.extend(["--train", "1"])

    if world_seed is not None:
        command.extend(["--seed", str(world_seed)])

    return command


def _validate_arguments(
    *,
    agent: str,
    mode: str,
    scenario: str,
    rounds: int,
    opponents: list[str],
) -> None:
    """Validate arguments before creating the run directory."""
    if not agent or not agent.strip():
        raise ValueError("Agent name must not be empty")

    if mode not in VALID_MODES:
        raise ValueError(
            f"Unsupported mode {mode!r}; expected one of "
            f"{list(VALID_MODES)}"
        )

    if not scenario or not scenario.strip():
        raise ValueError("Scenario name must not be empty")

    if rounds < 1:
        raise ValueError("Number of rounds must be positive")

    if any(not opponent.strip() for opponent in opponents):
        raise ValueError("Opponent names must not be empty")

    total_agents = 1 + len(opponents)
    if total_agents > 4:
        raise ValueError(
            "Bomberman supports at most four agents per game"
        )


def _create_run_id(
    *,
    started_at: datetime,
    agent: str,
    scenario: str,
) -> str:
    """Create a sortable and filesystem-safe run identifier."""
    timestamp = started_at.strftime("%Y%m%dT%H%M%S%fZ")
    safe_agent = _safe_identifier(agent)
    safe_scenario = _safe_identifier(scenario)
    return f"{timestamp}-{safe_agent}-{safe_scenario}"


def _safe_identifier(value: str) -> str:
    """Replace unsafe run-identifier characters with hyphens."""
    cleaned = "".join(
        character.lower()
        if character.isalnum()
        else "-"
        for character in value
    )
    parts = [part for part in cleaned.split("-") if part]
    return "-".join(parts) or "unnamed"


def _framework_agent_name(agent: str, opponents: list[str]) -> str:
    """Return the observed agent name used in framework statistics."""
    return f"{agent}_0" if agent in opponents else agent


def _agent_configuration_reference(agent: str) -> dict[str, str | None]:
    """Fingerprint the agent directory before the run starts."""
    agent_directory = REPOSITORY_ROOT / "agent_code" / agent
    relative_directory = Path("agent_code") / agent

    if not agent_directory.is_dir():
        return {
            "path": relative_directory.as_posix(),
            "sha256": None,
        }

    digest = hashlib.sha256()
    ignored_parts = {"__pycache__", "logs"}

    for path in sorted(agent_directory.rglob("*")):
        relative_path = path.relative_to(agent_directory)
        if (
            not path.is_file()
            or ignored_parts.intersection(relative_path.parts)
            or path.suffix == ".pyc"
        ):
            continue

        digest.update(relative_path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")

    return {
        "path": relative_directory.as_posix(),
        "sha256": digest.hexdigest(),
    }


def _git_commit() -> str:
    """Return the full commit SHA of the repository."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_is_dirty() -> bool:
    """Return whether the repository has tracked or untracked changes."""
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def _format_timestamp(value: datetime) -> str:
    """Format a timezone-aware datetime as an ISO 8601 UTC timestamp."""
    return value.astimezone(timezone.utc).isoformat().replace(
        "+00:00",
        "Z",
    )


def _write_json(
    path: Path,
    data: dict[str, Any],
) -> None:
    """Write formatted JSON with a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")


def _positive_integer(value: str) -> int:
    """Parse a positive command-line integer."""
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Expected an integer, got {value!r}"
        ) from error

    if result < 1:
        raise argparse.ArgumentTypeError(
            "Value must be a positive integer"
        )

    return result


def parse_arguments(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--agent",
        required=True,
        help="Observed agent directory below agent_code/",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=VALID_MODES,
        help="Run the observed agent in training or evaluation mode",
    )
    parser.add_argument(
        "--scenario",
        default="coin-heaven",
        help="Bomberman scenario name",
    )
    parser.add_argument(
        "--rounds",
        type=_positive_integer,
        default=5,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--world-seed",
        type=int,
        help="Seed passed to the Bomberman environment",
    )
    parser.add_argument(
        "--agent-seed",
        type=int,
        help=(
            "Seed exposed to the agent as the "
            "BOMBERMAN_AGENT_SEED environment variable"
        ),
    )
    parser.add_argument(
        "--opponents",
        nargs="*",
        default=[],
        help="Optional agents placed after the observed agent",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory in which run directories are created",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the command-line experiment workflow."""
    arguments = parse_arguments(argv)

    try:
        run_directory = run_experiment(
            agent=arguments.agent,
            mode=arguments.mode,
            scenario=arguments.scenario,
            rounds=arguments.rounds,
            world_seed=arguments.world_seed,
            agent_seed=arguments.agent_seed,
            opponents=arguments.opponents,
            output_root=arguments.output_root,
        )
    except Exception as error:
        print(
            f"Experiment failed: {error}",
            file=sys.stderr,
        )
        return 1

    print(f"Experiment completed: {run_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
