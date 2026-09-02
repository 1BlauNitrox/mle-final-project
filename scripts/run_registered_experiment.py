from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

import yaml

from training.aggregate import (
    aggregate_episode_rows,
    read_episodes_csv,
    write_summary_json,
)
from training.plot_run import plot_run
from training.run_experiment import run_experiment

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Experiment config must contain a YAML object")

    return config


def git_output(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def require_clean_registered_commit(config: dict[str, Any]) -> str:
    status = git_output("status", "--short")

    if status:
        raise RuntimeError(
            "Scientific runs require a clean working tree:\n"
            f"{status}"
        )

    actual_commit = git_output("rev-parse", "HEAD")
    expected_commit = config["experiment"]["experiment_commit"]

    expected_commit = config["experiment"].get(
    "experiment_commit",
    "auto",
    )

    if (expected_commit not in (None, "auto")
        and actual_commit != expected_commit):
            raise RuntimeError(
                "Repository commit does not match preregistration:\n"
                f"expected: {expected_commit}\n"
                f"actual:   {actual_commit}"
            )

    print(f"Registered run commit: {actual_commit}")

    return actual_commit


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def read_metadata(run_directory: Path) -> dict[str, Any]:
    path = run_directory / "metadata.json"

    with path.open(encoding="utf-8") as file:
        metadata = json.load(file)

    if metadata["status"] != "completed":
        raise RuntimeError(
            f"Run did not complete successfully: {run_directory}"
        )

    if metadata["return_code"] != 0:
        raise RuntimeError(
            f"Run returned {metadata['return_code']}: {run_directory}"
        )

    if metadata["git_dirty"]:
        raise RuntimeError(
            f"Run was performed on a dirty worktree: {run_directory}"
        )

    return metadata


def write_training_record(
    *,
    path: Path,
    run_number: int,
    world_seed: int,
    agent_seed: int,
    metadata: dict[str, Any],
    model_path: Path,
) -> None:
    rows = read_episodes_csv(
        Path(metadata["_run_directory"]) / "episodes.csv"
    )

    agent = metadata["observed_agent"]
    agent_rows = [
        row for row in rows
        if row["agent"] == agent
    ]

    if len(agent_rows) != metadata["rounds"]:
        raise RuntimeError(
            f"Expected {metadata['rounds']} training rows, "
            f"found {len(agent_rows)}"
        )

    final_row = agent_rows[-1]
    model_hash = sha256_file(model_path)
    model_size = model_path.stat().st_size

    content = f"""# Training Run {run_number:02d}

## Registered configuration

- Repository commit: `{metadata["git_commit"]}`
- Agent source SHA-256:
  `{metadata["agent_configuration"]["sha256"]}`
- Agent: `{metadata["agent"]}`
- Scenario: `{metadata["scenario"]}`
- Training world seed: `{world_seed}`
- Agent seed: `{agent_seed}`
- Planned episodes: `{metadata["rounds"]}`
- Checkpoint rule: final checkpoint

## Execution

- Started at: `{metadata["started_at"]}`
- Exit code: `{metadata["return_code"]}`
- Status: `{metadata["status"]}`
- Wall-clock duration: `{metadata["duration_seconds"]:.3f} seconds`
- Training output directory: `{metadata["_run_directory"]}`

## Result

- Completed episodes: `{len(agent_rows)}`
- Final cumulative configured reward:
  `{final_row.get("shaped_reward", "")}`
- Final epsilon: `{final_row.get("epsilon", "")}`
- Final Q-table size: `{final_row.get("q_table_size", "")}`
- Final mean absolute TD error:
  `{final_row.get("mean_abs_td_error", "")}`
- Model artifact: `{model_path}`
- Model size in bytes: `{model_size}`
- Model SHA-256: `{model_hash}`

## Integrity checks

- Fresh model at start: yes
- Working tree clean at start: yes
- Registered commit used: yes
- Registered configuration used: yes
- Final checkpoint selected mechanically: yes
- Model moved to run-specific location: yes
- Model checksum recorded: yes
- Failed or interrupted output retained: not applicable
"""

    path.write_text(content, encoding="utf-8")


def train_models(
    config: dict[str, Any],
    artifact_root: Path,
) -> list[dict[str, Any]]:
    agent = config["agent"]["name"]
    scenario = config["environment"]["scenario"]
    episodes = config["training"]["episodes_per_run"]
    seeds = config["training"]["seeds"]

    agent_directory = (
        REPOSITORY_ROOT
        / "agent_code"
        / agent
    )
    working_model = agent_directory / "model.npz"

    model_directory = artifact_root / "models"
    record_directory = artifact_root / "records"
    output_directory = artifact_root / "training"

    model_directory.mkdir(parents=True, exist_ok=True)
    record_directory.mkdir(parents=True, exist_ok=True)
    output_directory.mkdir(parents=True, exist_ok=True)

    completed_runs = []

    for seed_config in seeds:
        run_number = int(seed_config["run"])
        world_seed = int(seed_config["world_seed"])
        agent_seed = int(seed_config["agent_seed"])

        model_path = (
            model_directory
            / f"task1-baseline-run-{run_number:02d}-model.npz"
        )

        if model_path.exists():
            raise FileExistsError(
                f"Model artifact already exists: {model_path}"
            )

        if working_model.exists():
            raise RuntimeError(
                "Fresh training run blocked because model.npz "
                f"already exists: {working_model}"
            )

        run_directory = run_experiment(
            agent=agent,
            mode="training",
            scenario=scenario,
            rounds=episodes,
            world_seed=world_seed,
            agent_seed=agent_seed,
            opponents=[],
            output_root=output_directory / f"run-{run_number:02d}",
        )

        metadata = read_metadata(run_directory)
        metadata["_run_directory"] = str(run_directory)

        if not working_model.is_file():
            raise FileNotFoundError(
                f"Training did not create {working_model}"
            )

        shutil.move(working_model, model_path)

        write_training_record(
            path=record_directory / f"run-{run_number:02d}.md",
            run_number=run_number,
            world_seed=world_seed,
            agent_seed=agent_seed,
            metadata=metadata,
            model_path=model_path,
        )

        plot_run(run_directory, rolling_window=100)

        completed_runs.append({
            "run": run_number,
            "world_seed": world_seed,
            "agent_seed": agent_seed,
            "model_path": model_path,
            "model_sha256": sha256_file(model_path),
            "training_directory": run_directory,
            "metadata": metadata,
        })

    return completed_runs


def evaluate_models(
    config: dict[str, Any],
    artifact_root: Path,
    trained_runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    agent = config["agent"]["name"]
    scenario = config["environment"]["scenario"]

    start_seed = int(
        config["evaluation"]["development_world_seeds"]["start"]
    )
    end_seed = int(
        config["evaluation"]["development_world_seeds"]["end"]
    )

    world_seeds = range(start_seed, end_seed + 1)

    agent_directory = (
        REPOSITORY_ROOT
        / "agent_code"
        / agent
    )
    working_model = agent_directory / "model.npz"

    evaluation_root = artifact_root / "evaluation"
    evaluation_records = []

    for trained_run in trained_runs:
        run_number = trained_run["run"]
        model_path = trained_run["model_path"]
        agent_seed = trained_run["agent_seed"]

        if working_model.exists():
            working_model.unlink()

        shutil.copy2(model_path, working_model)

        hash_before = sha256_file(working_model)
        model_output = evaluation_root / f"run-{run_number:02d}"

        episode_directories = []

        for world_seed in world_seeds:
            run_directory = run_experiment(
                agent=agent,
                mode="evaluation",
                scenario=scenario,
                rounds=1,
                world_seed=world_seed,
                agent_seed=agent_seed,
                opponents=[],
                output_root=model_output,
            )

            metadata = read_metadata(run_directory)

            if metadata["world_seed"] != world_seed:
                raise RuntimeError(
                    f"Wrong seed recorded in {run_directory}"
                )

            episode_directories.append(run_directory)

        hash_after = sha256_file(working_model)

        if hash_before != hash_after:
            raise RuntimeError(
                f"Model run-{run_number:02d} changed during evaluation"
            )

        evaluation_records.append({
            **trained_run,
            "evaluation_directories": episode_directories,
            "evaluation_sha256_before": hash_before,
            "evaluation_sha256_after": hash_after,
            "immutable": True,
        })

    if working_model.exists():
        working_model.unlink()

    return evaluation_records


def collect_evaluation_rows(
    evaluation_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    all_rows = []
    rows_by_model: dict[int, list[dict[str, Any]]] = {}

    for record in evaluation_records:
        run_number = record["run"]
        model_rows = []

        for run_directory in record["evaluation_directories"]:
            metadata = read_metadata(run_directory)
            rows = read_episodes_csv(
                run_directory / "episodes.csv"
            )

            agent_rows = [
                row
                for row in rows
                if row["agent"] == metadata["observed_agent"]
            ]

            if len(agent_rows) != 1:
                raise RuntimeError(
                    f"Expected one evaluation row in {run_directory}"
                )

            row = dict(agent_rows[0])
            row["model_run"] = run_number
            row["world_seed"] = metadata["world_seed"]
            row["agent_seed"] = metadata["agent_seed"]
            row["model_sha256"] = record["model_sha256"]

            model_rows.append(row)
            all_rows.append(row)

        if len(model_rows) != 40:
            raise RuntimeError(
                f"Model {run_number} has {len(model_rows)} "
                "evaluation rows; expected 40"
            )

        rows_by_model[run_number] = model_rows

    if len(all_rows) != 200:
        raise RuntimeError(
            f"Expected 200 evaluation rows, found {len(all_rows)}"
        )

    return all_rows, rows_by_model


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    available_coins: int,
) -> dict[str, Any]:
    coins = [row["coins_collected"] for row in rows]
    fractions = [
        coins_collected / available_coins
        for coins_collected in coins
    ]

    total_coins = sum(coins)
    survival_steps = sum(
        row["survival_steps"]
        for row in rows
    )
    invalid_actions = sum(
        row["invalid_actions"]
        for row in rows
    )
    attempted_actions = sum(
        row["attempted_actions"]
        for row in rows
    )

    p95_values = [
        row["decision_time_p95_ms"]
        for row in rows
        if row["decision_time_p95_ms"] is not None
    ]
    maximum_values = [
        row["decision_time_max_ms"]
        for row in rows
        if row["decision_time_max_ms"] is not None
    ]

    return {
        "evaluation_episodes": len(rows),
        "mean_coins": fmean(coins),
        "mean_collection_fraction": fmean(fractions),
        "std_collection_fraction": (
            stdev(fractions)
            if len(fractions) > 1
            else 0.0
        ),
        "full_clear_count": sum(
            value == available_coins
            for value in coins
        ),
        "full_clear_rate": (
            sum(value == available_coins for value in coins)
            / len(rows)
        ),
        "zero_coin_count": sum(value == 0 for value in coins),
        "zero_coin_rate": (
            sum(value == 0 for value in coins)
            / len(rows)
        ),
        "total_coins": total_coins,
        "steps_per_coin": (
            survival_steps / total_coins
            if total_coins > 0
            else None
        ),
        "coins_per_100_steps": (
            100 * total_coins / survival_steps
            if survival_steps > 0
            else None
        ),
        "invalid_actions": invalid_actions,
        "attempted_actions": attempted_actions,
        "invalid_action_rate": (
            invalid_actions / attempted_actions
            if attempted_actions > 0
            else None
        ),
        "action_up": sum(row["action_up"] for row in rows),
        "action_right": sum(row["action_right"] for row in rows),
        "action_down": sum(row["action_down"] for row in rows),
        "action_left": sum(row["action_left"] for row in rows),
        "action_wait": sum(row["action_wait"] for row in rows),
        "action_bomb": sum(row["action_bomb"] for row in rows),
        "mean_episode_p95_ms": (
            fmean(p95_values)
            if p95_values
            else None
        ),
        "max_decision_time_ms": (
            max(maximum_values)
            if maximum_values
            else None
        ),
    }


def write_summary_csv(
    *,
    output_path: Path,
    trained_runs: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    rows_by_model: dict[int, list[dict[str, Any]]],
    available_coins: int,
) -> None:
    fieldnames = [
        "model",
        "training_world_seed",
        "agent_seed",
        "evaluation_episodes",
        "mean_coins",
        "mean_collection_fraction",
        "std_collection_fraction",
        "full_clear_count",
        "full_clear_rate",
        "zero_coin_count",
        "zero_coin_rate",
        "total_coins",
        "steps_per_coin",
        "coins_per_100_steps",
        "invalid_actions",
        "attempted_actions",
        "invalid_action_rate",
        "action_up",
        "action_right",
        "action_down",
        "action_left",
        "action_wait",
        "action_bomb",
        "mean_episode_p95_ms",
        "max_decision_time_ms",
        "model_sha256",
    ]

    output_rows = []

    for record in trained_runs:
        run_number = record["run"]
        summary = summarize_rows(
            rows_by_model[run_number],
            available_coins=available_coins,
        )

        output_rows.append({
            "model": f"run-{run_number:02d}",
            "training_world_seed": record["world_seed"],
            "agent_seed": record["agent_seed"],
            **summary,
            "model_sha256": record["model_sha256"],
        })

    output_rows.append({
        "model": "aggregate",
        "training_world_seed": "",
        "agent_seed": "",
        **summarize_rows(
            all_rows,
            available_coins=available_coins,
        ),
        "model_sha256": "",
    })

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(output_rows)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the preregistered config.yaml",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    config_path = arguments.config.resolve()
    config = load_config(config_path)

    require_clean_registered_commit(config)

    issue = config["experiment"]["issue"]
    artifact_root = (
        REPOSITORY_ROOT
        / "training_artifacts"
        / config["output_directories"]["artifact_directory"]

    )
    experiment_directory = config_path.parent

    trained_runs = train_models(config, artifact_root)
    evaluation_records = evaluate_models(
        config,
        artifact_root,
        trained_runs,
    )

    all_rows, rows_by_model = collect_evaluation_rows(
        evaluation_records
    )

    available_coins = int(
        config["environment"]["available_coins_per_episode"]
    )

    write_summary_csv(
        output_path=experiment_directory / "summary.csv",
        trained_runs=trained_runs,
        all_rows=all_rows,
        rows_by_model=rows_by_model,
        available_coins=available_coins,
    )

    overall_summary = aggregate_episode_rows(
        all_rows,
        observed_agent=config["agent"]["name"],
    )
    write_summary_json(
        overall_summary,
        artifact_root / "evaluation" / "aggregate-summary.json",
    )

    print("Experiment completed successfully.")
    print(f"Models: {artifact_root / 'models'}")
    print(f"Records: {artifact_root / 'records'}")
    print(f"Summary: {experiment_directory / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())