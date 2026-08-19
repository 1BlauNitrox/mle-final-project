# Training Orchestration

This directory is for tooling shared across experiments and agents:

- curriculum launchers;
- multi-seed runs;
- hyperparameter sweeps;
- result aggregation and plotting;
- optional training-only parallelization.

Framework callbacks and evaluation-time code belong in each agent's own
`agent_code/<agent_name>/` directory. Agents must never import this directory
when `self.train` is false because official evaluation copies only the selected
agent directory.

Every launcher should record the agent commit, configuration, scenarios,
opponents, seeds, rounds, and output location. Follow
[`docs/0004-experimentation-protocol.md`](../docs/0004-experimentation-protocol.md).

# Training and Experiment Pipeline

This directory contains repository-level tooling for reproducible Bomberman
training and evaluation runs.

The pipeline is responsible for:

- launching headless training and evaluation runs;
- recording the exact run configuration and repository state;
- converting framework statistics into episode-level metrics;
- aggregating episode results;
- generating reproducible plots from stored metrics; and
- keeping raw experiment artifacts outside version control.

Agent callbacks, feature extraction, rewards, model updates, and model
persistence belong inside the corresponding `agent_code/<agent_name>/`
directory. Evaluation-time agent code must never import from `training/`.

## Prerequisites

Install the development dependencies from the repository root:

```bash
python -m pip install -r requirements-dev.txt
```

The plotting command requires Matplotlib. It is a development and experiment
dependency, not an evaluation-time dependency of the submitted agent.

Run all commands in this document from the repository root.

## Quick start

### Evaluation smoke run

The following command runs five episodes with the supplied random agent in the
`coin-heaven` scenario:

```bash
python -m training.run_experiment \
  --agent random_agent \
  --mode evaluation \
  --scenario coin-heaven \
  --rounds 5 \
  --world-seed 1
```

A successful run prints the path of its output directory:

```text
Experiment completed: <repository>/training_outputs/<run-id>
```

This command is a pipeline smoke test. It does not measure learning progress
and must not be reported as evidence of agent quality.

### Training run

Use training mode only with an agent that provides the required training
callbacks:

```bash
python -m training.run_experiment \
  --agent <agent_name> \
  --mode training \
  --scenario coin-heaven \
  --rounds 100 \
  --world-seed 1001 \
  --agent-seed 2001
```

In training mode, the observed agent is always the first agent passed to the
framework and the runner adds:

```text
--train 1
```

Only this first agent is trained.

The runner exposes `--agent-seed` through the environment variable:

```text
BOMBERMAN_AGENT_SEED
```

An agent must explicitly read and use this variable for the seed to control its
randomness. Recording an agent seed does not by itself make an agent
reproducible.

### Adding opponents

Pass zero to three opponents after `--opponents`:

```bash
python -m training.run_experiment \
  --agent <agent_name> \
  --mode evaluation \
  --scenario classic \
  --rounds 20 \
  --world-seed 3001 \
  --opponents random_agent peaceful_agent
```

The observed agent remains the first agent in the framework command. Bomberman
supports at most four agents in total.

## Command-line options

Show all runner options with:

```bash
python -m training.run_experiment --help
```

The main options are:

| Option | Description |
| --- | --- |
| `--agent` | Observed agent directory below `agent_code/`. |
| `--mode` | Either `training` or `evaluation`. |
| `--scenario` | Bomberman scenario, such as `coin-heaven` or `classic`. |
| `--rounds` | Positive number of episodes. |
| `--world-seed` | Seed passed to the Bomberman environment. |
| `--agent-seed` | Seed exposed through `BOMBERMAN_AGENT_SEED`. |
| `--opponents` | Optional list of opponent agents. |
| `--output-root` | Parent directory for generated run directories. |

The exact generated framework command is stored as an argument list in
`metadata.json`.

## Generating plots

Plotting is separate from game execution. Generate all figures for an existing
run with:

```bash
python -m training.plot_run \
  training_outputs/<run-id>
```

Use a custom rolling-mean window with:

```bash
python -m training.plot_run \
  training_outputs/<run-id> \
  --rolling-window 20
```

The plotting command reads only `episodes.csv`. It does not rerun the game or
modify the recorded metrics.

Running the plotting command again replaces only the generated files below
`figures/`.

## Finding the latest run

From the repository root:

```bash
latest_run="$(
  find training_outputs \
    -mindepth 1 \
    -maxdepth 1 \
    -type d \
    | sort \
    | tail -n 1
)"
```

Display the selected path:

```bash
echo "$latest_run"
```

Inspect its metadata:

```bash
python -m json.tool "$latest_run/metadata.json"
```

Inspect its summary:

```bash
python -m json.tool "$latest_run/summary.json"
```

Generate its plots:

```bash
python -m training.plot_run "$latest_run"
```

## Successful and failed runs

A successful run records:

```json
{
  "status": "completed",
  "return_code": 0,
  "error": null
}
```

If the game process or post-processing fails, the run directory is retained and
`metadata.json` records:

```json
{
  "status": "failed",
  "return_code": 1,
  "error": "RuntimeError: ..."
}
```

A return code of zero together with `status: failed` means that the game process
completed but metrics normalization, aggregation, or another post-processing
step failed.

Failed run directories are retained for diagnosis. The pipeline must not create
a misleading `summary.json` when the game process itself fails.

## Output files

A successful run produces:

```text
training_outputs/<run-id>/
|-- metadata.json
|-- framework_stats.json
|-- episodes.csv
|-- summary.json
`-- figures/
    |-- learning_curve.png
    |-- task_metrics.png
    `-- behavior_diagnostics.png
```

The game runner creates the JSON and CSV files. The separate plotting command
creates the `figures/` directory.

### `metadata.json`

Contains the command, configuration, seeds, Git revision, dirty-worktree
indicator, Python version, timing, process status, and return code.

### `framework_stats.json`

Contains the nested statistics exported by the Bomberman framework. This file
is retained as the source record used by normalization.

### `episodes.csv`

Contains one normalized row per observed agent and episode. It is the stable
input for aggregation and plotting.

### `summary.json`

Contains overall and per-agent aggregates, including:

- coins and score;
- episode length and steps per coin;
- invalid-action rate;
- survival and termination counts;
- action totals and distributions;
- decision-time summaries; and
- available learning-specific metrics.

### `figures/learning_curve.png`

Shows coins and environment score over episodes, including rolling means.

### `figures/task_metrics.png`

Shows episode length and invalid-action rate.

### `figures/behavior_diagnostics.png`

Shows the action distribution and, when available, learning diagnostics such as
epsilon, Q-table size, and mean absolute temporal-difference error.

## Metric interpretation

### Invalid-action rate

The per-episode rate is:

```text
invalid_actions / attempted_actions
```

When no action was attempted, the value is unavailable rather than zero.

The aggregate rate is calculated from totals:

```text
sum(invalid_actions) / sum(attempted_actions)
```

It is not the unweighted mean of episode-level rates.

### Steps per collected coin

The diagnostic uses:

```text
episode_steps / max(coins_collected, 1)
```

A zero-coin episode therefore retains the full episode length as a worst-case
value instead of causing division by zero.

### Decision time

Per-episode metrics include the median, 95th percentile, and maximum duration of
all recorded `act()` calls.

The aggregate summary reports:

- the mean of episode medians;
- the mean of episode 95th percentiles; and
- the maximum of all episode maxima.

The mean of episode percentiles is not a global percentile over every individual
`act()` call and must not be described as one.

### Optional learning metrics

Agents may provide:

- cumulative shaped reward;
- epsilon;
- Q-table size; and
- mean absolute temporal-difference error.

Missing optional values remain empty. They are never replaced with zero because
zero is a valid measurement and has a different meaning from unavailable data.

## Reproducibility requirements

A run is not reproducible unless its record identifies:

- the Git commit;
- whether the worktree was dirty;
- the exact command;
- the agent and mode;
- the scenario and opponents;
- the number of episodes;
- the world seed;
- the agent seed when available; and
- the runtime version.

Training, development evaluation, and final held-out evaluation must use
separate seed populations as defined in
`docs/0007-task-1-baseline-contract.md`.

Do not tune features, rewards, hyperparameters, or model selection against the
final held-out seed set.

## Repository hygiene

The complete `training_outputs/` directory is ignored by Git. Do not commit:

- raw run directories;
- game logs;
- temporary plots;
- replay collections;
- accidental smoke-test models; or
- temporary checkpoints.

Compact summaries and selected final figures may be copied into a versioned
directory below `experiments/` only when they provide evidence for a documented
project decision.

Check ignored output with:

```bash
git status --ignored --short training_outputs
```

## Scientific scope

The pipeline records and visualizes observations. It does not establish that an
agent:

- learned successfully;
- converged;
- outperforms a baseline;
- satisfies the Task 1 completion contract; or
- is tournament-ready.

A 5–10 episode smoke run verifies only that the instrumentation works. Any
performance conclusion requires a preregistered experiment with controlled
variables, fixed seed populations, appropriate baselines, and uncertainty
reporting.

## Experiment output schema

Each experiment run writes its artifacts to a dedicated directory below
`training_outputs/`. Raw run directories are local artifacts and must not be
committed.

```text
training_outputs/<run-id>/
|-- metadata.json
|-- framework_stats.json
|-- episodes.csv
|-- summary.json
`-- figures/
    |-- learning_curve.png
    |-- task_metrics.png
    `-- behavior_diagnostics.png
```

The schema is versioned independently of the application version. The initial
schema version is `1`. Readers must reject unsupported schema versions with a
clear error message. Additional unknown fields should be tolerated to permit
backwards-compatible extensions.

### Run metadata

The file `metadata.json` describes how the run was produced. It contains the
following fields:

| Field | Type | Description |
| --- | --- | --- |
| `schema_version` | integer | Version of the experiment-output schema. The initial version is `1`. |
| `run_id` | string | Unique identifier of the run. |
| `agent` | string | Name of the observed agent below `agent_code/`. |
| `mode` | string | Either `training` or `evaluation`. |
| `scenario` | string | Bomberman scenario used for the run, for example `coin-heaven`. |
| `opponents` | array of strings | Ordered list of opposing agents. Empty when the observed agent plays alone. |
| `rounds` | integer | Number of requested episodes. |
| `world_seed` | integer or null | Seed passed to the Bomberman environment. |
| `agent_seed` | integer or null | Seed controlling agent-side randomness, when available. |
| `command` | array of strings | Exact command and arguments used to start the game. |
| `git_commit` | string | Full Git commit SHA from which the run was launched. |
| `git_dirty` | boolean | Whether the worktree contained uncommitted changes when the run started. |
| `started_at` | string | Run start time as an ISO 8601 UTC timestamp. |
| `duration_seconds` | number | Total wall-clock duration of the run in seconds. |
| `python_version` | string | Python version used to execute the run. |
| `status` | string | Either `running`, `completed`, or `failed`. |
| `return_code` | integer or null | Exit code of the game process. It is `null` while the run is still active. |

Unavailable values are stored as JSON `null`. Values must not be inferred when
the pipeline cannot determine them reliably. The command is stored as an array
rather than a shell-formatted string so that argument boundaries remain
unambiguous.

Example:

```json
{
  "schema_version": 1,
  "run_id": "2026-08-19T143000Z-random-agent-coin-heaven",
  "agent": "random_agent",
  "mode": "evaluation",
  "scenario": "coin-heaven",
  "opponents": [],
  "rounds": 5,
  "world_seed": 1,
  "agent_seed": null,
  "command": [
    "python",
    "main.py",
    "play",
    "--agents",
    "random_agent",
    "--no-gui",
    "--n-rounds",
    "5",
    "--scenario",
    "coin-heaven",
    "--seed",
    "1"
  ],
  "git_commit": "0123456789abcdef0123456789abcdef01234567",
  "git_dirty": false,
  "started_at": "2026-08-19T14:30:00Z",
  "duration_seconds": 1.42,
  "python_version": "3.13.7",
  "status": "completed",
  "return_code": 0
}
```

### Episode metrics

The file `episodes.csv` contains one row per observed agent and episode. Its
required columns are:

| Column | Type | Description |
| --- | --- | --- |
| `schema_version` | integer | Version of the experiment-output schema. |
| `round` | integer | One-based episode number. |
| `agent` | string | Name of the observed agent. |
| `mode` | string | Either `training` or `evaluation`. |
| `episode_steps` | integer | Number of actions attempted during the episode. |
| `score` | integer | Environment score obtained during the episode. |
| `coins_collected` | integer | Number of coins collected during the episode. |
| `invalid_actions` | integer | Number of actions rejected by the environment. |
| `attempted_actions` | integer | Number of actions returned by the agent. |
| `invalid_action_rate` | number or empty | `invalid_actions / attempted_actions`. Empty when no action was attempted. |
| `survived` | boolean | Whether the agent was alive when the episode ended. |
| `termination_reason` | string | Recorded termination category, such as `survived`, `killed`, `step_limit`, or `unknown`. |
| `action_up` | integer | Number of attempted `UP` actions. |
| `action_right` | integer | Number of attempted `RIGHT` actions. |
| `action_down` | integer | Number of attempted `DOWN` actions. |
| `action_left` | integer | Number of attempted `LEFT` actions. |
| `action_wait` | integer | Number of attempted `WAIT` actions. |
| `action_bomb` | integer | Number of attempted `BOMB` actions. |
| `decision_time_median_ms` | number or empty | Median wall-clock duration of `act()` calls in milliseconds. |
| `decision_time_p95_ms` | number or empty | 95th percentile of `act()` durations in milliseconds. |
| `decision_time_max_ms` | number or empty | Maximum `act()` duration in milliseconds. |

The following agent-specific learning metrics are optional:

| Column | Type | Description |
| --- | --- | --- |
| `shaped_reward` | number | Cumulative shaped reward for the episode. |
| `epsilon` | number | Exploration probability associated with the episode. |
| `q_table_size` | integer | Number of represented states or state-action entries, as defined by the agent. |
| `mean_abs_td_error` | number | Mean absolute temporal-difference error for the episode. |

Missing optional metrics are represented by empty CSV fields. They must not be
replaced by zero because zero is a valid measured value and has a different
meaning from unavailable data.

`invalid_action_rate` is calculated as:

```text
invalid_actions / attempted_actions
```

When `attempted_actions` is zero, the rate is unavailable and the CSV field is
left empty.

For aggregate reports, the overall invalid-action rate is calculated from the
totals:

```text
sum(invalid_actions) / sum(attempted_actions)
```

It is not calculated as the unweighted mean of the per-episode rates.

Decision-time fields include every recorded `act()` call in the episode. When no
decision time was recorded, all three fields are left empty. The percentile
implementation used by the pipeline must remain consistent and be covered by
tests.

The action counts should normally satisfy:

```text
attempted_actions =
    action_up
    + action_right
    + action_down
    + action_left
    + action_wait
    + action_bomb
```

A mismatch indicates incomplete instrumentation or malformed input and must be
reported during schema validation.

The generic pipeline records observations only. It does not interpret a smoke
run as evidence of agent quality, learning progress, convergence, or Task 1
completion.