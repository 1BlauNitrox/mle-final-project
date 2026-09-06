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

### Staged run plans

Use a version-one YAML run plan when an experiment needs curricula, independent
replicas, protected seed populations, regression suites, or resumable execution.
Always inspect the complete matrix and compute budget before starting it:

```bash
python -m training.run_plan training/run_plans/task3-example.yaml --dry-run
```

Execute a validated plan, or resume only its unfinished and failed jobs:

```bash
python -m training.run_plan training/run_plans/task3-example.yaml
python -m training.run_plan training/run_plans/task3-example.yaml --resume
```

`--resume` rejects any change to the configuration, repository training source,
framework, observed-agent directory, dependency versions, or parent artifacts.
Failed and interrupted attempts remain under their job directory; a retry starts
from the last completed stage rather than a partial checkpoint.

The bounded integration plan exercises the Task 1, Task 2, and Task 3 shapes
without training or making a performance claim:

```bash
python -m training.run_plan training/run_plans/tasks1-3-smoke.yaml
```

#### YAML schema version 1

Unknown mapping fields are tolerated for compatible metadata extensions. The
following fields control execution:

| Field | Type | Default | Rules |
| --- | --- | --- | --- |
| `schema_version` | integer | none | Required and exactly `1`. |
| `plan_id` | string | none | Required filesystem-safe identifier; selects the immutable plan output directory. |
| `agent` | string | none | Required existing directory below `agent_code/`. |
| `artifact_path` | string or null | `null` | Relative file path inside the staged agent. Required for training; absolute paths and `..` are rejected. |
| `max_parallel_training` | positive integer | `1` | Bounds independent replica workers. Evaluation remains serial and single-process. |
| `replicas` | list | none | At least one independent replica with a unique `id`, non-negative `world_seed`, non-negative `agent_seed`, and optional `parent_artifact`. |
| `training_stages` | list | `[]` | Ordered stages with unique `id`, supported `scenario`, positive `rounds`, and zero to three ordered `opponents`. |
| `evaluation_suites` | list | `[]` | Suites with unique `id`, protected `population`, scenario, rounds per seed pair, equal-length seed lists, and ordered opponents. |

Nested records use these fields:

| Record | Field | Type/default | Validation and meaning |
| --- | --- | --- | --- |
| replica | `id` | required string | Unique safe identifier and output-directory component. |
| replica | `world_seed` | required integer | Non-negative environment seed used for every ordered training stage. |
| replica | `agent_seed` | required integer | Non-negative observed-agent seed used for every ordered training stage. |
| replica | `parent_artifact` | path or `null` | Defaults to `null`; plan-file-relative or absolute existing file whose checksum is protected on resume. |
| stage | `id` | required string | Unique safe stage identifier. |
| stage | `scenario` | required string | One of the scenarios defined by the current framework. |
| stage | `rounds` | required integer | Positive exact episode budget for this stage and replica. |
| stage | `opponents` | list, default `[]` | Zero to three explicitly ordered supported agents. |
| suite | `id` | required string | Unique safe suite identifier. |
| suite | `population` | required string | `development`, `confirmation`, or `final`. |
| suite | `scenario` | required string | One current framework scenario. |
| suite | `rounds` | required integer | Positive episodes per paired seed measurement. |
| suite | `world_seeds` | required integer list | Non-empty, unique, non-negative environment seeds. |
| suite | `agent_seeds` | required integer list | Non-empty, unique, non-negative observed-agent seeds; same length as `world_seeds`. |
| suite | `opponents` | list, default `[]` | Zero to three explicitly ordered supported agents. |

`population` is one of `development`, `confirmation`, or `final`; replica seeds
form the protected `training` population. No numeric seed may occur in two
populations. World and observed-agent seed lists are explicit and paired by
position. Duplicate IDs or seeds, unsupported framework scenarios, unknown
opponents, non-positive budgets, and ambiguous artifact paths fail validation
before the output directory or first episode is created. Supplied opponents and
existing team-agent directories are supported; list order is preserved in
metadata and the framework command.

Stage order is YAML list order. Exact `rounds` values determine both the episode
budget and curriculum proportions, so expansion is deterministic for fixed
configuration and seeds. The selected checkpoint is mechanically the artifact
present after the exact final episode of each stage.

Example plans are in `training/run_plans/`:

- `task1-example.yaml`: `coin-heaven` without opponents;
- `task2-example.yaml`: visible coins, `loot-crate`, then `classic`, without opponents;
- `task3-example.yaml`: `classic` with peaceful and coin-collector opponents plus Task 1/2 regressions;
- `task4-example.yaml`: `classic` with ordered strong opponents plus earlier-task regressions; and
- `tasks1-3-smoke.yaml`: three one-round integration checks.

### Issue #46 Task 2 DQN development baseline

Issue #46 uses three immutable plans so the five trained artifacts and both
single-artifact baselines are evaluated separately on the same seeds. Prepare
the seed-compatible starts before loading the trained plan:

```bash
python -m training.prepare_dqn_task2_experiment
python -m training.run_plan training/run_plans/issue46-dqn-task2-trained.yaml --dry-run
python -m training.run_plan training/run_plans/issue46-dqn-task2-untrained.yaml --dry-run
python -m training.run_plan training/run_plans/issue46-dqn-task1-frozen.yaml --dry-run
```

The preparation command is idempotent: it verifies and reuses a complete
existing set, but refuses to overwrite an incomplete or changed directory.
Execute each validated plan by omitting `--dry-run`. Resume an interrupted plan
with `--resume`. When all three plans report `completed`, run:

```bash
python -m training.analyze_dqn_task2_experiment
```

The complete raw record, including deterministic repeats, remains below
`training_outputs/`; the prospective protocol is in
`experiments/2026-09-05-dqn-task2-development/`.

If an instrumentation defect invalidates completed evaluations while final
training workspaces are intact, preserve the old plan directory and execute a
fresh evaluation-only plan into a new output root:

```bash
python -m training.run_plan \
  training/run_plans/issue46-dqn-task2-trained.yaml \
  --evaluation-only \
  --workspace-root training_outputs/run-plans-invalid-denominator \
  --output-root training_outputs/run-plans-corrected
```

Repeat the command for the untrained and frozen Task 1 plans. The workspace
root must contain the corresponding old plan directories and their final
workspaces; evaluation-only mode never runs training and cannot resume an
existing output directory.

### Issue #97 staged-curriculum comparison for Task 2 DQN

**Backlog, blocked on Issue #86.** Two direct-classic plans are prepared —
`issue97-dqn-task2-direct-classic-unmasked.yaml` and
`issue97-dqn-task2-direct-classic-masked.yaml` — but only one will actually be
executed, whichever masking mode #86 adopts. Both dry-run cleanly on current
`main`; the `*-masked` plan's `action_masking` field is silently ignored until
PR #95 (Issue #86) merges, so it must not be executed for real before then. See
`experiments/2026-09-06-dqn-task2-curriculum/README.md` for the full protocol
and blocking dependencies.

```bash
python -m training.run_plan training/run_plans/issue97-dqn-task2-direct-classic-unmasked.yaml --dry-run
python -m training.run_plan training/run_plans/issue97-dqn-task2-direct-classic-masked.yaml --dry-run
```

#### Plan output and resume records

Before execution, the runner writes `resolved_plan.json` once and never mutates
it. `status.json` records plan/job state, timestamps, errors, and every attempt.
Each replica has its own staged agent workspace, each job has its own directory,
and each completed training stage has a checksum-labelled artifact copy:

```text
training_outputs/run-plans/<plan-id>/
|-- resolved_plan.json
|-- status.json
|-- replicas/<replica-id>/agent/
|-- artifacts/<replica-id>/<stage-id>/<artifact>
`-- jobs/<run-id>/
    |-- attempt-001/
    |-- attempt-001-input-agent/
    `-- attempt-001-failed-agent/  # only after failure/interruption
```

Every attempt delegates to `training.run_experiment`, so `framework_stats.json`,
`episodes.csv`, `summary.json`, and the existing plotting/aggregation interfaces
remain unchanged. Evaluation metadata explicitly records training off, one
process, and an immutable artifact; the runner verifies the artifact checksum
after every evaluation job. Evaluation suites use the replica workspace after
the final ordered training stage; earlier stage checkpoints are archived for
lineage and are not selected implicitly.

### Issue #41 registered DQN series

The five `DagobertDuckDQN` development-baseline runs are controlled by their
preregistered launcher:

```bash
python -m training.run_dqn_task1_baseline
```

It runs the five fixed 10,000-episode jobs serially, requires a clean worktree
and the registered source fingerprint, rejects seed collisions and an existing
checkpoint, and moves every final checkpoint into the ignored series directory
before the next independent run. It is not a general sweep interface. The
prospective protocol and blockers are recorded under
`experiments/2026-09-01-dqn-task1-development-baseline/`.

The runner exposes `--agent-seed` through the environment variable:

```text
BOMBERMAN_AGENT_SEED
```

An agent must explicitly read and use this variable for the seed to control its
randomness. Recording an agent seed does not by itself make an agent
reproducible.

The registered evaluator stages each immutable model as the ignored local file
`agent_code/DagobertDuckDQN/.evaluation-checkpoint.pt`. It passes only that file
name to the agent, which still resolves the artifact relative to its own module.
This keeps the worktree clean, avoids redundant per-job snapshots, and leaves
the normal submission path `checkpoint.pt` unchanged.

After evaluation, export the reviewable observations and verify every committed
table and figure with:

```bash
python -m training.dqn_task1_evidence export \
  training_outputs/issue-41-dqn-task1-baseline/<series-id> \
  experiments/2026-09-01-dqn-task1-development-baseline/evidence
python -m training.dqn_task1_evidence verify \
  experiments/2026-09-01-dqn-task1-development-baseline
```

The evidence bundle contains all evaluation episode rows, all raw evaluation
decision times, all training episode rows, the failed training run, artifact
hashes, and a sanitized evaluation manifest. The verifier checks evidence
hashes and rebuilds the committed summaries and figures byte-for-byte.

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

The plotting command reads `episodes.csv` and the observed-agent name from
`metadata.json`. It does not rerun the game or modify the recorded metrics.

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

Contains the command, observed agent, agent-directory fingerprint and optional
dirty-state snapshot, seeds, Git revision, Python version, timing, process
status, and return code.

### `framework_stats.json`

Contains the nested statistics exported by the Bomberman framework. This file
is retained as the source record used by normalization.

### `episodes.csv`

Contains one normalized row per participating agent and episode. It is the
stable input for aggregation and plotting.

### `summary.json`

Contains observed-agent aggregates under `overall` and separate aggregates for
every participant under `by_agent`, including:

- coins and score;
- episode length, steps per coin, and coins per 100 steps;
- invalid-action rate;
- survival and termination counts;
- action totals and distributions;
- decision-time summaries; and
- available learning-specific metrics.

### `figures/learning_curve.png`

Shows coins and environment score over episodes, including rolling means. Each
participant is displayed as a separate series for comparison.

### `figures/task_metrics.png`

Shows episode length and invalid-action rate. Each participant is displayed as
a separate series for comparison.

### `figures/behavior_diagnostics.png`

Shows the observed agent's action distribution and, when available, learning
diagnostics such as epsilon, Q-table size, and mean absolute temporal-difference
error.

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

The aggregate diagnostic uses the ratio of totals:

```text
sum(survival_steps) / sum(coins_collected)
```

Zero-coin episodes contribute all of their steps to the numerator without
introducing a fictitious collected coin. When the complete group collected no
coins, the value is unavailable and stored as JSON `null`.

For plotting and direct comparisons, the summary also reports:

```text
coins_per_100_steps = 100 * sum(coins_collected) / sum(survival_steps)
```

This rate is zero for a non-empty run with no collected coins. The summary also
records the zero-coin episode count and rate so that unstable agents cannot hide
failure episodes behind an aggregate efficiency value.

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

In training mode, an agent's `end_of_round` callback may return a mapping of
numeric episode diagnostics:

```python
def end_of_round(self, last_game_state, last_action, events):
    # Update the model and save it as usual.
    return {
        "shaped_reward": self.episode_reward,
        "epsilon": self.epsilon,
        "q_table_size": len(self.q_table),
        "mean_abs_td_error": self.mean_abs_td_error,
    }
```

The framework stores this mapping below the per-agent `learning_metrics` key in
`framework_stats.json`. The normalizer copies the supported schema fields into
`episodes.csv`, after which aggregation and plotting use them without importing
`training/` from the agent.

The version-one episode schema supports:

- cumulative shaped reward;
- epsilon;
- Q-table size; and
- mean absolute temporal-difference error.

Missing optional values remain empty. They are never replaced with zero because
zero is a valid measurement and has a different meaning from unavailable data.
Metric names must be non-empty strings and values must be finite numbers or
`None`. Returning `None` from `end_of_round`, as existing agents do, records an
empty `learning_metrics` object.

## Reproducibility requirements

A run is not reproducible unless its record identifies:

- the Git commit;
- whether the worktree was dirty;
- the agent-directory path and content hash;
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

When the worktree is dirty, the runner also creates `agent_snapshot/` before
starting the game so that the exact agent state remains recoverable.

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
backwards-compatible extensions. Artifacts created while this initial schema
was still under development are not a supported earlier schema version.

### Run metadata

The file `metadata.json` describes how the run was produced. It contains the
following fields:

| Field | Type | Description |
| --- | --- | --- |
| `schema_version` | integer | Version of the experiment-output schema. The initial version is `1`. |
| `run_id` | string | Unique identifier of the run. |
| `agent` | string | Name of the observed agent below `agent_code/`. |
| `observed_agent` | string | Runtime name used for the first agent in framework statistics. This includes the `_0` suffix when the same agent type also appears as an opponent. |
| `mode` | string | Either `training` or `evaluation`. |
| `scenario` | string | Bomberman scenario used for the run, for example `coin-heaven`. |
| `opponents` | array of strings | Ordered list of opposing agents. Empty when the observed agent plays alone. |
| `rounds` | integer | Number of requested episodes. |
| `world_seed` | integer or null | Seed passed to the Bomberman environment. |
| `agent_seed` | integer or null | Seed controlling agent-side randomness, when available. |
| `agent_configuration` | object | Agent-directory path and SHA-256 content hash captured before the run. When the worktree is dirty, `snapshot_path` points to a preserved copy inside the run directory. |
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
  "observed_agent": "random_agent",
  "mode": "evaluation",
  "scenario": "coin-heaven",
  "opponents": [],
  "rounds": 5,
  "world_seed": 1,
  "agent_seed": null,
  "agent_configuration": {
    "path": "agent_code/random_agent",
    "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    "snapshot_path": null
  },
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

The file `episodes.csv` contains one row per participating agent and episode. Its
required columns are:

| Column | Type | Description |
| --- | --- | --- |
| `schema_version` | integer | Version of the experiment-output schema. |
| `round` | integer | One-based episode number. |
| `agent` | string | Name of the observed agent. |
| `mode` | string | Either `training` or `evaluation`. |
| `episode_steps` | integer | Total environment steps in the episode. |
| `survival_steps` | integer | Environment step on which the agent died, or the total episode steps when it survived. |
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
| `action_unknown` | integer | Number of attempted actions outside the six supported action names. |
| `decision_time_median_ms` | number or empty | Median wall-clock duration of `act()` calls in milliseconds. |
| `decision_time_p95_ms` | number or empty | 95th percentile of `act()` durations in milliseconds. |
| `decision_time_max_ms` | number or empty | Maximum `act()` duration in milliseconds. |

The following agent-specific learning metrics are optional:

| Column | Type | Description |
| --- | --- | --- |
| `shaped_reward` | number | Cumulative shaped reward for the episode. |
| `epsilon` | number | Exploration probability associated with the episode. |
| `q_table_size` | integer | Number of represented states or state-action entries, as defined by the agent. |
| `replay_size` | integer | Number of transitions currently stored in experience replay. |
| `update_count` | integer | Cumulative optimizer-update count. |
| `mean_loss` | number | Mean optimizer loss for updates in the episode. |
| `mean_abs_td_error` | number | Mean absolute temporal-difference error for the episode. |
| `target_synchronizations` | integer | Cumulative target-network synchronization count. |
| `episode_target_synchronizations` | integer | Target-network synchronizations performed during the episode. |

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
    + action_unknown
```

A mismatch indicates incomplete instrumentation or malformed input and must be
reported during schema validation.
Stored counters must be non-negative, invalid actions cannot exceed attempted
actions, and stored numeric measurements must be finite.

The generic pipeline records observations only. It does not interpret a smoke
run as evidence of agent quality, learning progress, convergence, or Task 1
completion.
