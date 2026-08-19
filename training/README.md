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