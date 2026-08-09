# 0002 Repository Architecture

## Decision summary

Use one self-contained folder per learned agent. Keep framework-required
training callbacks and all evaluation-time code inside that folder. Use shared
root directories only for experiment orchestration, compact results, tests, and
documentation.

## Context

The course framework imports an agent from `agent_code/<agent_name>` and loads
`train.py` from the same folder only in training mode. During official
evaluation, the course team copies only the submitted agent directory into an
unchanged framework. Imports from arbitrary repository folders may therefore
work locally but fail in the tournament.

## Chosen structure

```text
agent_code/<agent_name>/
|-- callbacks.py        # Required evaluation interface
|-- train.py            # Required training interface
|-- README.md           # Required agent card
|-- model.py            # Optional learned model implementation
|-- features.py         # Optional state representation
|-- rewards.py          # Optional reward definitions
|-- config.py           # Optional default hyperparameters
|-- requirements.txt    # Only if agent-specific extras are required
`-- model.*             # Trained parameters required at evaluation
```

The exact optional modules can differ by agent. The invariants are:

- the directory can be copied into a clean official framework;
- it contains `callbacks.py`, `train.py`, and an agent card;
- it never requires absolute paths;
- it does not import evaluation-time code from `training/`, `experiments/`, or
  another agent;
- its final policy contains no multiprocessing;
- it loads its trained parameters relative to its own module location.

## Root `training/` directory

The root `training/` directory is appropriate for code that coordinates more
than one agent or is not shipped:

- curriculum launchers;
- hyperparameter sweep definitions;
- multi-seed experiment runners;
- optional parallel training orchestration;
- plotting and aggregation tools;
- compute-environment notes.

This code may call the framework and agent training callbacks, but evaluation
code must not call back into it.

## Root `experiments/` directory

Commit compact and useful scientific evidence:

- experiment plans;
- configuration files;
- small CSV or JSON summaries;
- final plots used for decisions;
- conclusions and links to the implementing commit.

Do not commit raw logs, replay collections, temporary checkpoints, or very large
training artifacts. Store those externally and record their location and
checksum where appropriate.

Recommended naming:

```text
experiments/
`-- YYYY-MM-DD-short-name/
    |-- README.md
    |-- config.yaml
    |-- summary.csv
    `-- figures/
```

## Agent cards

Every learned agent's `README.md` should answer:

- What hypothesis motivated the agent?
- Which learning algorithm is used?
- How is `game_state` represented?
- Which rewards and custom events are used?
- How was the agent trained, including scenarios, opponents, rounds, and seeds?
- Which hyperparameters and model artifact are used?
- What dependencies are required?
- Which baselines and metrics were used?
- What were the results and limitations?
- Which commit produced the stored model?

Copy `agent_code/_team_agent_template/` when beginning a new agent, then rename
the directory immediately.

## Shared code policy

Duplication is preferable to a hidden submission dependency. If multiple agents
share useful feature code, either:

1. copy and version it inside each agent directory; or
2. add a packaging step that vendors it into each agent directory and verify the
   packaged output in CI.

Use option 1 initially because it is simpler and less error-prone. Reconsider
only if duplication becomes a demonstrated maintenance problem.

## Framework changes

Framework changes may support faster training or custom scenarios, but:

- isolate and document them;
- never assume they exist in official evaluation;
- retest trained agents against an unchanged upstream framework;
- record the upstream framework commit used by the experiment.

## Alternatives considered

### One global training package

Rejected for framework callbacks and runtime code because the official
submission contains only one agent directory. Accepted only for orchestration
that is not needed at evaluation.

### One monolithic team agent

Rejected because the assignment requires at least two model approaches and
systematic comparisons. Separate agent directories make variants and artifacts
auditable.

### One agent per team member

Rejected because the handout explicitly values real teamwork across models.
Issues may have owners, but design, review, and experimental conclusions remain
team responsibilities.
