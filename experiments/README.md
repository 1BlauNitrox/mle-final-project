# Experiments

Store reproducible plans and compact result summaries here. Use one directory
per experiment:

```text
experiments/YYYY-MM-DD-short-name/

experiments/
└── YYYY-MM-DD-short-name/
    ├── README.md
    ├── config.yaml
    ├── episodes.csv
    ├── summary.csv
    ├── artifacts.json
    └── figures/
        ├── coin_fraction.png
        ├── learning_curve.png
        └── action_distribution.png
```


Proposed structure for the README.md

```text
# EXPERIMENT_NAME

## Metadata

- Issue: #...
- Author:
- Date:
- Baseline commit:
- Candidate commit:
- Framework commit:

## Research question
Does reachable-path information improve Task 1 coin collection?

## Hypothesis
State the exact prospective hypothesis.

## Baseline
Describe the current agent and artifact.

## Variant
Describe the one change being evaluated.

## Controlled variables
List rewards, hyperparameters, budgets, scenarios and other fixed values.

## Training protocol
List training seeds, agent seeds, episodes and checkpoint-selection rule.

## Evaluation protocol
List development seeds, scenarios, opponents and hardware constraints.

## Metrics and success criterion
State primary metric, guardrails, thresholds and confidence interval.

## Results
Show per-model and aggregate tables and figures.

## Interpretation
Explain what the evidence supports and what it does not support.

## Decision
Accepted, rejected or inconclusive.

## Follow-up
Link the next issue created from the result.
```

The config.yml contains the exact machine-readable protocol:

For example:
```text
experiment: bfs-coin-navigation

baseline:
  agent: DerKleineVermoegensumverteiler
  commit: abc123
  feature_schema: 1

candidate:
  agent: DerKleineVermoegensumverteilerBFS
  commit: def456
  feature_schema: 2

training:
  episodes_per_run: 10000
  world_seeds: [11001, 11002, 11003, 11004, 11005]
  agent_seeds: [21001, 21002, 21003, 21004, 21005]

evaluation:
  scenario: coin-heaven
  opponents: []
  world_seeds: [31001, 31002, 31003]
  training_enabled: false

success:
  minimum_mean_improvement: 0.05
  confidence_interval_lower_bound: 0.0
  maximum_invalid_action_rate: 0.01
  maximum_p95_decision_ms: 50
  required_bomb_count: 0
```


Do not commit raw logs, replay collections, temporary checkpoints, or large
training outputs. Record external artifact locations and checksums when those
artifacts are needed to reproduce a result.

See [`docs/0004-experimentation-protocol.md`](../docs/0004-experimentation-protocol.md).
