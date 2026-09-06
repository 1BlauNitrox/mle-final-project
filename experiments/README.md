# Experiments

Store reproducible plans and compact result summaries here. Use one directory
per experiment:

```text
experiments/YYYY-MM-DD-short-name/

experiments/
└── YYYY-MM-DD-short-name/
    ├── README.md
    ├── config.yaml
    ├── episodes.csv (local unless the claim requires these rows)
    ├── summary.csv
    ├── artifacts.json (only local)
    ├── evidence-manifest.json (when required external evidence exists)
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

## Evidence
List committed per-run/per-seed observations and the exact analysis command.
For required external evidence, record its durable location, SHA-256 checksum,
byte size, contents/schema, retrieval instructions and verification command.

## Interpretation
Explain what the evidence supports and what it does not support.

## Decision
Accepted, rejected or inconclusive.

## Follow-up
Link the next issue created from the result.
```


Do not commit raw logs, replay collections, temporary checkpoints, or large
training outputs merely because they were produced. Commit the smallest
per-run/per-seed evidence that reproduces the reported claims. When a large
artifact is required, record its durable location, SHA-256 checksum, byte size,
contents/schema, retrieval instructions, and verification command. A local path
or checksum without retrievable bytes is not evidence.

See [`docs/0004-experimentation-protocol.md`](../docs/0004-experimentation-protocol.md).
