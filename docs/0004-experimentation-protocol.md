# 0004 Experimentation Protocol

## Goal

Experiments must support a scientific claim about agent performance, not merely
produce a trained model. Every important design change should have an explicit
hypothesis and a controlled comparison.

## Scope of a single experiment: main effects and interactions

Change one main variable at a time by default. This supports the assignment's
requirement to show how performance changes after a modification and permits a
claim about that modification's effect. Issue #58's movement-shaping test
against #41 is this repository's reference example.

A registered experiment may vary more than one factor only when its
prospectively stated hypothesis explicitly concerns an interaction or a
factorial comparison. Before training, name every factor and level and define
the ablations needed to estimate the intended main and interaction effects.
Use the same controls, seeds, evaluation protocol, and stopping rule across all
cells. The resulting claims must not exceed what those planned contrasts can
support.

Changing unrelated features, rewards, and hyperparameters together merely to
save compute is not a controlled experiment. Such a build may be used for a
bounded integration smoke or documented exploratory diagnosis, but its result
cannot select the shipped defaults or support a causal improvement claim.
Record observations transparently, then register an isolated or interaction
experiment before using them to make a design decision.

## Before training

Record:

- hypothesis;
- independent variable, or factors and planned ablations for an explicitly
  stated interaction/factorial experiment;
- controlled variables;
- baseline or previous agent revision;
- scenarios and opponents;
- game and model random seeds;
- number of training and evaluation rounds;
- metrics;
- success criterion;
- expected compute cost;
- implementing issue, branch, and commit.

Do not choose the success criterion after seeing the result.

## Protocol and result changes

The protocol and its results are different reviewable outcomes:

1. A **prospective protocol PR** registers the items above, adds or configures
   the execution and analysis paths, and demonstrates that they parse, expand,
   or dry-run successfully. It does not need training or evaluation results.
   It must explicitly state that no scientific run or performance conclusion
   is part of the PR.
2. A **completed experiment PR** records the executed immutable revisions and
   artifacts, retained observations, aggregate results, uncertainty,
   interpretation, and decision. It may close the experiment issue only when
   all of that issue's acceptance criteria are satisfied.
3. A **partial or incomplete result PR** preserves useful evidence without
   pretending the experiment is complete. It documents failed or missing runs,
   invalid data, and unsupported claims, uses `Refs #<issue>`, and leaves the
   remaining acceptance criteria open.

Separating the protocol and result into two PRs is recommended when review or
long-running compute would otherwise overlap. The experiment issue remains open
between them unless a separate protocol issue was intentionally scoped to end
before execution.

## Minimum evaluation metrics

Use more than one metric because tournament score alone can hide failure modes:

- mean score per round;
- win or first-place rate;
- survival rate and mean survival steps;
- coins collected per round;
- opponents eliminated per round;
- self-kill rate;
- invalid-action rate;
- decision-time median, 95th percentile, and maximum;
- confidence interval or variation across seeds.

Task-specific metrics are encouraged, for example coin-collection efficiency in
Task 1 or bomb-escape success in Task 2.

## Baselines

Compare against suitable supplied agents:

- `random_agent`;
- `peaceful_agent`;
- `coin_collector_agent`;
- `rule_based_agent`;
- the previous revision of the same learned agent;
- other team agents when selecting the final model.

Use the same scenarios, opponent slots, seeds, and number of episodes for
compared variants.

## Randomness

- Use multiple fixed evaluation seeds.
- Keep evaluation seeds separate from training seeds.
- Record seed lists with the experiment.
- Report aggregate results and variation, not only the best run.
- Avoid tuning against the final evaluation seed set.

## Progressive curriculum

Evaluate all agents against the four project stages:

1. visible-coin navigation;
2. crate destruction and bomb survival;
3. hunting peaceful and coin-collecting opponents;
4. competitive play against strong agents.

An agent progressing to a later stage should retain earlier capabilities.
Include regression metrics from earlier stages.

## Result record

Create `experiments/YYYY-MM-DD-short-name/README.md` with:

```markdown
# Experiment title

## Metadata
- Issue:
- Commit:
- Agent:
- Date:
- Owner/reviewer:

## Hypothesis

## Setup
- Training seeds:
- Evaluation seeds:
- Scenarios:
- Opponents:
- Rounds:
- Hardware:

## Metrics and success criterion

## Results

## Interpretation

## Decision and follow-up
```

### Minimum evidence package

Commit the smallest evidence package that lets a reviewer check the reported
claim without access to the author's machine. It normally contains:

- the prospective protocol and exact executed configuration;
- immutable agent, framework, and analysis revisions;
- model and input-artifact identifiers and checksums;
- compact per-run or per-seed observations sufficient to recompute the reported
  aggregates and uncertainty;
- aggregate CSV or JSON tables and final plots used for the decision;
- the conclusion, limitations, and follow-up decision; and
- an exact analysis or verification command.

The necessary granularity depends on the claim. Seed-level observations are
usually sufficient for means and across-seed variation. Per-episode rows are
required only when the reported statistic or resampling method cannot be
reconstructed from a smaller lossless summary. Every intermediate checkpoint,
replay transition, and verbose log is not evidence by default.

### Large and external evidence

Raw replay data, verbose logs, temporary checkpoints, and large training
artifacts remain outside Git. If any such object is required to verify a claim,
the experiment record or a committed manifest must provide:

- a durable downloadable location or immutable release identifier;
- its SHA-256 checksum and byte size;
- a description of its contents and schema;
- exact retrieval instructions; and
- an exact verification or reproduction command.

A machine-local path or checksum without access to the referenced bytes is not
sufficient. If required evidence has been lost or cannot be shared, preserve
the available record, label the result incomplete or unverified, narrow the
conclusion, and keep the experiment issue open.

### Frozen-model evidence

Freezing a candidate records a selection decision, not a new performance
experiment. Retain the selected evaluation artifact in its submission-ready
agent directory and document its SHA-256 checksum, byte size, producing
configuration and code revision, source-checkpoint provenance, prospective
selection rule, and the evaluation evidence used for selection. If reproducing
the export requires a large source checkpoint, publish that source through the
external-evidence contract above and provide an exact export and verification
command. Unselected and intermediate checkpoints need not be published unless
they are required to audit the stated selection rule.

## Performance and compatibility

Before accepting a candidate:

- evaluate with training disabled;
- verify 0.5-second action latency on CPU;
- verify memory use remains below 8 GB;
- verify there is no evaluation multiprocessing;
- copy only the agent directory into a clean framework;
- run against three random agents;
- run the official Docker compatibility test;
- verify all file paths are relative.

## Reporting negative results

Failed hypotheses are valuable. Record them with enough information to prevent
the team from repeating the same experiment. Explain why the evidence did not
support the idea and what decision followed.
