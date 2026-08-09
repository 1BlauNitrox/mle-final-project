# 0004 Experimentation Protocol

## Goal

Experiments must support a scientific claim about agent performance, not merely
produce a trained model. Every important design change should have an explicit
hypothesis and a controlled comparison.

## Before training

Record:

- hypothesis;
- independent variable;
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

Small CSV/JSON summaries and final plots may be committed. Raw replay data,
temporary logs, and large intermediate checkpoints remain outside Git.

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
