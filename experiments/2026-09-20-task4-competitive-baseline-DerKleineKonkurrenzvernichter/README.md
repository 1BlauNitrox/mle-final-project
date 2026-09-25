# Tabular Task 4 competitive baseline

> status: completed-failed

## Metadata

- Issue: #228
- Agent: `DerKleineKonkurrenzvernichter`
- Date: 2026-09-20
- Plan: `training/run_plans/issue228-task4-competitive-baseline.yaml`

## Research question and hypothesis

How good will the task 3 agent perform in task 4?

## Protocol

Five independent replicas use the last excepted task 3 configuration.
Each trains for exactly 10,000 Classic episodes against three rule-based
opponents. Final checkpoints are evaluated on 20 fixed seed pairs per
replica in competitive, mixed-opponent, peaceful-opponent, opponent-free
Classic, Coin Heaven, and Loot Crate suites, followed by exact repeats.

The primary metric is strict first-place rate against three rule-based agents;
ties are reported separately and are not wins. A cluster bootstrap over the
five replica means provides a 95% interval. If every registered gate passes,
the selected checkpoint follows the fixed ordering in `config.yaml`; otherwise
no checkpoint is promoted.

## Results

The run plan completed all 1,205 jobs without a failed job. Across the five
replicas, strict first-place rate against three rule-based agents was 0.04,
below the registered 0.10 gate. The cluster-bootstrap 95% interval was
[0.02, 0.05]. Four replicas achieved a positive first-place rate, but no
replica exceeded 0.05. Tied-first rate was 0.05, mean placement was 2.83, mean
score was 1.93, survival rate was 0.26, and the agent eliminated 0.07 opponents
per episode.

Competitive and mixed-opponent self-kill rates were 0.41 and 0.47. The
equal-suite aggregate remained below the registered 0.20 limit at 0.182 because
the earlier-task suites were safer. Peaceful-opponent eliminations were 0.08,
below the 0.10 retention gate.

The collection gates passed: opponent-free Classic collection was 0.164, Coin
Heaven was 1.000, and Loot Crate was 0.204. Decision latency also passed; the
worst replica/suite p95 was 3.75 ms and the maximum was 11.65 ms. Exact repeats
failed in 212 of 600 primary/repeat pairs, so the determinism gate failed.

The prospective ranking ordered the checkpoints `r5`, `r4`, `r1`, `r2`, `r3`.
Because not every gate passed, the registered rule selects no checkpoint.

## Interpretation

Training the unchanged Task 3 state and reward design directly against three
rule-based opponents produces occasional strict wins but not a viable Task 4
policy. The confidence interval remains entirely below the registered 0.10
target, so the failure is not explained merely by one weak replica. Positive
mean score also does not imply competitive success because the agent survives
only 26% of competitive episodes and frequently kills itself.

Earlier coin-collection capability is retained at the registered thresholds,
but peaceful hunting regresses below its gate. A next Task 4 experiment should
therefore not simply extend the same strong-opponent training. The evidence
supports investigating a curriculum or rehearsal mixture that preserves
peaceful hunting while gradually introducing active opponents, together with
state information that distinguishes opponent bombs and immediate competitive
danger. Any such change requires a new controlled experiment.

Repeat failures occur across the evaluation matrix and remain a shared
reproducibility limitation. They do not rescue this baseline because the
primary first-place and peaceful-retention gates independently fail.

## Decision

The trained checkpoints are not ready to be a good task 4 baseline, but
because of sparse time still select as completed Bomberman agent.
