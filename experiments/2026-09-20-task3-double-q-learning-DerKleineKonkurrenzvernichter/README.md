# DRAFT — Issue #220: single Q-learning vs Double Q-learning

> Status: completed-failed

## Question

Does Double Q-learning reduce maximization bias and improve elimination of a
peaceful opponent without materially reducing retained Task 2 performance?

The treatment follows van Hasselt (2010), *Double Q-learning*. It maintains two
independent estimators, randomly updates one on each transition, selects the
bootstrap action with that estimator, and evaluates it with the other estimator.

## Preregistered design

- Control: one-step single-table Q-learning.
- Candidate: one-step Double Q-learning with two persisted Q-tables.
- Action selection for the candidate uses the sum of both estimators.
- Both arms use `compact_opponent`, the frozen Task 2 prior, and otherwise
  identical hyperparameters.
- Five independent replicas per arm train for 10,000 classic rounds against one
  `peaceful_agent`.
- Each replica is evaluated on 20 paired seeds in peaceful, coin-collector,
  classic, coin-heaven, and loot-crate suites, followed by exact repeats.
- Planned budget: 100,000 training and 2,000 evaluation episodes (2,010 jobs).

The complete gates and seeds are fixed in `config.yaml` and the two run plans.

## Results

All 1,005 jobs per arm completed without a failed job. Double Q-learning did
not improve peaceful-opponent hunting and materially reduced retained Task 2
performance:

| Metric | Single Q | Double Q | Difference |
| --- | ---: | ---: | ---: |
| Peaceful opponents eliminated / episode | 0.080 | 0.050 | -0.030 |
| Peaceful collection fraction | 0.258 | 0.102 | -0.156 |
| Classic collection fraction | 0.223 | 0.139 | -0.084 |
| Coin Heaven collection fraction | 1.000 | 1.000 | 0.000 |
| Loot Crate collection fraction | 0.290 | 0.107 | -0.183 |
| Aggregate self-kill rate | 0.086 | 0.270 | +0.184 |

Only one of five matched replicas improved peaceful elimination; two became
worse and two tied. The matched-replica bootstrap estimate was -0.030 with a
95% interval from -0.120 to +0.050. The candidate therefore failed the primary
mean, replica-count, and confidence-bound gates. It also failed Classic and
Loot Crate retention and the aggregate safety gate. Coin Heaven retention was
preserved.

The registered repeat gate failed in both arms, primarily in evaluations with
the stochastic `coin_collector_agent`; this is not specific to Double Q-learning.
The latency gate was also marked failed because one control decision recorded a
241.9-second scheduling outlier while both arms were executed concurrently. The
ordinary per-replica p95 decision times remained well below 50 ms. Both failures
are retained in `result.json` rather than being silently excluded.

## Interpretation

The Double Q-learning candidate is rejected and the single-table control is
retained. In this sparse tabular setting, splitting the same 10,000 training
episodes across two estimators reduces the effective update density per table.
The expected reduction in maximization bias did not offset that loss of sample
efficiency: peaceful hunting declined, and Classic, Loot Crate, coin-collector,
and safety outcomes became worse.

This result does not show that Double Q-learning is generally inferior. It shows
that the registered two-table treatment at the fixed Issue #220 budget is not a
safe improvement for this agent. A larger budget could test whether the two
estimators merely need more experience, but it would be a new experiment rather
than a reinterpretation of this failed comparison.
