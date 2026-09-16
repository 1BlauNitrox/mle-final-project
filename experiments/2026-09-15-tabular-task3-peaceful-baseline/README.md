# Tabular Task 3 peaceful-opponent baseline

> completed failed baseline

## Metadata

- Issue: #186
- Agent: `DerKleineKonkurrenzvernichter`
- Date: 2026-09-15
- Parent: frozen interim Task 2 r2, SHA-256 `93470f92082597b1848f2fa65265c7288dbb1b50ac90370e5747e085e775ea3e`
- Run plan: `training/run_plans/issue186-tabular-task3-peaceful-baseline.yaml`

## Hypothesis

Adding a publich opponent description while retaining task 2 capabilities 
make a working task 3 baseline.

## Protocol

Five independent replicas are trained for 10,000 Classic episodes against one
`peaceful_agent`.

The only new state values are the route direction to the nearest
opponent, route-distance bin, and current blast-line opportunity. Additionaly
added was the reward `KILLED_OPPONEND`with `+5`. Nothing else is changed.

Evaluation uses 20 fixed development seed pairs per
replica for peaceful, coin-collector, Classic, Coin Heaven and Loot Crate suites,
plus exact repeats.

## Results

The registered run completed all 5 training replicas and 1,000 evaluation
episodes (500 primary plus 500 repeats). Mean peaceful-opponent elimination
was `0.08` per episode, below the registered `0.20` target. Four replicas did
achieve at least one elimination, but no replica reached the target rate; the
replica means were `0.05`, `0.05`, `0.10`, `0.15`, and `0.05`.

Earlier-task retention gates passed. Mean collection fraction was `0.207` in
Classic, `1.000` in Coin Heaven, and `0.319` in Loot Crate. The aggregate
self-kill rate across all primary suites was `0.09`, below the `0.15` limit.
The maximum primary p95 decision time was `7.02 ms`. One primary decision-time
maximum was `159.82 ms`, so the registered `100 ms` maximum gate failed.

Exact repeats matched for 392 of 500 primary/repeat pairs. All solo Classic,
Coin Heaven, and Loot Crate pairs matched. The opponent suites did not fully
repeat: 88/100 peaceful pairs and 4/100 coin-collector pairs matched. Therefore
the registered repeatability gate failed and opponent evaluation needs a
separate seed-control investigation before deterministic opponent comparisons
are claimed.

The durable evidence is in `evidence.csv`, `summary.csv`, and `result.json`.
Figures are in `figures/`. Recompute them with:

    python -m training.analyze_issue186_task3_baseline
    MPLBACKEND=Agg python -m training.plot_issue186_task3_baseline

## Interpretation

The appended opponent state and direct `KILLED_OPPONENT` reward produced a
measurable but weak hunting signal. Because all five replicas remained below
the preregistered mean-elimination threshold, this experiment does not support
claiming Task 3 capability. It does support the narrower conclusion that the
strict Task 2 prior migration preserved the earlier collection behavior while
allowing occasional opponent eliminations.

The results suggest that nearest-opponent direction, a coarse distance bin,
and a current blast-line flag are insufficient on their own for reliable bomb
interception. Maybe in a follow up we could try is adding more information,
for example a possible opponent escape route, will make  better progress.

## Decision

Reject this trained baseline as the Task 3 candidate, but keep the negative
results for documentation, do follow up experiments to optimize the
performance.
