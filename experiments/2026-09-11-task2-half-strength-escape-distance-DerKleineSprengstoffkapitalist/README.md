# Half-strength escape-distance potential shaping

> Status: completed; candidate rejected

## Metadata

- Issue: #139
- Agent: `DerKleineSprengstoffkapitalist`
- Owner: LiliWestermann
- Date: 2026-09-11
- Registration base: `974fa7c`
- Implementation commit: `b46a54984d25cd4e15096eebbfa876614257590b`
- Experiment commit: `698c031893528dfad850b82251583fcb522a47a2`
- Analysis commit: `2c59feca1cdd9309900a3ce1b9ff5ce8116283d9`
- Approval: intermediate peer approval explicitly waived by the owner

## Research question and hypothesis

Does halving the time-aware escape-distance potential recover Classic
collection performance relative to the full-strength treatment while retaining
a lower self-kill rate than the Issue #135 no-shaping reference?

Issue #135 reduced Classic self-kills from `0.055` to `0.015`, but also reduced
collection from `0.132` to `0.048`. The hypothesis is that half strength keeps
the useful safety signal without dominating task progress as strongly.

## Treatments

- Control: `escape_distance` with potentials `0, -1, -2, -3, -5`.
- Candidate: `escape_distance_half` with potentials
  `0, -0.5, -1, -1.5, -2.5`.

Both use `F(s,s') = 0.9 * Phi(s') - Phi(s)`. The scale is the only treatment
difference. It changes learning rewards, not evaluation-time action rules.

## Controlled protocol

Both arms use compact state, zero-initialized one-step Q-learning, learning
rate `0.05`, discount factor `0.9`, epsilon `1.0 * 0.99` down to `0.1`, no
action mask, no useful-bomb bonus, identical paired seeds and final-checkpoint
selection. Five replicas each train for 2,000 Coin Heaven, 2,000 Loot Crate
and 6,000 Classic episodes. Each final model is evaluated on 40 fresh paired
development seeds per scenario and repeated deterministically.

## Decision rule

Accept half strength only if all registered criteria pass:

- mean Classic collection exceeds full strength;
- the paired 95% bootstrap CI lower bound for that difference is above zero;
- candidate Classic self-kill rate remains below the Issue #135 no-shaping
  reference of `0.055`;
- at least four of five candidate replicas remain below `0.055`;
- Coin Heaven collection decreases by no more than `0.05`;
- candidate/control mean visits per state is at least `0.90`;
- deterministic repeats match;
- decision-time p95 is below `50 ms` and maximum below `100 ms`.

The historical no-shaping value is only a preregistered safety guard. The new
scientific paired contrast is full strength versus half strength.

## Execution

```bash
python -m training.run_plan \
  training/run_plans/issue139-compact-task2-full-strength.yaml

python -m training.run_plan \
  training/run_plans/issue139-compact-task2-half-strength.yaml
```

The owner explicitly authorized immediate execution without the intermediate
approval pause. This waiver does not alter treatments, seeds, budgets, metrics
or criteria.

## Results

Both plans completed all 1,215 jobs without retries. The retained evidence
contains 1,200 primary and 1,200 deterministic-repeat evaluation episodes.

### Performance and safety

| Scenario | Metric | Full strength | Half strength |
| --- | --- | ---: | ---: |
| Classic | Collection fraction | 0.0472 | 0.1344 |
| Classic | Self-kill rate | 0.010 | 0.070 |
| Coin Heaven | Collection fraction | 1.000 | 1.000 |
| Coin Heaven | Self-kill rate | 0.000 | 0.000 |
| Loot Crate | Collection fraction | 0.1005 | 0.1633 |
| Loot Crate | Self-kill rate | 0.090 | 0.195 |

The paired half-minus-full Classic collection difference was `+0.0872`, with
a 95% nested-bootstrap interval of `[+0.0294, +0.1439]`. Both registered
collection-improvement gates therefore passed.

Half strength did not retain the registered historical safety guard. Its
Classic self-kill rate was `0.070`, above the Issue #135 no-shaping reference
of `0.055`. Replica rates were `0.050`, `0.075`, `0.050`, `0.125` and `0.050`,
so only three of five were strictly below the historical limit.

### Learning efficiency and latency

| Diagnostic | Full strength | Half strength |
| --- | ---: | ---: |
| Mean materialized states | 1,072.2 | 1,087.8 |
| Mean visits per state | 798.5 | 737.8 |
| Mean singleton-state fraction | 0.0710 | 0.0601 |
| Evaluation unseen-state rate | 0.0022% | 0.0083% |

The visits-per-state ratio was `0.924`, above the registered `0.90` minimum.
All primary and repeat outcomes matched deterministically. The worst primary
p95 was `1.71 ms`; the largest primary decision was `10.70 ms`. Both latency
gates passed.

Committed evidence and products:

- `evidence.csv`: 1,200 treatment/model/scenario/seed observations plus repeat
  equality and latency evidence;
- `training_diagnostics.csv`: final learning diagnostics for all ten models;
- `summary.csv`: per-replica aggregates;
- `result.json`: bootstrap result, criteria and diagnostics;
- `figures/performance_and_safety.png`;
- `figures/learning_efficiency.png`.

Reproduce the analysis and figures with:

```bash
python -m training.analyze_issue139_half_strength \
  --verify-from-evidence
python -m training.plot_issue139_half_strength
```

## Interpretation and decision

Reject half-strength shaping as the new default because both registered safety
criteria failed, despite a statistically supported collection improvement.

The result supports a genuine strength trade-off. Halving the potential
restored Classic collection to approximately the historical no-shaping level
from Issue #135 and improved Loot Crate collection, but also removed the safety
advantage and more than doubled Loot Crate self-kills relative to full
strength. This is not evidence that half strength is globally worse: it is
evidence that it does not satisfy the prospectively required combination of
collection and safety.

The fresh full-strength control also differed from Issue #135's full-strength
result, which shows that point estimates remain sensitive to trained replicas
and evaluation populations. The paired within-experiment collection result is
therefore the defensible causal comparison; the Issue #135 no-shaping value is
used only for the preregistered absolute guard.

A further scalar interpolation such as `0.75` could continue the sweep, but a
more informative follow-up would separate the strong penalty for having no
complete escape route from ordinary distance-progress shaping. That isolates
catastrophic bomb-placement risk without suppressing all productive movement.

## AI assistance

OpenAI Codex assisted with implementation, tests, protocol registration,
execution, analysis, plotting and documentation. AI output is not experimental
evidence; all claims must be computed from retained outputs and reviewed by the
owner and a non-author reviewer.
