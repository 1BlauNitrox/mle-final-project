# Task 1 DQN movement-coin reward shaping

> **Status: preregistered. No training run has started.**
> The Results, Interpretation, Decision, and Follow-up sections are intentionally
> empty. They must be filled only after the registered runs complete, and nothing
> above them may change once the first training run begins.

## Metadata

- Issue: [#58](https://github.com/1BlauNitrox/mle-final-project/issues/58)
- Author: unassigned — an owner and a non-author reviewer must be assigned
  before the first training run
- Date: 2026-09-02
- Branch: `experiment/58-dqn-task1-movement-shaping`
- Baseline experiment: `2026-09-01-dqn-task1-development-baseline` (issue #41)
- Baseline training commit: `e7e8b52f50b2acb46ccad04905d76c6948304e21`
- Candidate commit: pending — the shaping implementation is not yet written
- Repository commit at preregistration: `dbe259721409092448e8594cbeaaca469c4f9835`
- Framework revision: `0f55c1d` (imported `ukoethe/bomberman_rl`)
- Agent source SHA-256 at preregistration:
  `74407c19bf880e8f0a174427b559456c0bed2ec4d1e8ef20dd619402a9fe0fba`

## Research question

Does a dense per-step signal distinguishing movement toward a coin from movement
away from one remove the run-to-run training divergence that prevented the Task 1
DQN from being frozen in #41?

## Hypothesis

Adding the movement-coin shaping events already validated on the tabular agent to
`DagobertDuckDQN` will remove the divergence observed in #41, so that at least
four of five independently trained models reach a mean coin collection fraction
of `0.75` and the aggregate invalid-action rate falls below `0.01`.

The proposed mechanism is reward density. The DQN's only reward signals are
`COIN_COLLECTED` (`+10.0`), `INVALID_ACTION` (`-0.5`) and `WAITED` (`-0.1`).
Between coins the agent receives nothing, so early in training its value
estimates carry no directional information at all. That leaves room for two
degenerate policies, and #41 produced one of each:

- **run-03 collapsed to waiting.** It selected `WAIT` 6,487 times and reached a
  `0.4830` collection fraction. Waiting costs `-0.1` while an invalid move costs
  `-0.5`, so for a network that has not yet learned where coins are, standing
  still is the cheapest action available.
- **run-04 collapsed to blocked movement.** It reached a `0.6085` invalid-action
  rate and a `0.7395` collection fraction, repeatedly attempting moves into
  walls.

A per-step signal that separates progress from non-progress makes both
degenerate policies strictly worse than moving toward a coin, and it does so from
the first episode rather than only after the first accidental coin pickup.

This is a port of an already-validated result rather than a new idea: the same
change produced the largest single improvement in the tabular optimization
campaign (#35).

The hypothesis is falsifiable. If shaping does not change the number of divergent
runs, or if it improves stability while reducing aggregate collection below
`0.80`, the hypothesis is not supported and that result is retained.

## Baseline

The five `DagobertDuckDQN` artifacts trained under #41, evaluated on development
world seeds `31001`-`31040`.

| Quantity | Baseline value |
| --- | ---: |
| Aggregate coin collection fraction | `0.8334` |
| Between-episode SD | `0.2875` |
| Models reaching `>= 0.75` | `3 / 5` |
| Aggregate invalid-action rate | `0.164158` |
| Maximum per-model invalid-action rate | `0.608521` |
| Full clears | `125 / 200` |
| `WAIT` actions | `6505` |
| `BOMB` actions | `0` |

Per-model baseline values are in
`../2026-09-01-dqn-task1-development-baseline/summary.csv`.

## Variant

Exactly one thing changes: the reward mapping of
`agent_code/DagobertDuckDQN`.

| Event | Baseline | Variant |
| --- | ---: | ---: |
| `COIN_COLLECTED` | `+10.0` | `+10.0` |
| `INVALID_ACTION` | `-0.5` | `-0.5` |
| `WAITED` | `-0.1` | `-0.1` |
| `MOVED_TOWARDS_COIN` | absent | `+0.1` |
| `MOVED_AWAY_FROM_COIN` | absent | `-0.1` |

The two events are derived in `train.py` from the observed transition, mirroring
`_coin_movement_event` in `DerKleineVermoegensumverteiler` after PR #37. For a
movement action, the minimum Manhattan distance from the old position to any coin
visible in the **old** state is compared against the same distance from the new
position:

- distance decreased: emit `MOVED_TOWARDS_COIN`;
- distance increased: emit `MOVED_AWAY_FROM_COIN`;
- distance unchanged, no visible coin, or a non-movement action: emit nothing.

Using the old state's coin list for both measurements keeps the event well
defined on the step where a coin is collected and disappears.

The definition and both magnitudes are copied from the tabular agent
deliberately. Beyond reusing a validated result, it puts both model families on
the same reward footing, which #53 requires for a fair head-to-head comparison.
After PR #37 the tabular agent has shaping and the DQN does not, so #53 would
otherwise compare two agents trained under materially different conditions.

### Known theoretical limitation

This shaping is **not** potential-based. Theorem 9.19 of the lecture (eq. 9.50)
and Ng, Harada & Russell (1999), cited in footnote 2 of the project handout,
establish that only shaping of the form

```text
r~(t+1) = r(t+1) + gamma * Psi(s(t+1)) - Psi(s(t))
```

is guaranteed to leave the optimal policy unchanged. A flat `+/-0.1` on distance
change approximates that form with `Psi(s) = -distance`, but the missing `gamma`
factor breaks the telescoping cancellation, so policy invariance is not
guaranteed and some bias toward greedy short-range coin seeking is possible.

This is accepted deliberately here, for two reasons. First, the experiment's
purpose is to port the tabular result with exactly one variable changed;
"correcting" the form at the same time would change two things and make the
comparison uninterpretable. Second, the symmetric penalty removes the
back-and-forth oscillation exploit the handout warns about, because returning to
a tile cancels the reward earned by leaving it.

A potential-based variant carrying the correct `gamma` factor is a worthwhile
follow-up and must be registered as its own experiment rather than folded into
this one.

## Controlled variables

Everything below is held identical to #41 and must not be altered once training
starts:

- network `8 -> 64 -> 64 -> 5`, ReLU between hidden layers, no output activation;
- learning rate `0.001`, discount factor `0.9`, gradient clip norm `10.0`;
- batch size `64`, replay capacity `10000`, replay warm-up `256`, target update
  interval `250`;
- Adam optimizer, Huber (smooth L1) loss;
- epsilon `1.0`, per-episode decay `0.99`, floor `0.1`, evaluation epsilon `0.0`;
- the eight-value feature vector, its ordering, and its normalization
  (`coin_distance_bin / 3`);
- action order `UP, RIGHT, DOWN, LEFT, WAIT`, with `BOMB` excluded by
  construction;
- scenario `coin-heaven`, no opponents, 400-step episode limit, 50 coins;
- five fresh models, 10,000 episodes each, final-checkpoint selection;
- evaluation with training disabled, one PyTorch CPU thread, no multiprocessing;
- the evaluation and analysis code paths, apart from the defect fix listed under
  Blockers.

## Training protocol

Five independent runs, each from a fresh initialization, 10,000 `coin-heaven`
episodes, no opponents, final checkpoint retained.

| Run | World seed | Agent seed |
| ---: | ---: | ---: |
| 1 | 15001 | 25001 |
| 2 | 15002 | 25002 |
| 3 | 15003 | 25003 |
| 4 | 15004 | 25004 |
| 5 | 15005 | 25005 |

The `15xxx`/`25xxx` block was chosen to avoid every seed recorded in the
repository. `11xxx`/`21xxx` belong to the tabular campaign, `12xxx`/`22xxx` to
the #41 DQN baseline, `13xxx`/`23xxx` are in use by the slower-epsilon
experiment, and `14xxx`/`24xxx` are left free for the slower-learning experiment.

Checkpoint selection is mechanical: the final checkpoint of each run. No
evaluation-based selection is permitted.

A failed run is retained and documented rather than silently retried, following
the precedent set by the retained run-5 failure in #41.

## Evaluation protocol

- Scenario `coin-heaven`, no opponents, training disabled.
- Development world seeds `31001`-`31040`, one episode per model per seed, giving
  200 primary episodes.
- These are deliberately the **same** seeds as #41, so every episode pairs with a
  baseline episode by run index and world seed.
- All 200 episodes are repeated as a determinism pass.
- Confirmation seeds `31041`-`31050` and the final held-out population remain
  unopened by this experiment.
- Artifact SHA-256 is recorded before and after evaluation; every artifact must
  be byte-for-byte unchanged.
- CPU only, one thread, no multiprocessing, below 8 GB RAM.

The machine used at preregistration is recorded in `config.yaml`. The primary
metric is hardware-independent given fixed seeds, but the decision-time gates are
machine-specific and must be reported together with the machine that produced
them. The #41 baseline was measured on a different processor, so latency figures
are not directly comparable across the two experiments.

## Metrics and success criterion

Primary metric: mean coin collection fraction per model, and the number of models
reaching `0.75`.

The success criterion is exactly the pair of gates that #41 failed, because
passing them is what unblocks the #42 freeze:

1. at least four of five models reach a mean coin collection fraction
   `>= 0.75`; **and**
2. the aggregate invalid-action rate is `< 0.01` and every individual model is
   `< 0.01`.

Guardrails, so that stability is not bought with performance:

3. aggregate mean coin collection fraction `>= 0.80`;
4. `BOMB` selected exactly zero times;
5. evaluation deterministic and all five artifacts byte-for-byte unchanged;
6. decision time p95 `< 50 ms` and maximum `< 100 ms`.

Comparison against #41, reported whether or not the criterion passes: the mean
paired difference in coin collection fraction, with a two-stage 95% percentile
bootstrap confidence interval using 10,000 resamples, resampling run indices and
then paired world seeds within each run.

Unlike the tabular comparison attempted in #41, this interval **is** computable,
because both experiments use the same 40 evaluation seeds and the same five run
indices. The per-seed rows produced here must be committed so the interval stays
reproducible from the repository alone.

Descriptive measures, reported but not gated: full-clear count, zero-coin count,
`WAIT` totals, steps per coin, coins per 100 steps, and per-run learning curves.

## Stopping rule

Recorded before any result is seen, so the decision cannot be made after the
fact. **This is the last Task 1 tuning experiment for the DQN either way.**

- **Criterion passes:** proceed to #42, freeze mechanically from this
  experiment's evidence, and unblock #43.
- **Criterion fails:** do not run a third Task 1 tuning experiment. Amend #42
  prospectively to permit a documented neutral selection rule such as the
  median-performing run, freeze on that basis, preserve the instability as a
  retained negative result, and proceed to #43 regardless.

The project has a 21 September agent-code deadline with Tasks 2 to 4 still
outstanding. Further Task 1 optimization is not an acceptable use of the
remaining budget.

## Blockers

The `0 * inf` aggregation defect in `training/analyze_dqn_task1_baseline.py`
must be fixed before this experiment runs. `steps_per_coin` becomes `math.inf`
for a zero-coin model, and `_aggregate_summaries` multiplies it by a zero coin
count, producing `NaN` in the aggregate and an invalid `NaN` token in
`result.json`. Since this experiment exists precisely because some runs diverge,
a zero-coin run is a realistic outcome and must not silently corrupt the record.

## Results

Not yet available. This experiment is preregistered and has not been run.

## Interpretation

Not yet available.

## Decision

Not yet available.

## Follow-up

Not yet available. The follow-up issue is determined by the stopping rule above.
