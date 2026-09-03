# Task 1 DQN movement-coin reward shaping

> **Status: completed. Result is negative; the hypothesis is not supported.**
> Everything above the Results section is the registration as it stood before
> the first training run and has not been changed. The registered stopping rule
> applies: no third Task 1 tuning experiment follows.

## Metadata

- Issue: [#58](https://github.com/1BlauNitrox/mle-final-project/issues/58)
- Owner: 1BlauNitrox / Reviewer: Waffelmanufaktur
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

All five models completed 10,000 training episodes and were evaluated on
development world seeds `31001`-`31040`, giving 200 primary episodes, each
repeated once for the determinism pass.

Run 5 required one retry after an environment fault; see the amendment section
below. Its retry used unchanged agent source, so all five models are
attributable to fingerprint `b40b4b09...`.

### Per-model evaluation

| Model | Mean fraction | SD | Full clears | Zero-coin | Invalid rate | WAIT | BOMB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| run-01 | 0.7710 | 0.2937 | 16 | 0 | 0.461683 | 4 | 0 |
| run-02 | 0.8165 | 0.2606 | 12 | 0 | 0.000315 | 6 | 0 |
| run-03 | 0.9210 | 0.1751 | 28 | 0 | 0.307856 | 1218 | 0 |
| run-04 | 0.9150 | 0.2206 | 30 | 0 | 0.000764 | 4 | 0 |
| run-05 | 0.6535 | 0.3372 | 6 | 1 | 0.116915 | 1444 | 0 |
| **Aggregate** | **0.8154** | **0.2792** | **92** | **1** | **0.175804** | **2676** | **0** |

### Registered criteria

| # | Criterion | Threshold | Observed | Result |
| ---: | --- | --- | --- | :---: |
| 1 | Models at fraction `>= 0.75` | `>= 4 of 5` | 4 | Pass |
| 2 | Invalid-action rate | `< 0.01` aggregate and per model | `0.1758` aggregate, `0.4617` worst | **Fail** |
| 3 | Aggregate fraction | `>= 0.80` | `0.8154` | Pass |
| 4 | `BOMB` selections | `= 0` | 0 | Pass |
| 5 | Deterministic and immutable | required | both verified | Pass |
| 6 | Decision time | p95 `< 50 ms`, max `< 100 ms` | `0.464` / `68.99` ms | Pass |

**The primary criterion requires parts 1 and 2 together, so it fails.**

### Paired comparison with issue #41

Computed from the committed per-seed evidence of both series, pairing run index
and world seed, with 10,000 two-stage resamples and registered resampler seed
`58`:

| Quantity | Value |
| --- | ---: |
| Mean paired difference (#58 − #41) | `-0.0180` |
| 95% percentile interval | `[-0.2471, +0.2414]` |
| Registered non-inferiority margin | `> -0.02` |
| Result | **Fail** |

### Against the baseline

| Metric | #41 baseline | #58 shaping | Direction |
| --- | ---: | ---: | :---: |
| Aggregate fraction | 0.8334 | 0.8154 | worse |
| Models at `>= 0.75` | 3 / 5 | 4 / 5 | better |
| Aggregate invalid rate | 0.164158 | 0.175804 | worse |
| Worst-model invalid rate | 0.608521 | 0.461683 | better |
| Full clears | 125 / 200 | 92 / 200 | worse |
| Zero-coin episodes | 2 | 1 | better |
| `WAIT` actions | 6505 | 2676 | better |
| Steps per coin | 5.548 | 6.753 | worse |

## Interpretation

**The hypothesis is not supported.** Movement-coin shaping did not remove the
run-to-run divergence, and the registered criterion fails on the invalid-action
gate by more than an order of magnitude.

The single most informative number is the confidence interval,
`[-0.2471, +0.2414]`. It is roughly 24 points wide in both directions around a
point estimate of `-0.018`. With five runs and this much between-run variance,
the experiment cannot distinguish the shaped configuration from the baseline at
all. The instability is not a secondary nuisance around a real effect; it is
larger than any effect the shaping might have. That, rather than the sign of the
point estimate, is the finding.

The shaping did change the *character* of the failures without reducing their
size. `WAIT` actions fell from 6505 to 2676 and the worst-model invalid rate
fell from 0.61 to 0.46, which is consistent with the mechanism the hypothesis
proposed: a dense directional signal does make standing still less attractive.
But the models that previously failed by waiting now fail by attempting blocked
moves, and two models that were fine before (run-01 at a 0.46 invalid rate,
run-03 at 0.31) are now among the worst. Full clears fell from 125 to 92.

### The invalid-action rate is a deadlock count, not an error rate

Reporting `0.1758` as an aggregate rate misrepresents the mechanism. The
per-episode distribution is sharply bimodal:

| Invalid-action rate | Episodes |
| --- | ---: |
| `< 1%` | 171 |
| `1-20%` | **0** |
| `20-80%` | 12 |
| `> 80%` | 17 |

Nothing lands between 1% and 20%. An episode is either clean or catastrophic.

The catastrophic case is a deadlock. In run-01 on world seed `31028`, 399 of 400
actions were `DOWN` and 399 were rejected: the agent faced a wall, the greedy
policy selected the same blocked direction, the environment refused it, the
state was unchanged, and therefore the next argmax was identical. The episode
ended at the step limit with 1 coin.

The same model collects 42 of 50 coins on seed `31001`, and run-04 clears all 50
in 123 steps with zero invalid actions. So this is not a model that learned
badly; it is a deterministic policy with an absorbing failure state that some
starting positions lead into and others do not.

Epsilon at `0.1` guarantees escape during training, which is why training curves
never show it. Evaluation has no such escape.

This also means the aggregate coin fraction understates the working policy: 171
of 200 episodes are clean, and the aggregate is dragged down by 29 episodes in
which the agent is frozen rather than playing badly.

### The training-versus-evaluation gap is the most important observation

Training-time coin collection over the last 500 episodes was 0.990, 0.989,
1.000, 0.992 and — for the retried run 5 — similarly high. Deterministic
evaluation of the same checkpoints gives 0.7710, 0.8165, 0.9210, 0.9150 and
0.6535.

The gap is systematic, not noise. Training runs with epsilon floored at `0.1`,
so roughly one action in ten is random and the pure-greedy policy is never
exercised during training. Those forced random actions appear to be doing real
work: they break the agent out of states where the greedy policy would
repeatedly select a blocked direction. Evaluation removes that escape hatch, and
the invalid-action rate explodes.

This reframes the whole Task 1 DQN problem. The instability that #41 and #58
both measured is not primarily a *learning* failure — the networks clearly learn
to collect coins. It is a failure of the greedy policy in states where the
argmax points into a wall. The eight-feature representation does contain
`free_up`/`free_right`/`free_down`/`free_left`, so the information needed to
avoid this is present; the network simply does not always use it.

That points at legal-action masking at evaluation time as the obvious next
intervention, which is precisely what #41's own decision text proposed
("changing one factor only (for example legal-action masking)"). This experiment
tested a different factor first and can now say the shaping factor was the wrong
one to try.

### Secondary observations

Maximum decision time rose to `68.99 ms` from `27.60 ms` in #41. It still passes
the `100 ms` gate and the p95 is `0.464 ms`, so this is an outlier rather than a
trend, and the two series ran on different processors. It is recorded rather
than interpreted.

One zero-coin episode occurred, in run-05. The aggregate SD of `0.2792` is
essentially unchanged from #41's `0.2875`, which is another way of saying the
shaping did not stabilise anything.

## Decision

**Rejected.** The registered primary criterion fails, and the registered
non-inferiority comparison fails. Movement-coin shaping is not adopted for the
Task 1 DQN.

The result is retained in full. The shaping implementation stays in the agent so
that this record remains reproducible, and because the tabular agent uses the
identical definition, which #53 needs for a like-for-like comparison. It is not
claimed to be an improvement.

Applying the registered stopping rule, which was fixed before any result was
seen:

> **Criterion fails:** do not run a third Task 1 tuning experiment. Amend #42
> prospectively to permit a documented neutral selection rule such as the
> median-performing run, freeze on that basis, retain the instability as a
> negative result, and proceed to #43 regardless.

No third Task 1 tuning experiment will be run. The agent-code deadline is
21 September 2026 with Tasks 2 to 4 outstanding, and two experiments have now
shown that this factor is not where the problem lies.

## Follow-up

1. Amend #42 prospectively with a neutral selection rule. On this evidence the
   median-performing model is **run-02** at `0.8165`, which also has a clean
   invalid-action rate of `0.000315`. The rule must be stated before the
   artifact is chosen, not justified by that convenience.
2. Freeze the selected artifact under #42 and record that its series failed the
   registered criterion, so no later reader mistakes the freeze for a pass.
3. Proceed to #43 and the Task 2 successor.
4. Register legal-action masking as a Task 2 concern rather than a third Task 1
   experiment. The evidence above says the greedy policy walks into walls; that
   defect will follow the agent into Task 2, where a blocked move next to a live
   bomb is fatal rather than merely wasteful.
5. Merge the checkpoint-replace retry fix before Task 2 training begins.

## Amendment: run 5 failed once and was retried

Recorded on issue #58 before any evaluation ran, following the #41 precedent.

Run 5 terminated at episode 3950 / 10000 with
`PermissionError: [WinError 5]` when Windows denied the atomic checkpoint
replacement — the same environment fault that ended #41's run 5 at episode 616.
`save_checkpoint` performs an atomic replace after every episode, so a five-run
series makes roughly 50,000 attempts and one transient antivirus or indexer lock
ends the run.

The failed attempt's partial checkpoint is retained as
`run-05-failed-checkpoint.pt` and the series manifest records `attempts: 2`.

The retry repeated only run 5, with the same registered seed pair
`15005` / `25005`, inside the same series directory. It verified that the
working tree still matched the fingerprint the series recorded when it started,
rather than recomputing an expected value, so all five models remain
attributable to one agent revision.

The root-cause fix was deliberately **not** applied to this experiment.
`persistence.py` lives in the agent directory, so applying it would have changed
the fingerprint and run 5 would have been trained from different source than
runs 1 to 4 — a confound in an experiment measuring run-to-run stability. It
lands separately, before Task 2.

## Deviation: evaluation ran on a later agent revision than training

Training used fingerprint `b40b4b09...`. Evaluation used
`8e5f20b8...`. The only difference is the `BOMBERMAN_EVALUATION_CHECKPOINT`
indirection PR #49 added to `callbacks._setup_evaluation_policy`, plus the agent
card. It resolves which checkpoint file the frozen policy loads and is required
for the staged evaluation artifact; it touches neither the training path nor the
network, features, rewards or action selection.

Evaluating on the training revision would have reintroduced the worktree
dirtying that PR #49 fixed, which would have stamped `git_dirty: true` on all
400 evaluation records. The deviation is recorded rather than avoided.

The analysis code also advanced between preregistration and evaluation. That the
newer code reproduces the older numbers exactly is verifiable: running
`dqn_task1_evidence verify` on the #41 record rebuilds its `summary.csv`,
`result.json` and all three figures byte-for-byte.

## Evidence

`evidence/` carries the 400 per-episode rows, all raw decision times, the
compressed training episodes, and a manifest with per-file checksums.
`dqn_task1_evidence verify --issue 58` rebuilds `summary.csv`, `result.json` and
every figure from that evidence and requires byte equality. Both failure paths
were exercised: altering a committed table reports
`Committed output differs`, and altering the evidence reports
`Evidence checksum mismatch`.
