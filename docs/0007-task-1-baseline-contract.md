# 0007 Task 1 Baseline Contract

## Status and authority

This document defines the normative contract for the Task 1 learned baseline
introduced by [issue #21](https://github.com/1BlauNitrox/mle-final-project/issues/21).
It specializes the general requirements in
[`0001-project-requirements.md`](0001-project-requirements.md) and the evaluation
rules in [`0004-experimentation-protocol.md`](0004-experimentation-protocol.md).
If a later document changes this contract, the change must be reviewed before
any affected experiment starts.

Meeting this contract completes only the visible-coin-navigation milestone. It
does not establish tournament readiness or complete the requirement to develop
and compare at least two learned models.

## Objective and scope

The baseline must learn to collect visible coins in the `coin-heaven` scenario
with a compact, feature-based tabular Q-learning policy. Its purpose is to
establish a reproducible end-to-end path from state features and rewards through
training, persistence, deterministic evaluation, and evidence reporting.

The baseline is limited to:

- navigating toward visible coins;
- avoiding walls and invalid movement;
- learning action values from transitions and rewards;
- saving and loading the learned Q-table; and
- deterministic, read-only evaluation with reproducible metrics.

The following capabilities are outside this milestone:

- placing or escaping bombs;
- destroying crates;
- modelling or hunting opponents;
- deep learning, self-play, and tournament optimization; and
- claims about capabilities outside `coin-heaven`.

The supplied rule-based agents and the intentionally weak team template are
comparison or scaffolding resources, not valid learned Task 1 solutions.

## Learning and action contract

The model family is feature-based tabular Q-learning. The state representation
must be compact enough for tabular learning and must be documented in the agent
card. Features may represent local geometry, valid movement, and relative coin
information, but must not deterministically encode an optimal action computed by
a planner or rule-based policy.

The complete and ordered Task 1 action set is:

```text
UP, RIGHT, DOWN, LEFT, WAIT
```

`BOMB` is forbidden during both training and evaluation for this milestone. The
agent implementation must therefore never select it; the evaluation report must
confirm a `BOMB` selection count of zero.

During evaluation, exploration is disabled. Any remaining tie-breaking must be
fixed or controlled by an explicitly recorded agent seed. Given the same model
artifact, world seed, agent seed, framework revision, and configuration, the
action sequence and episode result must be reproducible. Evaluation must not
update or rewrite the model artifact.

## Seed populations and configuration freeze

Training, development evaluation, and final evaluation use mutually disjoint
seed populations:

| Population | Required size | Permitted use |
| --- | ---: | --- |
| Training root seeds | 5 | Independently initialize the agent and framework RNGs for five training runs |
| Development world seeds | 50 | Debugging, model selection, and configuration decisions |
| Final held-out world seeds | 100 | One final assessment after the configuration is frozen |

The exact seed lists, framework revision, scenario configuration, opponent
slots, round limit, hyperparameters, and compute budget must be versioned in an
experiment record before the first training run. The final held-out list must
not overlap with any world seed used during training, debugging, development
evaluation, or manual inspection.

The two trained models must be evaluated for exactly one episode on each of the
same 100 final world seeds, giving 100 final rounds per model. The random
baseline likewise runs one episode per final world seed. The final set may be
opened only after features, rewards, hyperparameters, checkpoint selection,
evaluation code, and the random-baseline implementation are frozen. A failed
final result remains a failed result. Continuing development requires a new
pre-registered experiment and a new uncontaminated final seed set; the threshold
must not be changed after results are observed.

## Metrics

Raw metrics are recorded per episode before aggregation. Let
`available_coins` be the number of coins present at the start of the episode,
`coins_collected` the number collected by the evaluated agent, `episode_steps`
the total number of environment steps, `survival_steps` the steps for which the
agent remained alive, and `attempted_actions` the number of actions returned by
the agent. These counts remain distinct when an action response is skipped or
the episode continues after the agent dies.

| Metric | Definition and role |
| --- | --- |
| Coin collection fraction | `coins_collected / available_coins`; primary metric |
| Coins collected | Raw episode count; required secondary metric |
| Full-clear success | Whether all initially available coins were collected before the 400-step limit |
| Steps per collected coin | `sum(survival_steps) / sum(coins_collected)` across the reported group; zero-coin episodes add their survival steps but no fictitious coin, and the result is unavailable when the group collects no coins |
| Coins per 100 steps | `100 * sum(coins_collected) / sum(survival_steps)`; aggregate efficiency metric that is zero for a non-empty all-zero-coin result |
| Episode length | Executed steps until termination or the 400-step limit |
| Invalid-action rate | Invalid actions divided by `attempted_actions` |
| Environment score | Framework score; required context metric, not the completion target |
| Cumulative shaped reward | Diagnostic only; comparable across variants only when the reward definition is identical |
| Decision time | Median, 95th percentile, and maximum wall-clock duration of all evaluation-time `act()` calls |

The evaluation report must include per-model results, the aggregate across the
two trained models, and variation across both training runs and world seeds.
It must also report the number and rate of zero-coin episodes and the number of
full-clear episodes rather than hiding them inside aggregate efficiency values.

## Random-policy baseline

The comparison baseline is a seeded, non-learning, state-independent policy
that samples uniformly from the five Task 1 actions. It must not mask invalid
actions because doing so would add environment knowledge absent from a uniform
random baseline. It uses its own recorded random seed and is evaluated on the
same final world seeds under the same scenario, round limit, framework revision,
and runtime instrumentation as the learned agents.

The learned-policy and random-policy episode results are paired by world seed.
The comparison must report the mean paired difference in coin collection
fraction and a two-stage 95% percentile bootstrap confidence interval that
resamples training runs and, within each run, paired world seeds. The experiment
record must fix the implementation and use 10,000 bootstrap resamples before
evaluation.

## Completion rule

The Task 1 baseline is complete only if every gate below passes on the final
held-out evaluation:

1. Two independently trained model artifacts are evaluated on all 100 final
   world seeds.
2. The mean coin collection fraction across all models and final episodes is at
   least `0.80`.
3. The learned policy exceeds the uniform-random baseline by at least `0.20`
   absolute coin-collection-fraction points, and the lower bound of the paired
   95% bootstrap confidence interval is greater than `0.00`.
4. The aggregate invalid-action rate is below `0.01`, and every individual model
   is below `0.01`.
5. `BOMB` is selected exactly zero times.
6. Evaluation is deterministic and leaves every model artifact byte-for-byte
   unchanged.
7. On the documented reference machine, evaluation uses one CPU thread and no
   multiprocessing; the 95th-percentile decision time is below `50 ms` and the
   maximum decision time is below `100 ms`. All `act()` calls are included.

These values are intended as an achievable first learned milestone with enough
margin to reveal unreliable training runs. They must be revisited only through
a prospective contract change, never in response to final held-out results.

## Required evidence

The completion evidence must be a compact experiment record following
[`0004-experimentation-protocol.md`](0004-experimentation-protocol.md). It must
identify the issue, commit, exact configuration, all seed lists, two model
artifacts and checksums, random-baseline configuration, per-episode data,
aggregate metrics, uncertainty calculation, runtime environment, and test
results. Raw logs and temporary checkpoints remain outside Git.

This contract contains no performance claim and authorizes no training or
evaluation by itself. Lifecycle documentation is tracked in
[issue #22](https://github.com/1BlauNitrox/mle-final-project/issues/22), and the
self-contained baseline package is tracked in
[issue #23](https://github.com/1BlauNitrox/mle-final-project/issues/23).
