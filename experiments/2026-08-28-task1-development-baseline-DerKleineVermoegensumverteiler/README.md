# Task 1 Q-learning development baseline

## Metadata
- Issue: [#36](https://github.com/1BlauNitrox/mle-final-project/issues/36)
- Coordinating optimization issue:
  [#35](https://github.com/1BlauNitrox/mle-final-project/issues/35)
- Implementing PR:
  [#32](https://github.com/1BlauNitrox/mle-final-project/pull/32)
- Branch: 'experiment/35-task1-iterative-optimization'
- Commit:
- Agent implementation commit before merge: `6471fd27f2b8c673e2f38c099a9943227d85cbe2`
- PR-32 merge commit: `59f454c3cd84f03832b7c43f5b1b4adf053001d0`
- Framework revision: `0f55c1d`
- Agent: `DerKleineVermoegensumverteiler`
- Date: 2026-08-28
- Owner: LiliWestermann

## Hypothesis
Five training runs will provide a good development baseline for task 1 with the agent
`DerKleineVermoegensumverteiler` unchanged after PR #32 implementation.

We expect at least 0.80 aggregated mean coin collection fraction -> 4 out of 5
models reach 0.75 and all invalif action rates are below 0.01.

A negative or mixed result remains valid and will be retained.

## Setup
The complete machine-readable configuration is stored in `config.yaml`.

- Scenario: `coin-heaven`
- Opponents: none
- Training runs: 5
- Training episodes per run: 10,000
- Evaluation episodes: 40 per model
- Checkpoint selection: final checkpoint
- Training disabled during evaluation

## Independent Variables
Source of variation is the training run.

Each uses:
- a distinct training world seed
- a distinct agent seed
- a fresh model initialization

No design variable is deliberatly changed.

## Controlled Variables
The complete Agent implementation including Hyperparameters and Reward configuration,
aswell as the training and evaluation conditions remain unchanged.

## Metrics and success criterion
The metrics and thresholds are preregistered in Issue #36 and reproduced
in `config.yaml`. They must not be changed after the first scientific
training run begins.

## Results

### Model artifacts

| Run | World seed | Agent seed | Status | Episodes | Duration | Model size | Model SHA-256 |
|---:|---:|---:|---|---:|---:|---:|---|
| 1 | 11001 | 21001 | completed | 10000 | 281.73545541707426 | 6802 | 2e55b43db148c38977dd8a96d6498df2bbe0c0964b9320d45589843e8fbcc2f0 |
| 2 | 11002 | 21002 | completed | 10000 | 286.4858457497321 | 6815 | 868668e9f6d61075dff1bf9b5076347391aa5628967377722a22a163089c09bc |
| 3 | 11003 | 21003 | completed | 10000 | 274.3399446248077 | 6798 | eb7e9e1bd141bf5e29f8eb99e429eb7b54097e26cf31ae2c474002ef434a8f84 |
| 4 | 11004 | 21004 | completed | 10000 | 285.84537595836446 | 6809 | 890fe090a04243dde6f3bdf069e4fbdb694bd8196a75d4ecb8b5ca2c1d4fcce7 |
| 5 | 11005 | 21005 | completed | 10000 | 272.91439162474126 | 6807 | 771c9e05de8de6c3055988e2929b08406cd6bba61048c57ddbc91c03717dfe4a |

### Aggregate evaluation results

| Model | Episodes | Mean coins | Mean collection fraction | SD | Full clears | Zero-coin episodes | Invalid-action rate | WAIT | BOMB | Max decision time [ms] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| run-01 | 40 | 50.000 | 1.0000 | 0.0000 | 40 | 0 | 0.001146 | 0 | 0 | 0.867 |
| run-02 | 40 | 50.000 | 1.0000 | 0.0000 | 40 | 0 | 0.000184 | 4 | 0 | 0.319 |
| run-03 | 40 | 50.000 | 1.0000 | 0.0000 | 40 | 0 | 0.000962 | 0 | 0 | 0.878 |
| run-04 | 40 | 26.100 | 0.5220 | 0.3259 | 4 | 0 | 0.000134 | 0 | 0 | 0.758 |
| run-05 | 40 | 48.775 | 0.9755 | 0.1550 | 39 | 0 | 0.000733 | 6 | 0 | 1.511 |
| **Aggregate** | **200** | **44.975** | **0.8995** | **0.2478** | **163** | **0** | **0.000496** | **10** | **0** | **1.511** |

### Successcriterion evaluation

| Criterion | Result | Threshold | Passed |
|---|---:|---:|:---:|
| Completed evaluation episodes | 200 | 200 | yes |
| Aggregate mean coins collected | 44.975 / 50 | >= 40.000 / 50 | yes |
| Aggregate mean coin-collection fraction | 0.8995 | >= 0.80 | yes |
| Models with mean coins >= 37.5 | 4 | >= 4 | yes |
| Models with mean fraction >= 0.75 | 4 | >= 4 | yes |
| Aggregate invalid-action rate | 0.000496 | < 0.01 | yes |
| Maximum per-model invalid-action rate | 0.001146 | < 0.01 | yes |
| BOMB actions | 0 | 0 | yes |
| Maximum decision time | 1.511 ms | < 100 ms | yes |
| Full-clear episodes | 163 / 200 | descriptive only | n/a |
| Zero-coin episodes | 0 / 200 | descriptive only | n/a |

## Interpretation

The experiment produced all 200 developement evaluation episodes, which consist of five independently
trained models evaluated on the same 40 world seeds.

As shown in the prvious table all Thresholds have passed, so the primary performance hypothesis was
fully supported. This indicates that the unchanged baseline can learn effective Task 1 behaviour
with acceptable reproducibility across independent training runs.

Run 4 perfomed worse than the others, this difference is consistent with sensitivity to the registered
training-world and agent-seeds, because all controlled variables were held constant.

There are many full clear episodes, which shows that that the mean performance was not only produced
by partial coin collection. No zero-coin episodes occured, so all model colected at least some coins.

Run 2 & 5 had comparatively high `WAIT`counts, which may indicate that ties between action values or uncertainty
exist in frequently visited encoded states. It coul mean that the current rewards are not sufficient
to distinguish useful movement reliably in some states.

`BOMB`was selected zero times, so compliance with task 1 is confirmed.

## Decision and follow-up

In conclusion we see that the unchanged PR #32 implementation provides a strong baseline and is a
good starting point for experiments concerning optimization.

Its main limitation is the somethimes high WAIT count. So the next experiment should test one change
which could help with that.

No claim of final task 1 completion or tournament readiness is made.