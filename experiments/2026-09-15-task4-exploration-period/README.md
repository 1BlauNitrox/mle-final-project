# Task 4 exploration period

## Decision

The registered screen did not pass and no checkpoint was selected. The installed
Task 3 incumbent remains the submission candidate.

The verdict is again narrower than the result. Every arm in this comparison beat
the unchanged incumbent on the registered primary endpoint with its interval
excluding zero — the first time more than one arm has done so — and the arm that
did best also passed every retention gate. What stops it being selectable is that
it is the control arm, which the promotion rule does not consider.

## Question and protocol

The registered #168 exploration schedule is binary: an episode is played fully
randomly or fully greedily, and one in five is random. The `initial_epsilon`,
`epsilon_decay` and `epsilon_floor` fields registered beside it never reach the
agent. A fifth of a long run therefore goes on random play, and the hypothesis was
that this is wasteful once the greedy policy improves.

- Changed factor: `random_episode_period` alone — 5 (`control`, a fifth of
  episodes random), 10 (`reduced`, a tenth), 20 (`sparse`, a twentieth).
- 4 replicas per arm, 1,200 episodes per replica-arm, 14,400 training episodes.
- Every arm shares `all_weights`, the mixed training line-up, and learning rate
  0.0002 — the dose the fine-tuning ladder selected.
- Registered in advance: the sparse arm risks under-exploring early.

## Results

Arm means on the tournament suite (`classic`, three `rule_based_agent`s):

| artifact | score | strict win | eliminations | collection | survived | self-kills |
|---|---:|---:|---:|---:|---:|---:|
| reference (unchanged incumbent) | 1.600 | 0.125 | 0.025 | 0.164 | 0.175 | 0.650 |
| **control — period 5** | **3.025** | **0.200** | **0.144** | **0.256** | 0.369 | 0.512 |
| reduced — period 10 | 2.663 | 0.163 | 0.100 | 0.240 | **0.431** | **0.481** |
| sparse — period 20 | 2.400 | 0.113 | 0.069 | 0.228 | 0.281 | 0.631 |

Score against the unchanged incumbent, every interval excluding zero:

| arm | difference | 97.5% interval |
|---|---:|---|
| control — period 5 | **+1.4250** | [+0.5188, +2.5125] |
| reduced — period 10 | +1.0625 | [+0.1313, +2.0000] |
| sparse — period 20 | +0.8000 | [+0.1187, +1.4751] |

The registered hypothesis is not supported, but neither is it refuted at this
precision. Point estimates fall monotonically as the random episode becomes rarer —
score, strict wins, eliminations and collection all decline — yet the direct
contrasts against the control arm both include zero by wide margins: period 10 is
−0.3625 [−1.4000, +0.7875] on score and period 20 is −0.6250 [−1.4688, +0.1187].
Those intervals permit a meaningful benefit from reducing random play as well as
the harm the point estimates suggest.

What the comparison does establish is that all three arms beat the unchanged
incumbent, each with its interval excluding zero. What it cannot establish is an
ordering between them. The period of five is the best of the three by point
estimate only.

The control arm eliminates 0.144 opponents per game against the incumbent's 0.025,
and collects 0.256 against 0.164, while cutting self-kills from 0.650 to 0.512.

### Retention

Collection fraction, registered margin 0.05:

| suite | reference | control | reduced | sparse |
|---|---:|---:|---:|---:|
| `classic-peaceful` | 0.364 | 0.380 pass | 0.300 **FAIL** | 0.364 pass |
| `coin-heaven` | 0.510 | 0.509 pass | 0.695 pass | 0.549 pass |
| `loot-crate` | 0.209 | 0.231 pass | 0.188 pass | 0.254 pass |

At learning rate 0.0002 retention largely holds, which corroborates the
dose-response: the rate, not the line-up, is what governs it.

### Why nothing is promotable

`reduced` fails `earlier_task_retention` on `classic-peaceful`. `sparse` fails
`self_kills_versus_control`, +0.1187 against a registered threshold of 0.05.
`control` fails nothing — but the promotion rule evaluates treatment arms only, so
the best arm in the comparison is structurally unselectable.

That is worth recording as a defect rather than a verdict. The control arm here is
not the incumbent; it is a trained agent that beats the incumbent by 1.425 score
and passes every gate. A rule that cannot select it is a rule that cannot select
the best available agent when the baseline setting wins.

### Gates

Integrity: `resources`, `genuine_updates`, `behavioral_repeats`,
`opponent_free_invariance` and `latency` pass. `frozen_online_weights` is reported
as not applicable, the re-specified treatment for a programme whose every arm moves
the inherited weights. `invalid_actions` fails, and as in the dose-response the
failure is informative rather than structural: all twelve trained artifacts exceed
the unchanged reference's own count of 38, the worst at 146.

Latency passed on the serial stage: median 1.85 ms, p95 4.58 ms, max 10.23 ms.

## Reproduction and evidence

12 training jobs, 52 evaluation, 13 latency, zero failed.
Training 14.71 CPU-hours in 3.52 wall-hours; evaluation 2.76 in 0.58. Evidence is
published; the release tag, asset URLs, byte size, checksums and the verification
command are in `results/evidence.json` and `results/verification.json`.

This run executed on the tool as it stood at `ecbafe98`, before the fix for the
framework's per-episode log handler leak. At 1,200 episodes per replica it stayed
under the descriptor ceiling that ended the line-up comparison at around 1,500, so
the run is unaffected; a longer budget on this tool would not have completed.

## Limitations and next step

Four replicas is thin, and the intervals are correspondingly wide — the control
arm's spans 0.52 to 2.51, and every between-arm contrast includes zero. The
ordering is consistent across endpoints, which makes it worth recording, but a
consistent ordering of imprecise estimates is a weaker thing than it looks: with
four replicas the comparison cannot separate the arms, and no equivalence margin
was registered that would let it claim they are the same either.

Two things this comparison hands the long training run. The exploration period
stays at five, because nothing here gives a reason to change it — not because
reducing it has been shown to cost performance. And the budget signal is
encouraging: the same family of configuration gained 0.708 score over the
incumbent at 400 episodes and 1.425 at 1,200, so the gain is still growing where
the campaign has so far stopped looking.

Refs [PR #198](https://github.com/1BlauNitrox/mle-final-project/pull/198)
