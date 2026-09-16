# Task 4 training line-up at the selected dose

## Decision

The registered screen did not pass and no checkpoint was selected. The installed
Task 3 incumbent remains the submission candidate.

Unlike the three comparisons before it, this one is negative rather than
narrowly-blocked. Neither arm beat the unchanged incumbent, no replica of either
arm was non-worse than it, and the factor under test moved nothing.

## Question and protocol

The opponent-mixture comparison concluded that the training line-up decides
whether earlier-task behaviour is kept. The dose-response then confounded that:
every one of its arms trained against the hard line-up and its middle dose passed
all three retention gates anyway, so the line-up had been measured at a rate where
the hard line-up loses capability whatever it trains against. This comparison
settles the question by varying the line-up alone at the dose the ladder selected.

- Changed factor: `training_opponents`: three `rule_based_agent`s (`control`)
  against one of each of `rule_based_agent`, `coin_collector_agent` and
  `peaceful_agent` (`treatment`).
- 3 replicas per arm, 2,400 episodes per replica-arm, 14,400 training episodes.
  Six times the budget both earlier readings were taken at.
- Every arm shares `all_weights`, learning rate 0.0002 and the #168 exploration
  schedule at period five.
- The trajectory was retained at episodes 600, 1,200, 1,800 and 2,400.

## Results

Arm means on the tournament suite (`classic`, three `rule_based_agent`s):

| artifact | score | strict win | eliminations | collection | survived | self-kills |
|---|---:|---:|---:|---:|---:|---:|
| reference (unchanged incumbent) | **3.100** | 0.275 | 0.175 | 0.247 | 0.225 | 0.700 |
| control (hard line-up) | 2.950 | 0.183 | 0.092 | 0.277 | 0.467 | 0.442 |
| treatment (mixed line-up) | 2.700 | 0.150 | 0.075 | 0.258 | 0.325 | 0.558 |

The registered contrast is empty. Treatment minus control on score is −0.2500
[−0.9750, +0.4002], and no endpoint on that contrast excludes zero. At the dose
the ladder selected, **the training line-up does not measurably matter**, which
is a real answer to the question the mixture comparison left confounded, just not
the answer its result implied.

Against the unchanged incumbent both arms are negative and neither significantly:
control −0.150 [−1.125, +0.783], treatment −0.400 [−1.283, +0.508]. Non-worse
replicas are **0 of 3 for both arms**, not one replica beat the untrained agent.

### The trajectory

The retained milestones were replayed on the same forty tournament worlds, so the
following is paired within this run rather than compared across runs.

| episode | control score | self-kills | survived | elims | treatment score | self-kills | survived | elims |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 600 | 2.650 | 0.625 | 0.308 | 0.100 | 2.642 | 0.550 | 0.358 | 0.067 |
| 1,200 | 2.600 | 0.525 | 0.333 | 0.042 | 2.508 | 0.583 | 0.275 | 0.083 |
| 1,800 | 3.125 | 0.533 | 0.358 | 0.142 | 2.575 | 0.592 | 0.300 | 0.083 |
| 2,400 | 2.950 | 0.442 | 0.467 | 0.092 | 2.658 | 0.558 | 0.325 | 0.067 |

Score is flat across a four-fold range of budget. There is no peak and no decay,
so the negative result above is not over-training.

What does move is survival. For the hard-line-up arm self-kills fall from the
reference's 0.700 to 0.442 and survival rises from 0.225 to 0.467, both
monotonically across the trajectory, while eliminations fall from 0.175 to 0.092.
Score is coins plus kills, so the agent is learning to stop destroying itself and
paying for it by killing less, and the two offset almost exactly.

### Gates

Integrity: `resources`, `genuine_updates`, `behavioral_repeats`,
`opponent_free_invariance` and `latency` pass. `behavioral_repeats` passing is
worth noting, because the dose-response failed it: that failure was diagnosed as
contention from unrelated work on the same machine during the evaluation stage,
and the machine was kept idle this time. `frozen_online_weights` is reported as
not applicable. `invalid_actions` fails, with five of six trained artifacts above
the reference's own count of 94.

Efficacy, treatment arm: only `earlier_task_retention` and
`self_kills_versus_reference` pass. Retention holds for both arms at this dose -
the treatment passes all three suites and the control fails only `coin-heaven` -
which corroborates the dose-response independently: the rate governs retention.

Latency passed on the serial stage: median 6.59 ms, p95 16.72 ms, max 27.37 ms.

## Reproduction and evidence

6 training jobs, 28 evaluation, 7 latency, zero failed. Training 31.59 CPU-hours in
10.58 wall-hours, evaluation 3.70 in 0.82. Evidence is published; the release tag,
asset URLs, byte size, checksums and the verification command are in
`results/evidence.json` and `results/verification.json`.

This run was interrupted twice. The first
attempt died at around episode 1,500 in every replica with
`OSError: [Errno 24] Too many open files`, a file-descriptor leak in the
framework's per-episode logging that is invisible at the 400-episode budgets every
earlier comparison used.
The second attempt stopped itself with five of six replicas complete, because that
ceiling was estimated rather than measured and did not allow for a sixth job
running alone after the first five finished. To rerun we raised it to
43,200 seconds; `results/verification.json` records the rebinding, and the
amendment is in the configuration. Exactly two configuration fields changed, the
wall ceiling and the note recording why, with seeds, arm settings, thresholds and
margins byte-identical.

## Limitations and next step

Three replicas is thin and the intervals are wide, around plus or minus 0.9 on
score. The comparison can exclude a large line-up effect at this dose, but it cannot
exclude a small one.

The reference scored 3.100 here, against 1.375, 1.600, 2.200 and 2.350 on the four
other independent world sets this campaign has measured. Read across all five, the
trained arms cluster between 2.1 and 3.0 while the untrained incumbent ranges from
1.4 to 3.1. That pattern, a consistent trained agent and an erratic untrained one,
is the more interesting reading of the campaign than any single comparison, and
it is not established by any of them individually. It should be checked properly
rather than asserted from five numbers taken at different budgets.

For the long training run, the flat score trajectory is the finding that matters:
across a four-fold budget range on shared worlds, more episodes did not buy more
score. Survival and self-kills were still improving at 2,400, so something is
still being learned, but an expectation that a far longer run produces a far
better score is not supported by the only within-run evidence available.

Refs [PR #199](https://github.com/1BlauNitrox/mle-final-project/pull/199)
