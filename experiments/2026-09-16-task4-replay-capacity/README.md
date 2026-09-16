# Task 4 replay capacity

## Decision

The registered screen did not pass and no checkpoint was selected. The installed
Task 3 incumbent remains the submission candidate.

The comparison answers its question clearly in the negative: over a ten-fold range
the replay capacity does not measurably change anything. That removes a
hyperparameter from the final training run's design, which is what it was run to do.

## Question and protocol

The agent's replay buffer holds 10,000 transitions and fills at around episode 37,
so from that point every new transition evicts one about thirty-seven episodes old.
At the budgets this campaign screened at the ceiling is invisible; over the tens of
thousands of episodes a tournament agent would be trained for it means the network
only ever learns from roughly the last 0.1% of its own experience, far shorter than
deep Q-learning is normally run with.

- Changed factor: `replay_capacity`: 10,000 (`control`), 50,000 (`wide`),
  100,000 (`widest`).
- 4 replicas per arm, 1,000 episodes per replica-arm, 12,000 training episodes.
- Every arm shares `all_weights`, the mixed line-up, learning rate 0.0002 and the
  #168 exploration schedule at period five.

## Results

Arm means on the tournament suite (`classic`, three `rule_based_agent`s):

| artifact | score | strict win | eliminations | collection | survived | self-kills |
|---|---:|---:|---:|---:|---:|---:|
| reference (unchanged incumbent) | 1.875 | 0.100 | 0.075 | 0.167 | 0.300 | 0.625 |
| control - 10,000 | 2.731 | 0.144 | 0.094 | 0.251 | 0.287 | 0.562 |
| wide - 50,000 | 2.581 | 0.156 | 0.119 | 0.221 | 0.350 | 0.569 |
| widest - 100,000 | 2.725 | 0.175 | 0.087 | 0.254 | **0.431** | **0.456** |

Score against the unchanged incumbent, none excluding zero: control +0.8562
[−0.1750, +1.8625], wide +0.7063 [−0.3375, +1.7000], widest +0.8500
[−0.1187, +1.7875].

The three arms are indistinguishable on score, 2.731, 2.581, 2.725 across a
ten-fold capacity range, with the smallest buffer neither better nor worse than
the largest. **Replay capacity is not a lever at this budget**, and the final training
can keep the inherited 10,000 without further thought.

The one direction worth noting is survival: the widest arm survives 0.431 against
the control's 0.287 and kills itself least, 0.456 against 0.562. That did not
convert into score, and it is a single unreplicated contrast, so it is recorded
rather than claimed.

### Retention

All three arms pass `classic-peaceful` and `loot-crate`. The widest arm fails
`coin-heaven`, which is the only retention failure in the comparison and the reason
that arm is not promotable on its own terms.

### Why nothing is promotable

This is the first run analyzed under the corrected promotion rule, which judges
every arm rather than only the treatments and therefore counts three hypotheses
instead of two. The interval widened from 95% to 98.33% accordingly, and no arm's
score gain clears zero at that width. `control` also fails `nonworse_replicas`,
2 of 4 against a threshold of 3, and `widest` fails retention.

It is worth being explicit that the wider interval is a consequence of a rule
change made before this run's results existed, not after them. The exploration
comparison's control arm cleared the score gate at a 97.5% interval with a gain of
1.425; the gains here are around 0.85 and would not have cleared at either width.

### Gates

Integrity: `resources`, `genuine_updates`, `behavioral_repeats`,
`opponent_free_invariance` and `latency` pass. `frozen_online_weights` is reported
as not applicable, and `self_kills_versus_control` is reported as null for the
control arm itself. Both are the corrected rule declining to veto on a gate that
cannot apply. `invalid_actions` fails, as in every completed Task 4 comparison.

## Reproduction and evidence

12 training jobs, 52 evaluation, 13 latency, zero failed.
Training 9.68 CPU-hours in 2.40 wall-hours; evaluation 2.95 in 0.61. The evidence
archive is published; the release tag, asset URLs, byte size, checksums and the
verification command are in `results/evidence.json` and `results/verification.json`.

## Limitations and next step

Four replicas and 1,000 episodes can exclude a large capacity effect at this
Budget. They cannot exclude a small one, and they say nothing about capacity at the
budgets a long run would reach, which is where the argument for a larger buffer was
always strongest. N no effect is visible where this campaign can see, so
the inherited value is kept because nothing justifies changing
it, not because a larger buffer has been shown to be useless.

Refs [PR #200](https://github.com/1BlauNitrox/mle-final-project/pull/200)
