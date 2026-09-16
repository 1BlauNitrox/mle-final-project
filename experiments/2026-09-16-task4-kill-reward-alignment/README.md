# Task 4 kill-reward alignment

## Decision

The registered screen did not pass and no checkpoint was selected. The installed
Task 3 incumbent remains the submission candidate.

The comparison found no support for the hypothesis it was built on, and it is
important to be precise about what that does and does not mean. On the endpoint the
hypothesis was about, eliminations, the point estimates did not rise — but the
intervals are wide enough to contain the rise the hypothesis predicted. The
hypothesis is unsupported by this comparison, not refuted by it.

## Question and protocol

The tournament scores a kill at five points and a coin at one, so score is coins
plus five times kills. The agent's reward table pays ten for a coin and five for a
kill, valuing a kill at half a coin, a ten-fold inversion of the ratio it is
graded on. It also pays five for surviving the round, which the tournament does not
score, against minus ten for dying. The table was inherited from Task 2, where it
was correct because there were no opponents to kill.

Every earlier Task 4 result was consistent with that inversion. The line-up
comparison's trajectory showed training driving self-kills from 0.700 to 0.442 and
survival from 0.225 to 0.467 while eliminations fell from 0.175 to 0.092, with
score flat throughout. The hypothesis was that the agent was succeeding at the
wrong objective, and that paying it the tournament's own ratio would convert that
effort into kills.

- Changed factor: `kill_reward`: 5.0 (`control`, inherited), 20.0 (`weighted`),
  50.0 (`aligned`, matching the tournament's five-to-one ratio against a coin
  reward of ten).
- 4 replicas per arm, 800 episodes per replica-arm, 9,600 training episodes.
- Every arm shares `all_weights`, the mixed line-up, learning rate 0.0002 and the
  #168 exploration schedule at period five.

## Results

Arm means on the tournament suite (`classic`, three `rule_based_agent`s):

| artifact | score | strict win | eliminations | collection | survived | self-kills |
|---|---:|---:|---:|---:|---:|---:|
| reference (unchanged incumbent) | 1.850 | 0.100 | 0.100 | 0.150 | 0.175 | 0.800 |
| control — kill 5 | **2.631** | 0.169 | 0.094 | 0.240 | **0.375** | **0.588** |
| weighted — kill 20 | 2.413 | 0.150 | 0.094 | 0.216 | 0.275 | 0.631 |
| aligned — kill 50 | 2.325 | 0.175 | 0.081 | 0.213 | 0.250 | 0.662 |

**No rise in eliminations was detected.** 0.094, 0.094, 0.081 across a ten-fold
increase in what a kill is worth to the agent. The direct contrasts against the
control are +0.0000 [−0.1250, +0.1500] for reward 20 and −0.0125 [−0.1500, +0.1437]
for reward 50. Those intervals permit increases larger than the whole baseline
elimination rate of roughly 0.1 per game, so the comparison cannot exclude the
effect it was designed to find. What it can say is that no effect large enough to
see at this precision appeared.

Score fell monotonically with the reward: 2.631, 2.413, 2.325. Survival fell,
0.375 to 0.250, and self-kills rose, 0.588 to 0.662. Aligned minus control on score
is −0.3063 [−1.2563, +0.7188], so the ordering is consistent but not individually
significant at this power.

Against the unchanged incumbent, the only contrast in the comparison whose interval
excludes zero is the control arm's collection fraction, +0.0903 [+0.0264, +0.1549].
Training improves coin collection. It does not improve hunting.

### What this means

The misalignment identified in the reward table is real — the ratio genuinely is
inverted relative to the tournament — but correcting it produced no measurable
gain in eliminations here, and the point estimates for survival and score moved the
wrong way. This comparison therefore gives no reason to change the elimination
reward, and some reason to suspect that raising it carries a cost.

Across every Task 4 comparison the elimination rate sits near 0.08 to 0.15 per game
regardless of trainable scope, learning rate, exploration schedule, training
line-up, replay capacity and now the elimination reward. One reading is that the
rate is capability-bound rather than incentive-bound. This comparison is consistent
with that reading but does not establish it: each individual contrast is too
imprecise to exclude an incentive effect, and a consistent pattern across
underpowered comparisons is suggestive rather than conclusive.

Context for the rate itself: `rule_based_agent`, the hand-written reference
implementation, achieves 0.188 eliminations per game against three copies of itself
over 40 games, while killing itself 0.544 times per game and never surviving. The
agent's rate is therefore within the range the reference implementation reaches,
which is a different situation from being unable to attack.

### Gates

Integrity: `resources`, `genuine_updates`, `behavioral_repeats`,
`opponent_free_invariance` and `latency` pass. `frozen_online_weights` is reported
as not applicable and `self_kills_versus_control` as null for the control arm
itself, both the corrected promotion rule declining to veto on a gate that cannot
apply. `invalid_actions` fails, as in every completed Task 4 comparison.

Every arm fails `earlier_task_retention` on `classic-peaceful` collection, and both
raised-reward arms additionally fail `coin-heaven`. Non-worse replicas is 2 of 4
for all three arms against a threshold of 3. No arm is promotable on any reading.

Latency passed on the serial stage: median 6.05 ms, p95 17.02 ms, max 25.89 ms.

## Reproduction and evidence

12 training jobs, 52 evaluation, 13 latency, zero failed. Training 17.03 CPU-hours
in 4.17 wall-hours; evaluation 6.14 in 1.33. The evidence archive is published; the
release tag, asset URLs, byte size, checksums and the verification command are in
`results/evidence.json` and `results/verification.json`.

The elimination rewards used here were added to the registered reward vocabulary
before the run. The guard that enforces that vocabulary refused the 50.0 arm on the
first attempt, which is the guard working: an unregistered reward must not reach a
training game through a configuration edit alone.

## Limitations and next step

Four replicas at 800 episodes cannot resolve the effects this comparison produced.
The elimination contrasts span roughly ±0.15, which is larger than the baseline
rate itself, so the comparison excludes neither a substantial increase nor a
substantial decrease. No equivalence margin was registered, so the arms cannot be
called equivalent either. Every statement above is bounded by that.

A follow-up worth considering is not another reward or hyperparameter. If the
elimination rate is capability-bound — which this comparison suggests without
establishing — the question is whether the agent's features and horizon can
represent an attack at all: the discount factor is 0.9, giving an
effective horizon of about ten steps, and a bomb takes four steps to explode plus
the steps needed to corner an opponent and survive the blast. That is the first
thing to check before spending a long training run on an objective the agent may be
unable to reach.

Refs [PR #201](https://github.com/1BlauNitrox/mle-final-project/pull/201)
