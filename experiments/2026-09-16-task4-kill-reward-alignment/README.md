# Task 4 kill-reward alignment

## Decision

The registered screen did not pass and no checkpoint was selected. The installed
Task 3 incumbent remains the submission candidate.

The comparison refutes the hypothesis it was built on, and does so in the most
useful way available: not by finding no effect, but by finding the opposite of the
predicted one on the endpoint the hypothesis was about.

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

**Eliminations did not rise.** 0.094, 0.094, 0.081 across a ten-fold increase in
what a kill is worth to the agent. The endpoint the intervention was designed to
move did not move, and if anything moved down.

Score fell monotonically with the reward: 2.631, 2.413, 2.325. Survival fell,
0.375 to 0.250, and self-kills rose, 0.588 to 0.662. Aligned minus control on score
is −0.3063 [−1.2563, +0.7188], so the ordering is consistent but not individually
significant at this power.

Against the unchanged incumbent, the only contrast in the comparison whose interval
excludes zero is the control arm's collection fraction, +0.0903 [+0.0264, +0.1549].
Training improves coin collection. It does not improve hunting.

### What this means

Paying more for eliminations bought more dying without buying more killing. The
misalignment identified in the reward table is real, the ratio genuinely is
inverted relative to the Tournament, but correcting it does not help, which says
the agent's low elimination rate is not an incentive problem.

Across every Task 4 comparison the elimination rate sits near 0.08 to 0.10 per game
regardless of trainable scope, learning rate, exploration schedule, training
line-up, replay capacity and now the elimination reward itself. The most economical
explanation is that the agent does not know how to kill an opponent, and that
raising the price of something it cannot do only makes it accept worse risks
trying. That is a capability question, not a reward question, and nothing in this
campaign has addressed it.

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

Four replicas at 800 episodes can exclude a large effect on eliminations and cannot
exclude a small one. The reading above does not rest on the score ordering, which
is not individually significant, but on the elimination rate being flat across a
ten-fold change in its own price, which is the cleaner observation.

The registered follow-up is not another reward or hyperparameter. If the
elimination rate is capability-bound, the question is whether the agent's features
and horizon can represent an attack at all: the discount factor is 0.9, giving an
effective horizon of about ten steps, and a bomb takes four steps to explode plus
the steps needed to corner an opponent and survive the blast. That is the first
thing to check before spending a long training run on an objective the agent may be
unable to reach.

Refs [PR #201](https://github.com/1BlauNitrox/mle-final-project/pull/201)
