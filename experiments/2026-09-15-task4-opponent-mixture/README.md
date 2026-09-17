# Task 4 training opponent mixture

## Decision

The registered screen did not pass. No arm is promotable, no checkpoint was
selected, and the installed Task 3 incumbent remains the submission candidate.

The comparison nonetheless answers a question the campaign needed answered, and
not the one it was registered to answer. It found no tournament difference between
the two training line-ups, and it found that the line-up decides whether the agent
keeps its earlier-task behaviour.

## Question and protocol

Three `rule_based_agent`s kill an untrained agent quickly, so most training episodes
end before the agent has collected anything. The registered question was whether a
softer line-up — one `rule_based_agent`, one `coin_collector_agent`, one
`peaceful_agent` — produces a better agent when both are measured on the same hard
tournament suite.

- Changed factor: `training_opponents`, hard (control) against mixed (treatment).
  Every other setting is shared, including the `all_weights` trainable scope and
  learning rate 0.0005.
- 6 replicas per arm, 400 episodes per replica-arm, 4,800 training episodes.
- Primary endpoint: mean native score per game on `classic-rule-based` against the
  unchanged incumbent.
- Registered before execution: the hard arm trains on the same distribution it is
  evaluated on and the mixed arm does not, so the mixed arm has to overcome that
  mismatch to win.

## Results

Arm means on the tournament suite (`classic`, three `rule_based_agent`s):

| artifact | score | strict win | eliminations | collection | survived | self-kills |
|---|---:|---:|---:|---:|---:|---:|
| reference (unchanged incumbent) | 2.200 | 0.175 | 0.075 | 0.203 | 0.200 | 0.750 |
| control (hard line-up) | 2.404 | 0.125 | 0.079 | 0.223 | 0.350 | 0.583 |
| treatment (mixed line-up) | 2.329 | 0.117 | 0.088 | 0.210 | 0.242 | 0.650 |

The registered contrast is flat. Treatment minus control: score −0.0750
[−0.7292, +0.6250]; no endpoint's interval excludes zero, and neither does any
endpoint of either arm against the unchanged incumbent (control +0.2042
[−0.5917, +1.0001], treatment +0.1292 [−0.6626, +0.8751]). Non-worse replicas:
4 of 6 for both arms, against a registered threshold of 5.

The registered mismatch penalty was therefore not paid: training on a softer
line-up than the one evaluated cost nothing measurable on the tournament suite.
Nor did it gain anything.

### Retention - important results

| suite | control | treatment | reference | margin |
|---|---:|---:|---:|---|
| `classic-peaceful` | 0.239 **FAIL** | 0.275 pass | 0.314 | 0.05 |
| `coin-heaven` | 0.475 **FAIL** | 0.572 pass | 0.606 | 0.05 |
| `loot-crate` | 0.138 **FAIL** | 0.164 pass | 0.203 | 0.05 |

Collection fraction. The hard line-up fails all three retention gates; the mixed
line-up passes all three. The treatment still degrades slightly on each suite —
0.039, 0.034 and 0.039 — but stays inside the registered margin, and
`earlier_task_retention` is a passing gate for it.

This replicates across experiments. The control arm here is the same configuration
as the trainable-scope comparison's treatment arm, and it failed retention there
too.

It does not follow that the line-up is what decides retention, and the
dose-response run afterwards shows why. Every arm of that ladder trained against
the hard line-up, and its middle dose — learning rate 0.0002 — passed all three
retention gates at this same budget. Both arms here ran at 0.0005, a rate at which
the hard line-up loses capability whatever it trains against, so this comparison
cannot separate the line-up from the rate it was measured at. The registered
`lineup-trajectory` comparison varies the line-up alone at the selected rate and
is what settles the question.

### Gates

Integrity: `resources`, `genuine_updates`, `behavioral_repeats`,
`opponent_free_invariance` and `latency` pass. `frozen_online_weights` fails as
expected for the `all_weights` scope both arms register, and `invalid_actions`
fails under the known unrelaxed gate defect.

Efficacy, treatment arm: `earlier_task_retention` and `self_kills_versus_reference`
pass; `exposure_increase`, the hunting endpoints, `score_versus_reference_ci`,
`self_kills_versus_control` and `nonworse_replicas` fail. That combination is the
reason a retention-safe arm is still not promotable: it is safe without being
better.

Latency passed on the serial stage: median 2.26 ms, p95 6.57 ms, max 28.15 ms over
22,800 decisions, against registered limits of 50 ms and 100 ms.

## Reproduction and evidence

12 training jobs, 52 evaluation, 13 latency, zero failed.
Training 3.56 CPU-hours in 0.90 wall-hours; evaluation 2.65 in 0.55; latency 0.06
in 0.07. The evidence archive is published; its release tag, both asset URLs, byte
size, checksums and the verification command are in `results/evidence.json` and
`results/verification.json`.

## Limitations and next step

Nothing here beats the incumbent, and the comparison was not powered to detect the
effect sizes it produced: the interval on the primary endpoint is roughly ±0.75
score against differences of a fifth of that. The campaign's variance is at the
replica level, so a real tournament gain has to be established with more replicas
rather than more evaluation worlds.

What the completed comparisons support is the `all_weights` scope, which is the
only thing that has moved tournament score, trained at the learning rate the
dose-response selected, which is what has held retention. Whether the mixed
line-up adds anything on top of the right rate is the question those runs leave
open, and it is the one `lineup-trajectory` was registered to answer.

Refs [PR #196](https://github.com/1BlauNitrox/mle-final-project/pull/196)
