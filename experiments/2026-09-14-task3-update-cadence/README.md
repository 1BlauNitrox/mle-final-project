# Task 3 update-cadence confirmation

## Decision

This exploratory comparison failed its registered screen. No checkpoint is
selected or promoted, and Task 2 remains cumulatively incomplete.

Two things did come out of it. The #178 update-cadence effect did not replicate
at higher power, and both trained arms were established as reliably worse than
the unchanged reference, with a measured mechanism for why.

## Question and protocol

[#178](../2026-09-14-task3-update-frequency/README.md) reported the first Task 3
intervals that excluded zero: score `+1.280` (95% CI `[+0.450, +2.100]`), strict
wins `+0.110` (`[+0.010, +0.220]`) and collection `+0.1117`
(`[+0.0455, +0.1778]`) for one update per eight eligible transitions against one
per transition. Its registered elimination endpoint failed at `+0.055`
(`[-0.035, +0.145]`).

The same unchanged reference checkpoint had measured 0.075, 0.200 and 0.250
eliminations per game in #178, #168 and #175. With a gate demanding `>= +0.100`
and a positive interval lower bound, that endpoint could not be resolved on 40
worlds. This comparison therefore repeated the identical contrast at three times
the hunting evaluation sample and six times the training budget, to test whether
the #178 result was real.

Arms varied only optimizer updates per eligible transition: `control` every
transition, `treatment` every eighth. Skipped transitions still entered replay
with their full reward, and target synchronisation still counted actual
optimizer updates. Both arms froze the inherited online weights and trained only
the 832 opponent-input columns, used the #168 episode-mixture exploration
schedule, the native `+5` kill reward and learning rate 0.0005. Five paired
replicas trained 600 episodes per arm: 6,000 training episodes. The ten
fixed-final checkpoints and the unchanged reference were evaluated on 120 shared
hunting worlds and 40 worlds in each of three opponent-free retention suites,
with two repeats: 5,280 evaluation episodes, plus 110 episodes in a separate
serial latency stage.

## Results

**The #178 effect did not replicate.** Same direction, same rough magnitude, but
none of the three intervals that previously excluded zero still does:

| treatment minus control | #178 (40 worlds, 100 episodes) | this comparison (120 worlds, 600 episodes) |
| --- | ---: | ---: |
| eliminations | +0.055 `[-0.035, +0.145]` | +0.052 `[-0.032, +0.145]` |
| score | +1.280 `[+0.450, +2.100]` | +1.007 `[-0.207, +2.527]` |
| strict win | +0.110 `[+0.010, +0.220]` | +0.180 `[-0.018, +0.405]` |
| collection | +0.1117 `[+0.0455, +0.1778]` | +0.083 `[-0.017, +0.204]` |

**Tripling the evaluation worlds bought no precision.** The elimination interval
width moved from 0.180 to 0.177. Replicas within an arm ranged from 0.000 to
0.217, so the dominant variance is between replicas and cannot be reduced by
adding worlds. This is a design lesson for every remaining Task 3 experiment:
buy replicas, not worlds.

**Both trained arms are reliably worse than doing nothing.** Every interval
below excludes zero:

| versus unchanged reference | eliminations | score | collection |
| --- | ---: | ---: | ---: |
| every eighth transition | -0.127 `[-0.218, -0.033]` | -1.75 `[-2.79, -0.61]` | -0.124 `[-0.201, -0.038]` |
| every transition | -0.178 `[-0.270, -0.082]` | -2.75 `[-3.89, -1.61]` | -0.207 `[-0.294, -0.121]` |

No treatment replica of five matched the reference on eliminations; the
registered requirement was four.

**The mechanism is opponent avoidance.** Mean opponent distance rose against the
reference in both arms with intervals excluding zero: `+1.026`
(`[+0.278, +1.788]`) for every-eighth and `+1.414` (`[+0.430, +2.357]`) for
every-transition. Attack opportunities per 100 steps fell from the reference's
0.688 to between 0.005 and 0.444 across trained replicas. The 832 trainable
opponent-input weights learn to keep the agent away from opponents.

All three opponent-free retention suites passed for both arms, and opponent-free
behaviour was identical to the reference throughout. Genuine updates, behavioural
repeats, frozen weights and resource limits passed. Decision latency passed on
the serial stage: median 1.08 ms, p95 2.60 ms, maximum 6.94 ms, against 22.54 ms
maximum in the concurrent evaluation stage, which is not gated.

The zero-invalid-action gate failed with 20 retained invalid actions, four of
them the unchanged reference's own. The gate cannot be passed by any arm while
the reference contributes to it, and needs re-specifying prospectively rather
than relaxing here.

## Reproduction and evidence

Training used 5.12 CPU-hours over 1.08 wall-hours and evaluation 2.57 over 0.55,
on the laptop; all 65 jobs completed with no failed attempt. Executed source
`3a7e392`, runtime `c4ddfa4`, configuration SHA-256
`8e86c781aba9c0ec3b8ecf3979e83f158cdbec194043b833798a110275dcf5b3`.

The archive was transferred to the PC, checksum-verified against its published
manifest, and its analysis reproduced there independently. The executed copies
of all ten tool files in the archive are byte-identical to the registered branch.
The executed seed audit scanned 689 tracked text blobs against this profile's
seeds, 61 more than the registration audit, and still reported no collision and
no seed shared with the companion approach-shaping profile.

[Protocol](config.json) - [all gates, intervals and summaries](results/analysis.json) -
[per-artifact table](results/summary.csv) - [provenance](results/verification.json) -
[archive locator](results/evidence.json).

## Limitations and next step

Two independent runs on two hosts both put the update-cadence effect in the same
direction without either establishing it. That is suggestive and nothing more;
the intervention is not adopted and no default changes.

The useful output is the negative one. Training the opponent-input columns under
this reward structure teaches avoidance, which is now measured rather than
inferred, and it explains the whole run of failed Task 3 hunting experiments
more economically than any of the optimizer-level hypotheses tested so far. The
companion approach-shaping comparison was registered to attack exactly that
mechanism. These development seeds must not become a confirmation set.

Refs [PR #187](https://github.com/1BlauNitrox/mle-final-project/pull/187).
