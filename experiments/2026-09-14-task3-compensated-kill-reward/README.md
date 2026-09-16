# Task 3 compensated elimination reward

## Decision

This exploratory comparison **failed its registered screen**. No arm is
promotable, no checkpoint is selected, and the unchanged reference remains the
incumbent Task 3 artifact.

Raising the elimination reward under strong approach pressure changed nothing.
The useful observation is a different one: both arms sit closer to the reference
than the same shaped configuration did at a longer training budget, and that
difference tracks the number of optimizer updates rather than the shaping.

## Question and protocol

The completed [approach-shaping comparison](../2026-09-14-task3-approach-shaping/README.md)
found that potential scale 3.0 raised attack exposure more than any other
setting, and that its arm lost strict wins and collection against the control.
That cost was registered before execution: the potential is worth
`scale * gamma = 2.7` next to an opponent, so it lowers the immediate reward of
an eliminating transition from `+5` to `+2.3`, with the remainder carried by
the bootstrap.

The question here is whether restoring a strongly positive elimination signal
lets the exposure that shaping creates actually be reinforced.
[#175](../2026-09-14-task3-elimination-reward/README.md) compared the same two
rewards and found nothing, but in a regime where the agent reached almost no
attack opportunities to reinforce.

Arms varied only the attributable elimination reward: `control` at the native
`+5`, `treatment` at `+20`. Every arm held approach potential scale 3.0, frozen
inherited weights, the #168 episode-mixture exploration schedule, learning rate
0.0005 and one optimizer update per eight eligible transitions. Ten paired
replicas trained 400 episodes per arm: 8,000 training episodes. The twenty
fixed-final checkpoints and the unchanged reference were evaluated on 40 shared
hunting worlds and 40 worlds in each of three opponent-free retention suites,
with two repeats: 6,720 evaluation episodes, plus 210 in a separate serial
latency stage.

Following the update-cadence power result, this comparison buys replicas rather
than evaluation worlds: ten replicas on forty worlds instead of five on a
hundred and twenty.

## Results

Means on the hunting suite:

| arm | attack opportunities / 100 steps | opponent distance | eliminations | score | strict win | collection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control (`+5`) | 0.426 | 15.17 | 0.193 | 3.015 | 0.682 | 0.228 |
| treatment (`+20`) | 0.503 | 15.21 | 0.198 | 2.965 | 0.660 | 0.220 |
| unchanged reference | 0.475 | 14.72 | 0.200 | 3.950 | 0.725 | 0.328 |

**The elimination reward did nothing.** Treatment minus control was `+0.005`
eliminations (97.5% CI `[-0.085, +0.100]`), `-0.050` score
(`[-0.833, +0.765]`) and `+0.077` attack opportunities per 100 steps
(`[-0.252, +0.409]`). Every interval is centred on zero. #175 reached the same
conclusion without shaping; it now replicates with strong approach pressure, so
the earlier explanation that there were no opportunities to reinforce does not
survive.

**Both arms match the reference on hunting.** Against the reference,
eliminations were `-0.008` (`[-0.145, +0.123]`) for control and `-0.003`
(`[-0.128, +0.110]`) for treatment; score `-0.935` (`[-2.145, +0.233]`) and
`-0.985` (`[-2.165, +0.110]`); opponent distance `+0.455` and `+0.489`, both
spanning zero. Non-worse replicas were 5 of 10 and 7 of 10, against 0 of 5 in
the update-cadence comparison. The registered requirement was 9 of 10.

**Collection remains the one surviving deficit.** It was `-0.100`
(`[-0.180, -0.021]`) for control and `-0.108` (`[-0.190, -0.029]`) for
treatment, the only intervals in the whole comparison that exclude zero.

All three opponent-free retention suites passed for both arms, and opponent-free
behaviour was identical to the reference throughout, as frozen inheritance
requires. Genuine updates, behavioural repeats, frozen weights and resource
limits passed. Decision latency passed on the serial stage: median 3.60 ms,
p95 8.96 ms, maximum 20.71 ms. The zero-invalid-action gate failed again, and
again part of the count belongs to the unchanged reference itself, so no arm can
pass it; it is preserved as a failure rather than relaxed.

## Why this arm differs from the shaping comparison's strongest arm

Scale 3.0 with the native `+5` reward looks like the approach-shaping
comparison's `high` arm, which measured score `-2.313` (`[-3.380, -1.173]`)
against the reference while the same settings measured `-0.935` here, not
significant. The two are not the same run. That arm trained 600 episodes per
replica and averaged 20,074 optimizer updates; this one trained 400 and averaged
13,292, a third fewer gradient steps.

The difference therefore tracks the update count rather than the shaping, in the
direction the rest of the programme predicts: the further the opponent block is
driven from its zero initialisation, the worse the agent gets. The same ordering
appears in the update-cadence comparison, where one update per eight eligible
transitions beat one per transition on two hosts, and in the shrinkage
comparison, where penalising the block back toward zero recovers the reference
monotonically.

What this arm does not support is the reading that approach shaping closes the
gap to the reference. Its gap is smaller because it trained less, not because it
was shaped. World difficulty does not account for it either: the same unchanged
reference scores between 3.950 and 4.450 across the four comparisons' world
sets, a spread well below the 1.4 points separating these two arms.

## Reproduction and evidence

[Protocol](config.json) - [all gates, intervals and summaries](results/analysis.json) -
[per-artifact table](results/summary.csv) - [provenance](results/verification.json) -
[archive locator](results/evidence.json).

## Limitations and next step

Two independent comparisons now show the elimination reward has no measurable
effect on this agent, at two very different exposure levels. It should not be
revisited without a new mechanism to justify it.

Ten replicas were still not enough to resolve the registered `+0.100` elimination
threshold. The comparison with the longer-trained arm above points at training
budget, not replica count, as the dominant influence on how far this agent falls
below the reference. These development seeds must not become a confirmation set,
and nothing here establishes Task 2 completion or tournament readiness.

Refs [PR #187](https://github.com/1BlauNitrox/mle-final-project/pull/187).
