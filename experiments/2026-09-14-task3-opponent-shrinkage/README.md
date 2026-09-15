# Task 3 opponent-weight shrinkage

## Decision

This exploratory comparison **failed its registered screen**. No arm is
promotable, no checkpoint is selected, and the unchanged reference remains the
incumbent Task 3 artifact.

It is nevertheless the comparison that settles the Task 3 question. The
registered dose-response came out monotone toward the reference, which is the
outcome the protocol declared in advance would indicate undirected drift rather
than opponent signal.

## Question and protocol

A zero opponent block reproduces the unchanged reference exactly, because all
thirteen opponent inputs are zero-initialised and every inherited weight is
frozen. An L2 penalty on those 832 columns is therefore a continuous dial
between training them freely and not training them at all.

That makes the dose-response discriminating, and the protocol registered both
readings before execution:

* if the block only accumulates undirected drift, performance should rise
  monotonically with the coefficient, toward the reference;
* if it carries genuine opponent signal, an intermediate coefficient should beat
  both the unpenalised arm *and* the reference.

[#179](../2026-09-14-task3-opponent-regularization/README.md) tested `0.01`,
which is two orders of magnitude too small to bind against a 0.0005 learning
rate, and found nothing.

Arms varied only the coefficient: `control` at `0.0`, `mild` at `0.1`, `strong`
at `1.0`. Every arm held approach potential scale 1.0, frozen inherited weights,
the #168 episode-mixture exploration schedule, the native `+5` elimination
reward and one optimizer update per eight eligible transitions. Eight paired
replicas trained 400 episodes per arm: 9,600 training episodes. The twenty-four
fixed-final checkpoints and the unchanged reference were evaluated on 40 shared
hunting worlds and 40 worlds in each of three opponent-free retention suites,
with two repeats: 8,000 evaluation episodes, plus 250 in a separate serial
latency stage.

## Results

Means on the hunting suite:

| arm | attack opportunities / 100 steps | opponent distance | eliminations | score | strict win | collection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control (`lambda = 0`) | 0.729 | 17.08 | 0.156 | 3.128 | 0.769 | 0.261 |
| mild (`lambda = 0.1`) | 0.371 | 17.13 | 0.138 | 3.181 | 0.731 | 0.277 |
| strong (`lambda = 1.0`) | 0.301 | 16.42 | 0.178 | 4.013 | 0.762 | 0.347 |
| unchanged reference | 0.261 | 16.93 | 0.200 | 4.000 | 0.775 | 0.333 |

**The dose-response is monotone toward the reference.** Score rises 3.128,
3.181, 4.013 and collection rises 0.261, 0.277, 0.347 as the penalty
strengthens, ending at the reference's 4.000 and 0.333. Against the reference,
the strongly penalised arm differs by `+0.013` score (97.5% CI
`[-1.356, +1.303]`), `+0.014` collection (`[-0.080, +0.103]`) and `-0.022`
eliminations (`[-0.163, +0.116]`): indistinguishable on every measure.

**No intermediate coefficient beats either endpoint.** The registered
alternative reading required exactly that, and nothing in the table supports it.

**Attack exposure falls monotonically as the penalty strengthens**, 0.729,
0.371, 0.301 against the reference's 0.261. The extra exposure the unpenalised
arm showed was therefore itself part of the drift, not a behaviour worth
keeping: it came with the lowest score and collection of the three arms.

Read against the pre-registered criteria, this is the drift outcome. The 832
trainable opponent columns carry no signal that survives at this training
budget; what they accumulate is movement away from a starting point that was
already the best available policy.

All three opponent-free retention suites passed for every arm, and opponent-free
behaviour was identical to the reference throughout. Genuine updates,
behavioural repeats, frozen weights, resources and latency all passed. The
zero-invalid-action gate failed again, partly on the unchanged reference's own
actions, so no arm can pass it; it is preserved as a failure rather than
relaxed. Non-worse replicas were 2, 3 and 3 of 8 against a registered
requirement of 7.

## Reproduction and evidence

[Protocol](config.json) - [all gates, intervals and summaries](results/analysis.json) -
[per-artifact table](results/summary.csv) - [provenance](results/verification.json) -
[archive locator](results/evidence.json).

## Limitations and next step

A monotone dose-response is consistent with drift but does not prove that no
signal exists: it shows that at this training budget, this exploration schedule
and this reward structure, no amount of the signal survives the noise. A larger
budget, a denser opponent interaction, or a different opponent might still
differ, and the opportunity measurement recorded with these protocols gives a
concrete reason to expect the peaceful opponent to be the hardest case.

What it does establish is that further single-factor interventions on the
existing opponent block are not worth more compute. The reference remains the
selected Task 3 artifact, and the honest reading is that Task 3's gains would
have to come from a change of mechanism rather than a change of coefficient.

Refs [PR #187](https://github.com/1BlauNitrox/mle-final-project/pull/187). 
