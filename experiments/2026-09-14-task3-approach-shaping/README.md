# DRAFT - agent-drafted, not yet reviewed. Delete this line once you have read, verified and edited the text below.

# Task 3 opponent-approach potential shaping

## Decision

This exploratory comparison **failed its registered screen**. No arm is
promotable, no checkpoint is selected, and the unchanged reference remains the
incumbent Task 3 artifact. Task 2 remains cumulatively incomplete.

The result is not empty. Shaping at scale 1.0 removed the training-induced
opponent avoidance that the companion update-cadence comparison established,
and shaping at scale 3.0 was measurably harmful in the way the protocol
predicted before execution.

## Question and protocol

The [#162 diagnosis](../2026-09-12-task3-hunting-diagnosis/README.md) recorded
14 geometric attack states in 3,372 opponent-present states, only two of them
with `BOMB` available, at a median opponent Manhattan distance of 16 on a 17x17
board. The hypothesis was that attack opportunity, not policy quality, is the
binding constraint on hunting, and that potential-based shaping toward the
nearest public opponent would raise exposure and through it eliminations.

The registered potential is

```text
Phi(s) = scale * gamma ** d
```

with `d` the Manhattan distance to the nearest public opponent and `gamma` the
agent's own discount factor, and `Phi = 0` without a visible opponent and at
terminal states. This is the discounted value of a `scale`-sized reward `d`
steps away, so the shaped term `F(s, s') = gamma * Phi(s') - Phi(s)` is exactly
`0` for a step that closes the distance by one, `-(1 - gamma) * Phi` for
standing still and `-(1 - gamma**2) * Phi` for retreating. A potential linear in
distance was rejected during design: at `gamma = 0.9` its drift term is several
times larger than its own per-step gradient, so it would penalise closing
distance at every realistic separation.

Potential-based shaping is policy invariant, so the shaped optimal policy is the
unshaped one. The registered price, stated in `config.json` before execution, is
that the immediate reward of an eliminating transition falls by `scale * gamma`:
from `+5` to `+4.1` at scale 1.0 and to `+2.3` at scale 3.0, with the remainder
carried by the bootstrap.

Three arms varied only that scale: `control` 0.0, `low` 1.0, `high` 3.0. All
arms froze the inherited online weights and trained only the 832 opponent-input
columns, used the #168 episode-mixture exploration schedule, the native `+5`
kill reward, learning rate 0.0005 and one optimizer update per eight eligible
transitions. Five paired replicas trained 600 episodes per arm: 9,000 training
episodes. The fifteen fixed-final checkpoints and the unchanged reference were
evaluated on 120 shared hunting worlds and 40 worlds in each of three
opponent-free retention suites, with two repeats: 7,680 evaluation episodes. A
separate serial stage replayed 160 episodes for decision latency. Intervals are
Bonferroni-adjusted to 97.5% for the two arms that could be promoted.

## Results

Means on the 120-world hunting suite:

| arm | attack opportunities / 100 steps | opponent distance | eliminations | score | strict win | collection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control (no shaping) | 0.325 | 17.47 | 0.083 | 2.19 | 0.668 | 0.197 |
| low (scale 1.0) | 0.345 | 16.58 | 0.172 | 2.83 | 0.695 | 0.219 |
| high (scale 3.0) | 0.387 | 17.05 | 0.102 | 1.93 | 0.583 | 0.158 |
| unchanged reference | 0.428 | 15.76 | 0.217 | 4.24 | 0.808 | 0.351 |

**Shaping at scale 1.0 closed the hunting gap to the reference.** Against the
reference, the unshaped control is worse on both mechanism measures with
intervals excluding zero - eliminations `-0.133` (97.5% CI `[-0.243, -0.018]`)
and opponent distance `+1.709` (`[+0.382, +3.041]`) - while the shaped low arm
is not: `-0.045` (`[-0.147, +0.055]`) and `+0.818` (`[-0.177, +1.959]`). Low
versus control on eliminations was `+0.088` (`[-0.005, +0.187]`), missing the
interval requirement by five thousandths.

**Shaping at scale 3.0 was harmful.** Against control it lost strict wins,
`-0.085` (`[-0.170, -0.002]`), and collection, `-0.040` (`[-0.080, -0.002]`);
both intervals exclude zero. The registered trade-off materialised as written.

No arm beat the reference. Low remained worse on score `-1.412`
(`[-2.415, -0.380]`), strict wins `-0.113` (`[-0.220, -0.008]`) and collection
`-0.132` (`[-0.204, -0.058]`). One low replica of five was non-worse than the
reference on eliminations; the registered requirement was four.

**The registered primary endpoint was the wrong instrument.** Attack-opportunity
exposure produced the widest intervals of any measure - low versus control
`+0.020` (`[-0.418, +0.289]`) - because replicas within an arm ranged from 0.050
to 1.310. It was chosen for being dense within an episode, but the dominant
variance is between replicas, not within them.

All three opponent-free retention suites passed for every arm, and opponent-free
behaviour was identical to the reference in every comparison, as frozen
inheritance requires. Genuine updates, behavioural repeats, frozen weights and
resource limits passed. Decision latency passed on the serial stage: median
3.64 ms, p95 9.05 ms, maximum 20.67 ms.

The zero-invalid-action gate failed with 60 retained invalid actions, of which
four belong to the unchanged reference itself. Shaped arms produced more than
control (low 22, high 28, control 6), so pushing toward opponents appears to
raise blocked-move attempts. This gate cannot be passed by any arm while the
reference contributes to it; that is a defect in the gate, and it must be
re-specified prospectively rather than relaxed for this result.

## Reproduction and evidence

Training used 14.80 CPU-hours over 3.07 wall-hours, evaluation 8.86 over 1.83,
and the latency stage 0.16 over 0.16. All 95 jobs completed with no failed
attempt. Executed source `2f95a02`, runtime `c4ddfa4`, configuration SHA-256
`26e66bc192564f8ae7ee5a6d8c946b39c7458275debaab177c699defcdfd2e98`.

[Protocol](config.json) - [all gates, intervals and summaries](results/analysis.json) -
[per-artifact table](results/summary.csv) - [provenance](results/verification.json) -
[archive locator](results/evidence.json).

## Limitations and next step

Five replicas per arm cannot resolve a `+0.10` elimination effect: the companion
update-cadence comparison showed that tripling evaluation worlds left the
interval width essentially unchanged because the variance sits at the replica
level. Any follow-up should spend its budget on replicas rather than worlds.

The dose-response suggests a useful scale below 3.0 and at or near 1.0, but one
non-significant contrast is not evidence of an optimum, and these development
seeds must not become a confirmation set. Nothing here establishes Task 2
completion or tournament readiness.

Refs #TBD. Peer review by a non-author is required before merge. AI assistance
was used.
