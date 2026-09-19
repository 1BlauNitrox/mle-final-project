# Survival guard, confirmation on fresh worlds

Decision: DO NOT SHIP. Two of the three registered conditions failed. The promising factorial result did not replicate.

We ran the confirmation on `monitoring/final-training-evaluation` under `monitoring/survival-guard-confirmation/` at commit `d5ab420`. Both cells completed, with 6,500 games in total. We validated the runs before looking at the results.

## Why we ran a confirmation

The [factorial](../2026-09-18-survival-guard-factorial/README.md) measured the guard at score +0.184 [-0.021, +0.381] and self-kills -0.044 [-0.087, +0.00000].

Our registered rule required the self-kill interval to clear zero. The upper bound landed exactly on zero. The result therefore failed by the smallest possible margin, while every point estimate still pointed in the expected direction.

We did not want to interpret this result from a single experiment. Instead of changing the first registration after seeing the numbers, we registered a second test. We changed the type of decision rule rather than its threshold.

With twelve checkpoints from six training seeds, a significance test would fail about two times in three even if the effect were real. We therefore report the interval, but base the decision on point estimates fixed in advance. The registration states this explicitly and also states that we would not call the result significant regardless of the outcome.

## Protocol

We evaluated twelve milestones from episodes 6,000 and 8,000, plus the untrained reference for context, on 250 fresh worlds neither variant had played before.

We used two cells: guard off and both vetoes on. Each cell contains 3,250 games.

## Results

Guard on minus guard off, paired by checkpoint and world:

| metric          | difference | 95% interval     |
| --------------- | ---------: | ---------------- |
| score           |     +0.018 | [-0.157, +0.196] |
| self-kills      |     +0.003 | [-0.038, +0.043] |
| survived        |     +0.035 | [-0.014, +0.085] |
| kills           |     -0.007 | [-0.035, +0.020] |
| coins           |     +0.053 | [-0.047, +0.155] |
| invalid actions |     +0.057 | [-0.101, +0.215] |

Six of twelve checkpoints improved on score.

Registered decision:

* score point estimate +0.018 is at least 0: PASS
* score lower bound -0.157 is above -0.15: FAIL
* self-kill point estimate +0.003 is at most 0: FAIL

## What happened

The effect disappeared. Score dropped from +0.184 to +0.018, while the self-kill reduction changed from -0.044 to +0.003, slightly in the wrong direction. Six of twelve checkpoints improved, giving an even split.

Nothing about the guard changed between the two runs. The worlds and checkpoints changed.

The mechanism still works, but our trained agents do not seem to need it. On the untrained reference, the same guard improves score by +1.004 [+0.672, +1.336]. This matches the factorial result of +1.19 on its own worlds.

Vetoing provably fatal actions strongly helps an agent that has not learned to survive. By episode 6,000, our agents have learned most of this behaviour themselves, so the veto has little left to prevent.

## Why this matters beyond the guard

This was the clearest methodological result from the week. Our first measurement on one set of worlds showed a plausible, mechanistically sensible, nearly significant effect. The confirmation on fresh worlds did not reproduce it.

If we had shipped based on the factorial alone, we would have shipped a change with almost no measured benefit in the confirmation while believing it was worth about +0.18 score.

We see the same experimental shape with the attack rule. For this reason, we also test it under a separate registration with thresholds fixed before looking at the results.
