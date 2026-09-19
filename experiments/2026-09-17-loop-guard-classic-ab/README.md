# Loop guard, A/B on fresh classic worlds

Decision: DO NOT SHIP. All three registered checks failed. Bomb-omb keeps the policy's own choices.

We ran the experiment on `monitoring/final-training-evaluation` under `monitoring/loop-guard-classic-ab/20260917-1411/` at commit `81e1928`. The verdict is stored in `analysis.json`.

## Why we built it

Our trained Bomb-omb policies tend to move between two or three tiles once nearby movement no longer gives a reward. On an empty board, this loses most of the available coins. We also see the same pacing in tournament games after the opponents are dead.

The guard only intervenes when the agent is stuck, meaning the last eight positions cover at most three tiles, the board is calm, meaning no bomb or explosion is within five tiles, and the policy's selected move would return to a recently visited tile.

In this case, we take the move the policy itself values most among legal moves for which its own danger model predicts a surviving continuation. The guard never selects WAIT or BOMB, and we never activate it during training.

We already knew the guard helped enormously on an empty board. The question was whether this improvement came with any cost in the tournament setting.

## Protocol

We tested six episode-4,000 checkpoints and the unchanged reference on 100 fresh classic worlds against three `rule_based_agent`s.

We compared guarded against unguarded behaviour, paired by checkpoint and world. The experiment contained 1,400 classic games plus 560 coin-heaven games as a descriptive secondary analysis. We fixed the thresholds before execution.

## Results

Guarded minus unguarded, using a hierarchical paired bootstrap over checkpoints and then worlds:

| metric          | difference | 95% interval     |
| --------------- | ---------: | ---------------- |
| score           |     -0.102 | [-0.497, +0.270] |
| self-kills      |     +0.050 | [-0.042, +0.137] |
| survived        |     -0.002 | [-0.077, +0.073] |
| coins           |     -0.002 | [-0.185, +0.182] |
| kills           |     -0.020 | [-0.082, +0.040] |
| invalid actions |     +0.492 | [+0.150, +0.880] |

All three registered checks failed:

* score lower bound -0.497 is not above -0.30
* score point estimate -0.102 is not at least 0
* self-kill upper bound +0.137 is not at most +0.10

The only interval that clears zero is invalid actions, and it does so in the wrong direction. Breaking a loop means moving toward a tile an opponent might occupy first.

## The finding that outlived the verdict

On coin-heaven, the guard has a large effect. Collection fraction roughly doubles for most checkpoints. `halved-r1` increases from 7.0 to 32.2 coins, `control-r2` from 17.8 to 38.6, and the unchanged reference from 28.6 to 46.7.

The guard therefore does what we designed it to do.

The improvement does not translate into tournament score. Solo coin collection and classic score are not the same skill, and improving the first did not improve the second.

This gave us the first clear evidence that our opponent-free retention suites should not replace the tournament suite when deciding what to ship.

## What we learned later

Two days after this test, we measured why the agent paces. This changed how we interpret the result.

Stalling is 4.5 times more likely once every crate is destroyed: 16.9% of steps while crates remain and 76.3% after all crates are gone. Of 1,272 stalled steps, 828 occurred with no crates, no visible coins, and an opponent still alive.

In this state, moving somewhere else has the same value as moving back. A guard that only breaks the cycle therefore does not solve the underlying problem.

The missing behaviour is not simply to stop looping. The agent needs to actively pursue another objective. The attack rule attempts to provide this behaviour.

See [`scripts/diagnose_late_game_stalls.py`](../../scripts/diagnose_late_game_stalls.py) and the [attack rule](../2026-09-18-attack-rule/config.json).
