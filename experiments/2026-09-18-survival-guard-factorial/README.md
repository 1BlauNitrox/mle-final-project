# Survival guard, 2x2 factorial on fresh classic worlds

Status: DO NOT IMPLEMENT. The registered rule failed. Bomb-omb stays exactly as trained.

We registered a follow-up in `experiments/2026-09-18-survival-guard-confirmation/`. The follow-up uses a decision rule instead of repeating the same test.

Results are in `monitoring/survival-guard-factorial/` on `monitoring/final-training-evaluation` at commit `3f7f7fe`. All four cells completed, with 9,120 games in total.

## Why we ran it

Bomb-omb kills itself in about half of all classic rounds. This rate had not changed since roughly episode 1,500 of the final run, while coins and crates continued to improve.

The agent already receives relevant safety information. `features/assemble.py` gives the network five flags for whether a move has a surviving continuation, plus one flag for whether placing a bomb still leaves an escape. `legality.py` only masks framework legality, as stated in its docstring. We do not otherwise enforce this safety information.

We therefore built a play-time guard to enforce it. The guard never invents a move. We go down the policy's own Q-ranking and select the highest-ranked legal action which the agent's danger model considers survivable. If no survivable action exists, we keep the policy's original choice.

We use two switches to separate the two ways of dying. This gives us a 2x2 design with one shared control:

| cell    | bomb veto | move veto |
| ------- | --------- | --------- |
| control | off       | off       |
| both    | on        | on        |
| bomb    | on        | off       |
| move    | off       | on        |

The guard only affects inference. We do not retrain the agent or interact with the running training process. We would apply the guard to whichever checkpoint we select on Sunday.

## Results

We evaluated 18 milestones, episodes 2,000, 4,000, and 6,000 from all six jobs, on 120 fresh worlds. Each cell contains 2,280 games.

Absolute results across the evaluated checkpoints:

|            | control | both  | bomb  | move  |
| ---------- | ------- | ----- | ----- | ----- |
| score      | 2.976   | 3.160 | 3.042 | 3.072 |
| self-kills | 0.530   | 0.486 | 0.494 | 0.506 |
| survived   | 0.379   | 0.450 | 0.403 | 0.420 |
| coins      | 2.457   | 2.579 | 2.488 | 2.526 |
| invalid    | 1.259   | 1.420 | 1.242 | 1.376 |

Guard on minus guard off, paired per checkpoint and world, with a hierarchical bootstrap over checkpoints and then worlds:

| metric     | difference | 95% interval       |
| ---------- | ---------: | ------------------ |
| score      |     +0.184 | [-0.021, +0.381]   |
| self-kills |     -0.044 | [-0.087, +0.00000] |
| survived   |     +0.071 | [+0.024, +0.119]   |
| coins      |     +0.121 | [+0.013, +0.229]   |
| invalid    |     +0.161 | [-0.048, +0.357]   |

## The decision

Our registration required three conditions. The first failed:

* self-kill interval upper bound below 0: FAIL, exactly +0.00000
* score interval lower bound above -0.15: PASS, -0.021
* score point estimate at least 0: PASS, +0.184

97.4% of the bootstrap draws for self-kills are below zero. The result landed directly on the registered boundary, but the rule still gives us a failed result.

We defined this rule before seeing the data so we would not change the decision afterwards. We therefore do not ship the guard based on this evidence.

## What we learned anyway

Nine out of ten self-kills are invisible one step ahead. A hard veto on every provably unsurvivable action only reduces self-kills from 0.530 to 0.486.

Our own danger model searches ten steps while treating opponents as a snapshot. It therefore misses most deaths before they happen. This changes how we understand the problem. The remaining suicides are multi-step planning failures, which a one-step filter does not address.

The two factors add up. The bomb veto alone gives -0.036 self-kills, the move veto alone gives -0.024, and both together give -0.044. This gives an interaction of +0.016. Neither individual factor is detectable on its own. We see no antagonism between them and no reason to prefer one.

The benefit did not fade as the agent learned. Score difference by episode was +0.085 at 2,000, +0.254 at 4,000, and +0.212 at 6,000. We expected the guard might become redundant as the network learned safety on its own. We did not observe this, which makes testing a late checkpoint worthwhile.

The effect also varies substantially between games. Outcomes changed in 47% of games. The guarded version performed better in 25.7% and worse in 21.3%. Fifteen of 18 checkpoints improved on score. One checkpoint, `control-r3@4000`, became clearly worse at -0.617.

The guard also increases invalid actions by +0.161 per game. The replacement action often moves into a tile an opponent takes first. Every invalid action we classified in this project came from a contested tile. We therefore treat this as a tempo cost rather than an illegal-move bug, but the cost is still present.

On the untrained reference, the guard improves score by +1.19 [+0.73, +1.68]. This result only provides context and did not enter our decision. It confirms the mechanism works and shows how much of this safety our trained agents already learned without the guard.

## What we would do differently

Our design spent experimental power on separating the factors, while the main constraint was the number of checkpoints.

The guard effect varies more between checkpoints, sd 0.267, than its mean effect of 0.184. With six training seeds, adding more worlds does not solve this limitation.

If we repeated the experiment, we would train more replicas instead of evaluating more boards. This applies to many of the experiments we ran this week.

## Reproducing

```text
python scripts/analyze_survival_guard_factorial.py --root monitoring/survival-guard-factorial
```

We produced the cells with `scripts/evaluate_milestones.py`, using `--agent Bomb-omb-survivalguard` and the two guard switches, with one output folder per cell.

Every row contains its variant label. The evaluator refuses to mix labels within one folder, so a mislabelled cell does not stay hidden. `run.json` records the exact environment for each cell.
