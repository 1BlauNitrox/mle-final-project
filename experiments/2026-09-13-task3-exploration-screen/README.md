# Task 3: exploration timing and a learning pilot

**Decision: the learning pilot failed its registered screen. No checkpoint is
promoted.** Episode-based exploration supplied much more useful training
experience than stepwise exploration, but did not establish better hunting
and did not preserve all earlier capabilities. Task 3 remains exploratory.

## Why we tested this

The #163 control recorded only two eliminations during 25,000 training episodes.
We first tested the same frozen policy without updating its weights:

* Phase A: reducing per-action randomness from 100% to 20% increased survival
from 8.15 to 24.05 steps, but both settings self-killed in all 20 worlds and
reached no hunting opportunities. The exposure filter failed; phase B was not run.
* Phase C, on 20 fresh paired worlds: 80% greedy / 20% fully random episodes
increased survival from 23.8 to 274.65 steps versus stepwise 20% randomness.
Self-kills fell from 100% to 35%; five mixture episodes reached hunting
opportunities. All five exposure checks passed. These were frozen-policy
diagnostics, not evidence of learning.

## Learning experiment

Phase D compared those two exploration schedules with unchanged DQN rewards,
features and learning settings. Both began with identical weights from the
same provisional #91-derived Task 3 initialization. Three paired replicas
trained for 50 episodes per arm: 300 training episodes. All six final
models and the unchanged parent then received 280 greedy evaluation episodes
across peaceful hunting and three earlier-task suites, including exact repeats.

|Greedy evaluation metric|Frozen parent|Stepwise exploration|Episode mixture|
|-|-:|-:|-:|
|Peaceful eliminations/game|0.200|0.133|0.200|
|Peaceful score/game|4.20|2.47|4.53|
|Coin-heaven collection|88.0%|24.5%|71.2%|
|Loot-crate collection|13.2%|10.7%|14.7%|
|Classic without opponents: collection|40.0%|9.6%|32.6%|

Hunting improved by only 0.067 eliminations/game over stepwise exploration,
below the required 0.10. Its descriptive paired 95% interval was
\[-0.400, +0.533]. The mixture matched the parent's mean hunting result;
two of three replicas were no worse than the parent.

|Earlier-task retention against parent|Collection change|Self-kill change|Decision|
|-|-:|-:|-|
|Coin-heaven|-16.80 percentage points|+13.33 points|Both gates fail|
|Loot-crate|+1.47 points|-13.33 points|Both gates pass|
|Classic without opponents|-7.41 points|+26.67 points|Both gates fail|

The registered margins were at most five percentage points of collection loss
and five points of additional self-kills. All six models genuinely learned,
all behavioral repeats matched, and evaluation recorded zero invalid actions.
Decision latency passed: p95 16.66 ms, maximum 36.34 ms.

## Interpretation and follow-up

Mixture training produced 24 eliminations in 150 episodes, versus zero for
stepwise training, and 11,223-13,855 updates per replica versus 406-634.
Equal episode budgets therefore produced very different amounts of experience
and learning. This demonstrates improved training exposure, not better greedy
hunting or equal-compute learning efficiency. Only five independent evaluation
worlds per suite were used; repeats do not double that sample size.

The next proposed pilot holds episode-based exploration fixed and compares
learning rates 0.0005 versus 0.00005, starting again from the unchanged
initialization. It tests whether smaller updates improve retention without
reducing hunting. Excessive update size is a hypothesis, not an established
cause. Do not simply train longer, select the best-looking replica, or proceed
through the failed coin-collector continuation gate.

[Protocol](phase-d-config.json) ? [All metrics, uncertainty and gates](phase-d-analysis.json)
? [Per-model table](phase-d-summary.csv) ? [Evidence and reproduction](phase-d-reproduction.md)
? [Earlier diagnostics](evidence-manifest.json). Refs #168 / PR #170.
