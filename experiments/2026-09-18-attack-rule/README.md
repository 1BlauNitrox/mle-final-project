# Attack rule

## The gap it attacks

Against three `rule_based_agent`s, our episode-8,000 agent gets 0.105 kills per game. They get 0.23 kills against us. Score is coins plus five times kills, and our numbers reproduce this exactly: 2.643 + 5 x 0.105 = 3.168.

The kill deficit therefore costs us 0.63 points. Our coin surplus returns 0.39 points. This difference explains the full -0.232 gap between our agent and the heuristic.

When watching the agent play, we also saw the same pattern directly. The agent is good at collecting coins and consistently wins on this part, but it is bad at hunting.

## Why this is not a seventh reward experiment

Six registered experiments tried to teach the network to attack:

| attempt                                       | budget                      |        result on eliminations |
| --------------------------------------------- | --------------------------- | ----------------------------: |
| elimination reward +5 to +20                  | 5 replicas x 100 episodes   |       +0.010 [-0.080, +0.105] |
| the same with approach shaping                | 10 replicas x 400 episodes  |       +0.005 [-0.085, +0.100] |
| wasteful-bomb penalty removed on safe attacks | 50,000 episodes             |               0 points [0, 0] |
| approach shaping at scale 3.0                 |                             |     more exposure, less score |
| training line-up, hard against mixed          | 6 replicas x 400 episodes   | score -0.075 [-0.729, +0.625] |
| training line-up at the selected rate         | 3 replicas x 2,400 episodes | score -0.250 [-0.975, +0.400] |

The hunting diagnosis explains why none of these worked. A bomb aimed at an opponent with no crate in its blast receives the wasteful-bomb penalty of -0.5 immediately and with certainty. The +5 kill reward arrives four steps later, discounted by gamma^4 to about two thirds, and only if the opponent fails to walk away.

At our measured conversion rate of roughly one kill per hundred attack steps, the expected value of attacking stays negative regardless of the kill reward. We therefore treat this as a credit-assignment problem rather than a knowledge problem.

For this reason, the attack rule sits outside the network. When an opponent stands inside the blast footprint of a bomb placed at our current position, and our own danger model says we still have an escape, we place the bomb.

The rule uses the framework's own blast geometry. A wall protects the opponent, while a crate does not. The rule never fires without a bomb available, never overrides a bomb already selected by the policy, and never fires during training.

## Protocol

We used three cells on 150 fresh audited worlds and the twelve milestones at episodes 6,000 and 8,000, plus the untrained reference for context:

| cell               | attack rule | what it answers                            |
| ------------------ | ----------- | ------------------------------------------ |
| `control`          | off         | the baseline                               |
| `attack`           | on          | the registered contrast: does hunting pay? |
| `attack-selective` | selective   | fires only when the opponent cannot escape |

We spend the evaluation budget on checkpoints rather than more worlds because checkpoint-level variation dominates this type of contrast.

Like the guard confirmation, this uses a decision rule rather than a significance test. The registration states that we will not call the result significant regardless of the outcome. We only ship the rule if it produces kills and those kills pay for themselves in score.

### The third cell changed before it ran

We originally planned the third cell as the attack rule combined with the survival guard. We dropped this question the same evening after the guard confirmation returned +0.018 score and +0.003 self-kills.

We changed the third cell to the selective attack variant instead. The amendment, timestamp, and reason are recorded in `config.json`. Cells one and two were unchanged and continued running.

## What the smoke runs suggested

We ran smoke tests on 25 worlds at episode 8,000. These results do not settle the question on their own. The untouched reference agent ranged from 1.4 to 2.76 score across these runs. We used the smoke tests to rule out failure modes that would waste an evaluation cell.

|                  | score | kills | self-kills | survived |
| ---------------- | ----: | ----: | ---------: | -------: |
| control          |  2.52 |  0.12 |       0.56 |     0.32 |
| attack           |  3.64 |  0.20 |       0.48 |     0.20 |
| attack-selective |  3.28 |  0.16 |       0.44 |     0.44 |

Both variants fired and both produced kills. The permissive variant appeared to pay for them in survival, while the selective variant did not. This is the trade-off the third cell was designed to measure.

## Results

All three cells completed with 1,950 games each and 5,850 games in total. We did not stop any cell early.

The results below are paired over the same checkpoint and world. Intervals are hierarchical paired-bootstrap 95% intervals over checkpoints and worlds.

| Contrast                      |                   Score |                   Kills |              Self-kills |                Survived |                   Coins |
| ----------------------------- | ----------------------: | ----------------------: | ----------------------: | ----------------------: | ----------------------: |
| `attack` − control            | −0.052 [−0.268, +0.156] | +0.021 [−0.013, +0.054] | +0.102 [+0.047, +0.153] | −0.154 [−0.205, −0.101] | −0.157 [−0.263, −0.047] |
| `attack-selective` − control  | +0.006 [−0.226, +0.233] | +0.013 [−0.021, +0.047] | +0.000 [−0.047, +0.046] | −0.019 [−0.060, +0.022] | −0.058 [−0.184, +0.069] |
| `attack-selective` − `attack` | +0.058 [−0.172, +0.292] | −0.008 [−0.045, +0.027] | −0.102 [−0.149, −0.056] | +0.135 [+0.088, +0.182] | +0.099 [−0.041, +0.238] |

The primary `attack` contrast had a positive kill estimate, but failed both score gates. It also clearly increased self-kills, reduced survival, and lost coins.

The selective variant avoided almost all of the safety loss. Its score and kill estimates still did not exclude a meaningful loss, and its lower score bound also missed the registered limit.

## Decision

We do not ship either rule. Bomb-omb remains the learned policy without an attack override.

The experiment shows that current blast geometry alone does not solve hunting. Firing whenever an immediate attack is available causes too many self-kills. Firing more selectively removes most of the safety loss, but also removes the possible gains.

## Reproduction

```text
python scripts/analyze_attack_rule.py --root monitoring/attack-rule
```

The compact per-game observations, paired tables, run metadata, and analysis are tracked on `origin/monitoring/final-training-evaluation` at `5855cb8fd3a620411ee3ee9afe20605c5d4a5191` under `monitoring/attack-rule/`.

The recorded analysis commit is `48261a950bbf1de293b488dd42afb8693c3a7b41`.
