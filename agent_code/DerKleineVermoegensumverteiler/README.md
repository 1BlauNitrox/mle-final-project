# Team Agent Template

> Status: implementations begun, not yet ready for anything

Copy this directory and rename it when an experiment introduces a distinct
learned model:

```bash
cp -R agent_code/_team_agent_template agent_code/<agent_name>
```

On Windows PowerShell:

```powershell
Copy-Item -Recurse agent_code\_team_agent_template agent_code\<agent_name>
```

## Hypothesis

<!-- What testable idea motivates this agent? -->

The scaffold demonstrates that framework training callbacks can update and
persist a learned value. It estimates one mean reward per action and ignores the
game state, so it is useful only as an integration baseline.

## Learning method

The agent uses ag feature-based tabular Q-learning model. Each encoded state maps
to one floating-ponint value for each of the five oredered task 1 actions.

Unseen states are initialized to zero and reading them during evaluation
does not add it to the Q-table.

Training uses the update:

```text
Q(s, a) <- 0(s, a) + alpha *
           (reward + gamma * max Q(s', a') - Q(s, a))
```
Terminal transitions do not bootstrap from a future state, bute use only
the final reward.

Training action selection is epsilon-greedy. Random exploration and greedy
tie-breaking use an explicitly seeded NumPy generator. Evaluation uses
epsilon = 0

- Incremental sample-average action values
- Epsilon-greedy exploration with epsilon 0.20 during training
- No state representation

## State representation and features

The agent maps the game states to an eight_element int tuple:

```Text
(
  up,
  right,
  down,
  left,
  coinVisible,
  coinHorizontal,
  coinVertical,
  coinDistanceBin,
)
```
The four movement features are binary and indicate wether the tile next to 
the agent ist free from walls, crates, bombs or other agents.

coinHorizontal and coinVertical encode the signs of th coordinate difference to the 
nearest visible coin. Each value is -1, 0 or 1.

coinDistanceBin ist the Manhatten-distance:
- 0 when no coin visible
- 1 for distance 1
- 2 for distance 2-3
- 3 for distance >= 4

Ties can be resolved by coordinate. This representation does not perform pathfinding
or encodes a preffered or optimal state. 

## Rewards and custom events

<!-- List values, rationale, and reward-shaping risks. -->

The scaffold uses simple rewards for coins, kills, survival, invalid actions,
self-kills, and deaths. Replace these through controlled experiments.

## Training procedure

<!-- Scenarios, opponents, curriculum, rounds, seeds, hardware, and duration. -->

```bash
python main.py play --my-agent <agent_name> --train 1 --no-gui --n-rounds 100
```

## Model artifact

Training creates `model.npz` beside `callbacks.py`. Record the producing commit,
configuration, and checksum here before treating an artifact as a candidate.

## Evaluation and results

<!-- Link experiment records; include baselines, metrics, seeds, and uncertainty. -->

No scientific evaluation has been performed.

## Dependencies

- NumPy

## Limitations and next steps

- The scaffold ignores `game_state`.
- It cannot learn navigation, bomb safety, or opponent behavior.
- Replace it with a justified learned model; do not relabel this baseline as a
  completed project agent.

## Action contract

actions allowed for task 1 in ordered set:

```Text
UP, RIGHT, DOWN, LEFT, WAIT

BOMB is excluded in tarining and in evalutation
````
