# Team Agent Template

> Status: runnable scaffold, intentionally weak, and not a tournament candidate.

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

<!-- Algorithm, update rule, exploration strategy, and relevant citations. -->

- Incremental sample-average action values
- Epsilon-greedy exploration with epsilon 0.20 during training
- No state representation

## State representation and features

<!-- Explain every feature and its shape/range. -->

Not implemented in the scaffold.

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
