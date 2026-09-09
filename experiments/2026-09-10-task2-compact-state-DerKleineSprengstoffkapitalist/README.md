# Tabular Task 2 compact state abstraction

> Status: registered_pending_approval

## Metadata

- Issue: #128
- Agent: `DerKleineSprengstoffkapitalist`
- Owner: LiliWestermann
- Date: 2026-09-10
- Registration base: `8927781`
- Experiment commit: to be recorded after the registration commit
- Framework revision: `0f55c1d`
- Approval: required before training

## Research question

Does a compact, decision-oriented state representation improve state reuse and
hidden-coin collection for the tabular Task 2 agent?

## Hypothesis

Replacing the current 17-feature representation with five compact factual
features will reduce Q-table sparsity and improve `classic` hidden-coin
collection without increasing self-kills.

## Treatments

- Control: current 17-feature Task 2 representation with zero-initialized
  Q-values.
- Candidate: compact five-feature representation with zero-initialized
  Q-values.

Both arms use:

- `action_masking: none`;
- `useful_bomb_reward: 0.0`;
- fresh empty Q-tables;
- no Task 1 or Task 2 parent Q-values.

The completed Issue #102 model remains a historical reference only.

## Compact state

```python
state = (
    danger_level,
    safe_directions_mask,
    coin_direction,
    crate_direction,
    bomb_status,
)
```

### Feature semantics

`danger_level`:

- `0`: the current tile is outside all predicted blast intervals;
- `1`: the earliest predicted lethal interval starts more than two actions
  later;
- `2`: the tile is currently lethal or becomes lethal within two actions.

`safe_directions_mask`:

- uses four bits in `UP`, `RIGHT`, `DOWN`, `LEFT` order;
- a bit is set only if that movement is currently traversable and begins a
  complete time-aware route to a tile outside all predicted blast intervals;
- it is an input feature only and does not filter action selection or Bellman
  targets.

`coin_direction`:

- `0` means no reachable visible coin;
- `1–4` encode `UP`, `RIGHT`, `DOWN`, `LEFT`;
- the direction is the deterministic first BFS step to the nearest reachable
  visible coin.

`crate_direction`:

- `0` means no reachable bombing tile;
- `1–4` encode `UP`, `RIGHT`, `DOWN`, `LEFT`;
- the direction is the deterministic first BFS step to the nearest reachable
  tile from which a bomb could destroy a crate.

`bomb_status`:

- `0`: no bomb is available;
- `1`: a bomb is available but would not destroy a crate;
- `2`: a bomb would destroy a crate, but no complete safe escape exists;
- `3`: a bomb would destroy a crate and a complete safe escape exists.

The theoretical upper bound is 4,800 states. The feature extractor supplies
factual state properties only. It does not select an objective, prescribe an
action, or override the epsilon-greedy Q-learning policy.

## Controlled variables

The following remain identical between arms:

- tabular one-step Q-learning update;
- learning rate `0.05`;
- discount factor `0.9`;
- epsilon schedule `1.0`, multiplied by `0.99` per episode to a minimum of
  `0.1`;
- reward configuration;
- action-masking mode;
- zero initialization;
- training curriculum and episode budget;
- training and evaluation seeds;
- scenarios and opponents;
- final-checkpoint selection;
- evaluation suites.

## Training protocol

Train five independent replicas per treatment:

1. 2,000 `coin-heaven` episodes;
2. 2,000 `loot-crate` episodes;
3. 6,000 `classic` episodes.

Only the checkpoint after exactly 10,000 episodes is evaluated. Both treatments
therefore use 50,000 training episodes.

## Evaluation protocol

Evaluate every replica on 40 paired development seeds in:

- `classic`;
- `coin-heaven`;
- `loot-crate`.

Each suite is repeated with the identical world and agent seeds to verify
determinism. There are 1,200 evaluation episodes per treatment and 2,400 in
total. Confirmation seeds remain unused.

## Primary metric

Candidate-minus-control coin collection fraction on `classic`, paired by
replica and world seed.

## Learning-efficiency diagnostics

Record per replica:

- number of materialized Q-table states;
- total state visits;
- mean visits per materialized state;
- fraction of states visited exactly once;
- fraction of evaluation decisions using previously unseen states;
- final Q-table artifact size.

## Decision rule

Accept the candidate only if:

- the mean paired `classic` collection difference is above zero;
- the paired 95% bootstrap CI lower bound is above zero;
- at least four of five replicas improve;
- the evaluation unseen-state rate is below control;
- mean visits per materialized state exceed control;
- aggregate `classic` self-kill rate does not exceed control plus `0.02`;
- the candidate-minus-control `coin-heaven` retention CI lower bound is above
  `-0.05`;
- all deterministic repeats match;
- decision-time p95 remains below `50 ms`;
- maximum decision time remains below `100 ms`.

## Compute budget

- maximum training episodes: 100,000;
- maximum evaluation episodes: 2,480;
- maximum CPU time: 36 CPU-hours;
- paid resources: none.

## Registered run plans

```bash
python -m training.run_plan \
  training/run_plans/issue128-tabular-task2-baseline-state.yaml \
  --dry-run

python -m training.run_plan \
  training/run_plans/issue128-tabular-task2-compact-state.yaml \
  --dry-run
```

No training may begin until the protocol commit and both dry runs have been
reviewed and personally approved by another team member.

## Results

Not run.

## Decision

Pending.

