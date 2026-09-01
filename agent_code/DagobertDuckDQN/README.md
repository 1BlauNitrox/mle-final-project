# DagobertDuckDQN

> Status: Task 1 development baseline evaluated in issue #41. The result is
> mixed/negative and no candidate is frozen.

## Hypothesis

A small Deep Q-Network using the same eight input features, five actions, and
initial reward mapping as the tabular Task 1 agent can learn visible-coin
navigation in `coin-heaven`.

Issue #41 evaluated this hypothesis scientifically. Aggregate navigation met
its threshold, but reproducibility and invalid-action thresholds failed.

## Scope

The agent is limited to visible-coin navigation in `coin-heaven`.

Its ordered action space is:

```text
UP, RIGHT, DOWN, LEFT, WAIT
```

`BOMB` is excluded by construction during training and evaluation. Crates,
bomb survival, opponents, self-play, and tournament optimization are outside
the current scope.

The agent is a separate learned model family. It does not import runtime code
from `DerKleineVermoegensumverteiler` or any other agent.

## Learning method

The agent implements a standard Deep Q-Network with:

- a feed-forward online network;
- a separate frozen target network;
- uniform experience replay with bounded capacity;
- mini-batch temporal-difference updates;
- hard target-network synchronization;
- epsilon-greedy training exploration;
- seeded exploration, sampling, initialization, and tie-breaking;
- Adam optimization;
- Huber loss; and
- gradient clipping.

For an ordinary transition, the target is:

```text
y = reward + gamma * max_a Q_target(next_state, a)
```

For a terminal transition:

```text
y = reward
```

Terminal transitions never bootstrap.

The online network minimizes the Huber loss between `Q_online(state, action)`
and the fixed target `y`. Gradients never flow through the target network.

Reference:

- Mnih et al. (2015), “Human-level control through deep reinforcement
  learning,” Nature 518, 529–533,
  https://doi.org/10.1038/nature14236

## Network architecture

The CPU network is:

```text
8 inputs
-> Linear(8, 64)
-> ReLU
-> Linear(64, 64)
-> ReLU
-> Linear(64, 5)
```

The five outputs correspond exactly to the documented action order. No output
activation is used because Q-values are not probabilities.

## State representation

The raw state representation duplicates the controlled eight-feature contract
of the tabular agent locally:

```text
(
    free_up,
    free_right,
    free_down,
    free_left,
    coin_visible,
    coin_dx,
    coin_dy,
    coin_distance_bin,
)
```

The four movement flags are `1` when the adjacent tile is traversable and `0`
when it is blocked by the field, a bomb, or another agent.

The nearest visible coin is selected by Manhattan distance. Equal distances are
resolved deterministically by coordinate. `coin_dx` and `coin_dy` contain only
`-1`, `0`, or `1`.

The distance bin is:

- `0`: no visible coin;
- `1`: distance one;
- `2`: distance two or three;
- `3`: distance four or greater.

The encoder performs no pathfinding and does not encode an optimal action.

### Input normalization

The network receives `float32` values. Movement flags, coin visibility, and
signed directions already lie in `[-1, 1]` and remain unchanged. The distance
bin is divided by three:

```text
normalized_distance = coin_distance_bin / 3
```

Feature order is unchanged by normalization.

## Rewards

The initial reward mapping is identical to the tabular comparison agent:

| Event | Reward |
| --- | ---: |
| `COIN_COLLECTED` | `+10.0` |
| `INVALID_ACTION` | `-0.5` |
| `WAITED` | `-0.1` |

All unlisted events contribute zero reward. No distance-based shaping is used.

## Experience replay

The replay buffer:

- stores at most 10,000 transitions;
- discards the oldest transition when full;
- copies inserted states defensively;
- samples uniformly without replacement;
- uses a dedicated seeded NumPy random generator;
- distinguishes terminal and non-terminal transitions explicitly; and
- stores no successor state for a terminal transition.

Training starts after 256 stored transitions and uses batches of 64. At most
one optimizer update is performed for each finalized environment transition.

## Exploration

Training uses epsilon-greedy exploration:

| Parameter | Value |
| --- | ---: |
| Initial epsilon | `1.0` |
| Episode decay | `0.99` |
| Minimum epsilon | `0.1` |

Evaluation always uses epsilon zero. Greedy ties are resolved using the
explicitly seeded action RNG.

Action and replay RNG streams are derived independently from
`BOMBERMAN_AGENT_SEED`.

## Hyperparameters

| Parameter | Value |
| --- | ---: |
| Hidden layers | `64, 64` |
| Learning rate | `0.001` |
| Discount factor | `0.9` |
| Batch size | `64` |
| Replay capacity | `10,000` |
| Replay warm-up | `256` transitions |
| Target synchronization | every `250` optimizer updates |
| Gradient clipping | maximum norm `10.0` |
| Optimizer | Adam |
| Loss | Huber / smooth L1 |
| PyTorch CPU threads | `1` |
| Default agent seed | `0` |

These values are implementation defaults. They have not been optimized and are
not evidence of performance.

## Framework transition handling

The framework may expose the final surviving transition once through
`game_events_occurred()` and again through `end_of_round()`.

The agent therefore keeps the newest surviving transition pending:

- an older pending transition is finalized as ordinary when the next callback
  arrives;
- a matching final surviving transition is finalized exactly once as terminal;
- when the agent dies, the previous pending transition is finalized as
  ordinary and the death-causing action is finalized separately as terminal.

This prevents duplicate learning and terminal bootstrapping.

## Checkpoint

The checkpoint is stored at:

```text
agent_code/DagobertDuckDQN/checkpoint.pt
```

The path is resolved relative to the agent module.

The schema-validated checkpoint contains:

- online-network parameters;
- target-network parameters;
- Adam optimizer state;
- optimizer update count;
- bounded replay contents;
- replay-sampling RNG state;
- action-selection RNG state;
- epsilon and completed-episode count;
- agent seed;
- architecture and hyperparameters;
- feature, model, and checkpoint schema versions;
- action order; and
- reward configuration.

Writes use a temporary file in the same directory, followed by `fsync` and
atomic replacement. A failed write leaves an existing checkpoint unchanged.

Evaluation loads only a frozen online policy, disables gradients and
exploration, and does not write the checkpoint.

No candidate checkpoint is committed. The five development artifacts remain in
the ignored raw experiment store and must not be mistaken for a frozen model.

## Training

A short integration run can be started with:

```bash
python -m training.run_experiment \
  --agent DagobertDuckDQN \
  --mode training \
  --scenario coin-heaven \
  --rounds 2 \
  --world-seed 1001 \
  --agent-seed 2001
```

Training uses no opponents in the current scope.

## Evaluation

After training has produced a checkpoint:

```bash
python -m training.run_experiment \
  --agent DagobertDuckDQN \
  --mode evaluation \
  --scenario coin-heaven \
  --rounds 2 \
  --world-seed 3001 \
  --agent-seed 2001
```

Evaluation is CPU-only, uses one PyTorch thread, performs no learning, creates
no replay buffer or optimizer, and must leave `checkpoint.pt` byte-for-byte
unchanged.

## Learning metrics

The training callback returns the metrics supported by the repository
experiment pipeline:

- cumulative shaped reward;
- epsilon used during the completed episode; and
- replay-buffer size;
- cumulative optimizer-update count;
- mean Huber loss and mean absolute TD error when at least one optimizer update
  occurred; and
- per-episode and cumulative target-network synchronization counts.

An episode without an optimizer update reports the TD-error metric as
unavailable rather than zero.

## Dependencies

- Python 3.13
- NumPy
- PyTorch `>=2.13,<2.14`

The agent-local `requirements.txt` declares the additional evaluation
dependency.

## Validation status

Implemented unit and contract tests cover:

- action order and exclusion of `BOMB`;
- raw features and normalization;
- reward equivalence;
- network tensor shapes and deterministic initialization;
- Bellman targets, including terminal masking;
- optimizer updates and hard target synchronization;
- bounded and seeded replay;
- exact learner, optimizer, replay, and RNG round trips;
- schema validation and atomic checkpoint failure behavior;
- deterministic action selection and tie-breaking;
- frozen, read-only evaluation;
- framework callback initialization; and
- ordinary, surviving-terminal, and death-terminal transitions.

The following integration checks were run on 2026-08-28 on Windows with an
Intel Core i7-8550U CPU, Python 3.13.15, and CPU-only PyTorch 2.13.0:

- a two-round seeded training smoke run produced a non-empty checkpoint and
  reported a mean absolute TD error;
- a one-round resumed training run continued at epsilon `0.9801`, changed the
  checkpoint, and increased its stored replay data;
- evaluation reported no learning metrics and left the checkpoint hash and
  modification time unchanged;
- two repeated two-round evaluations with world seed `5001` and agent seed
  `2001` produced identical per-episode actions, scores, and termination data;
- a ten-round evaluation with world seed `6001` measured a mean episode p95
  decision time of approximately `0.73 ms` and a maximum of approximately
  `6.02 ms`, below the issue gates of `50 ms` and `100 ms` respectively;
- evaluation setup configured PyTorch to use exactly one CPU thread;
- a 20-round evaluation with world seed `6003` measured approximately
  `213.64 MiB` peak working-set memory across the Windows virtual-environment
  launcher and runtime process, below the `8 GiB` limit, and left the
  checkpoint unchanged;
- `scripts/package_agent.py` produced an archive containing only the agent
  directory and excluding caches, logs, tests, and other agents; and
- the packaged agent completed a one-round headless evaluation from a clean
  `git archive` export of `main` with exit code zero;
- GitHub Actions run `33416810178` built the supplied Dockerfile from the pinned
  Miniconda Python 3.13 base, installed only the agent-local runtime requirement
  into the test image, and validated PyTorch `2.13.0`;
- inside that image, a seeded two-round smoke training run produced the package
  checkpoint, and a training-disabled evaluation against three `random_agent`
  instances preserved its SHA-256 checksum;
- the packaged agent was extracted into a clean framework export and completed
  another evaluation against three `random_agent` instances without modifying
  its checkpoint; and
- the Docker evaluation used Python `3.13.9`, PyTorch `2.13.0+cu130`, and one
  PyTorch CPU thread on Linux. Its mean episode p95 decision time was about
  `0.171 ms`, and its maximum was about `0.601 ms`.

These are implementation smoke checks, not a scientific performance
evaluation. The temporary checkpoint and raw run directories are not intended
for version control. The Docker checks validate dependency, packaging, and
runtime compatibility; they do not replace the final submission-stage test with
the selected trained artifact in the course-provided environment.

## Scientific evaluation

Issue #41 trained five independent 10,000-episode models and evaluated each on
development seeds `31001`--`31040`. The aggregate coin-collection fraction was
`0.8334`, but only three models reached the individual `0.75` threshold.
Aggregate invalid-action rate was `0.1642`, dominated by run 4 (`0.6085`). Run
3 collected only `0.4830` of available coins on average and selected `WAIT`
6,487 times. Across all models, `BOMB` was never selected. Exact repeats were
deterministic, artifacts stayed byte-identical, maximum model p95 decision time
was `0.752 ms`, and maximum observed decision time was `27.601 ms`.

The paired DQN-minus-tabular bootstrap could not be computed because PR #37
does not retain per-seed tabular rows and its original artifacts are not
available. Its reported aggregate fraction is `0.8995`, versus `0.8334` here;
that descriptive difference is not a paired confidence interval. Full
configuration, hashes, failures, figures, and the negative decision are in
`experiments/2026-09-01-dqn-task1-development-baseline/`.

## Known limitations and next steps

- The eight-feature representation contains no pathfinding or global maze map.
- Reward shaping is intentionally minimal.
- Hyperparameters have not been tuned; run-to-run stability is inadequate.
- The policy can repeatedly choose invalid moves because legal actions are not
  masked; this caused the registered invalid-action criterion to fail.
- The agent handles only visible-coin navigation.
- No candidate should be frozen from issue #41.
- Final submission compatibility must be repeated with the selected trained
  artifact in the course-provided environment.
- The next experiment should prospectively test one controlled change aimed at
  invalid actions and instability, such as legal-action masking.
