# DagobertDuckDQN

> Status: frozen Task 1 baseline (issue #42). Selected mechanically from the
> issue #58 series, which itself failed its own registered primary criterion.
> This is a development-selected reference for Task 2 regression, not a
> passing Task 1 result. Training is disabled; use a separately named
> successor for Task 2.

## Hypothesis

A small Deep Q-Network using the same eight input features, five actions, and
initial reward mapping as the tabular Task 1 agent can learn visible-coin
navigation in `coin-heaven`.

Issue #41 evaluated this hypothesis and found the reproducibility and
invalid-action thresholds failing. Issue #58 tested whether porting the
tabular agent's movement-coin reward shaping would resolve the instability; it
did not (see "Scientific evaluation" below). Per the stopping rule registered
in #58 ("no third Task 1 DQN tuning experiment"), a candidate was then
selected mechanically from #58's series rather than run further, and frozen
here under issue #42.

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

The frozen reward mapping, including the movement-coin shaping tested in
issue #58:

| Event | Reward |
| --- | ---: |
| `COIN_COLLECTED` | `+10.0` |
| `INVALID_ACTION` | `-0.5` |
| `WAITED` | `-0.1` |
| `MOVED_TOWARDS_COIN` | `+0.1` |
| `MOVED_AWAY_FROM_COIN` | `-0.1` |

All unlisted events contribute zero reward. The movement-coin shaping is not
potential-based; see issue #58 for the accepted deviation from policy
invariance.

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

`persistence.py` defines a single evaluation-only artifact schema, not the
resumable training schema a live DQN run would need. The schema-validated
checkpoint contains only:

- online-network parameters;
- completed-episode count;
- architecture and hyperparameters;
- feature, model, and artifact schema versions;
- action order; and
- reward configuration.

It deliberately excludes the target network, optimizer state, replay
contents, replay and action-selection RNG state, epsilon, and agent seed --
everything a resumable training checkpoint would additionally carry. An
earlier version of this artifact kept those fields reset to empty rather than
omitting them, which left the committed file structurally a resumable
checkpoint; issue #42 requires the committed baseline to contain only the
evaluation state required at runtime, so this module has no code path left
that can read or write that resumable shape at all.

Writes use a temporary file in the same directory, followed by `fsync` and
atomic replacement. A failed write leaves an existing checkpoint unchanged.

Evaluation loads only the frozen online policy, disables gradients, and does
not write the checkpoint. `setup_training` (`train.py`) rejects training
unconditionally, before any of this runs, regardless of `self.train`.

Repository evaluation tooling may set `BOMBERMAN_EVALUATION_CHECKPOINT` to one
file name. The agent resolves that name inside this directory; path components
are rejected. This permits an ignored, temporary artifact copy during a
registered evaluation without changing the default submission artifact
`checkpoint.pt` or relying on an absolute path.

**The committed `checkpoint.pt` is the frozen artifact selected under issue
#42.** It was produced by
[`scripts/freeze_dagobertduckdqn_task1_baseline.py`](../../scripts/freeze_dagobertduckdqn_task1_baseline.py)
from run-02's full resumable checkpoint (issue #58): the script verifies that
checkpoint's SHA-256 and byte size against values recorded before export,
refusing to run against any other file, then extracts and exports only the
online network's weights -- the target network, optimizer, replay buffer, and
RNG state are not carried into the committed artifact at all, rather than
being carried and reset. A self-check compares Q-values from the exported
artifact against the source network before the script reports success.
`agent_code/DagobertDuckDQN/artifact.json` records the frozen artifact's own
checksum and lineage, plus the source checkpoint's checksum, size, original
local path, immutable release tag, asset name, and direct download URL.

### Independent freeze reproduction

The exact 862,842-byte source checkpoint is retained outside Git as the
[`run-02-final-checkpoint.pt`](https://github.com/1BlauNitrox/mle-final-project/releases/download/issue-58-dqn-task1-run-02-checkpoint-v1/run-02-final-checkpoint.pt)
asset of release
[`issue-58-dqn-task1-run-02-checkpoint-v1`](https://github.com/1BlauNitrox/mle-final-project/releases/tag/issue-58-dqn-task1-run-02-checkpoint-v1).
Its SHA-256 is
`1d8f2bc9c33d775b59595f0b5ae0978078e2f2fe3571a9a1299748a042857924`.

A reviewer can reproduce the source verification and evaluation-only export
without access to the producing machine:

```bash
gh release download issue-58-dqn-task1-run-02-checkpoint-v1 \
  --repo 1BlauNitrox/mle-final-project \
  --pattern run-02-final-checkpoint.pt \
  --dir <temporary-directory>
python -m scripts.freeze_dagobertduckdqn_task1_baseline \
  --source <temporary-directory>/run-02-final-checkpoint.pt \
  --output <temporary-directory>/frozen-checkpoint.pt
```

The exporter first enforces the recorded source byte size and SHA-256, then
loads the restricted source schema and compares the exported network's Q-values
with the source network. A successful run reports a 23,829-byte frozen artifact
with SHA-256
`eb08e3f67b620ac2a253a2af4db3d5b4c6ea9e667a2aaf1d91e3fccf4ba8b05e`,
which is the committed `checkpoint.pt`.

## Training

Training is disabled. `setup_training` raises `RuntimeError` unconditionally;
see `train.py`. Task 2 development must use a separately named successor.

## Evaluation

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
- CPU-only PyTorch `2.13.0`

The agent-local `requirements.txt` selects the official PyTorch CPU wheel index.
Linux and Windows use the explicitly CPU-tagged `2.13.0+cpu` build; macOS uses
the CPU-only `2.13.0` build published for that platform.

## Validation status

Implemented unit and contract tests cover:

- action order and exclusion of `BOMB`;
- raw features and normalization;
- reward equivalence;
- network tensor shapes and deterministic initialization;
- Bellman targets, including terminal masking;
- optimizer updates and hard target synchronization;
- bounded and seeded replay;
- exact evaluation-only network and metadata round trips;
- schema validation and atomic checkpoint failure behavior;
- source-checkpoint checksum and size verification, including rejection of a
  mismatched or missing source;
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
6,487 times. Across all models, `BOMB` was never selected. Repeated outcomes and
action totals matched, artifacts stayed byte-identical, maximum model p95
decision time was `0.752 ms`, and maximum observed decision time was
`27.601 ms`. Ordered action sequences were not recorded in this original run,
so its determinism gate remains unverified under the strengthened check from
PR #59.

Issue #58 then ported the tabular agent's movement-coin reward shaping
verbatim, on the hypothesis that a dense directional signal would remove the
run-to-run divergence that stopped #41 from freezing a candidate. It trained
five more independent 10,000-episode models on world seeds `15001`--`15005`
and evaluated them on the same development seeds. The result was negative:
the primary criterion failed (aggregate invalid-action rate `0.1758`, maximum
per-model `0.4617`, both above the `0.01` bar) and the registered paired
non-inferiority comparison against #41 was inconclusive rather than
supportive (mean difference `-0.018`, 95% CI `[-0.2471, +0.2414]` against a
`-0.02` margin) -- the between-run variance is larger than any plausible
effect of the shaping. `WAIT` selections fell from 6,505 to 2,676 and the
worst per-model invalid rate fell from `0.61` to `0.46`, but full clears fell
from 125 to 92: models that previously failed by waiting now fail by walking
into walls instead. Training itself reached `0.989`--`1.000` mean collection
fraction (epsilon `0.1` throughout), while deterministic greedy evaluation of
the same five checkpoints gave only `0.65`--`0.92` -- forced exploration
during training was masking a greedy action-selection failure, not a learning
failure. Full configuration, evidence, figures, and the negative decision are
in `experiments/2026-09-02-dqn-task1-movement-shaping/`; the record is in
PR #66.

Per the stopping rule registered in #58, no third Task 1 DQN tuning experiment
was run. Instead, issue #42 registered a neutral, mechanical candidate-selection
rule prospectively and applied it to #58's five models: `run-02`, the median
performer by primary evaluation fraction (`0.8165`), which also happens to
carry the cleanest invalid-action rate of the five (`0.000315`). That model is
the `checkpoint.pt` committed in this directory. Selecting it does not mean it
passed Task 1's own bar -- it did not -- only that it is the model this
repository's process identifies as the Task 2 reference point.

The frozen DQN-versus-tabular candidate comparison is tracked separately in
issue #53.

## Known limitations and next steps

- The eight-feature representation contains no pathfinding or global maze map.
- Hyperparameters have not been tuned against the invalid-action and
  reproducibility failures beyond the single shaping intervention in #58.
- The frozen policy can repeatedly choose invalid moves under greedy
  evaluation because legal actions are not masked; #58 traced this to a gap
  between training's forced exploration and evaluation's pure-greedy policy,
  not to insufficient learning.
- The agent handles only visible-coin navigation.
- Final submission compatibility must be repeated with the selected trained
  artifact in the course-provided environment.
- Legal-action masking is deferred to Task 2, where a blocked move next to a
  live bomb is fatal rather than merely wasteful.
