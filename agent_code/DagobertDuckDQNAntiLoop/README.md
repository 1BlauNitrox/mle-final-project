# DagobertDuckDQNAntiLoop agent card

## Hypothesis and scope

Issue [#221](https://github.com/1BlauNitrox/mle-final-project/issues/221) tests whether recent-position information and a small conditional movement penalty reduce persistent loops without sacrificing the frozen fallback's score, survival, or collection capabilities.

The control continues training without these active additions. Both arms use the same widened architecture and initial trained weights.

The changed factor combines history features and reward shaping. A positive result would support this combination. The experiment does not identify which component caused an improvement. This is an exploratory pilot, not a submission freeze.

## Learning algorithm and representation

We use the inherited DQN implementation with a feed-forward 56–64–64–6 network, replay, a target network, Adam, Huber loss, and gradient clipping. The six actions are UP, RIGHT, DOWN, LEFT, WAIT, and BOMB.

The registered run inherits the parent's one-step target, discount 0.9, batch size 64, replay capacity 10,000, and target synchronization interval of 500 updates. The learning rate is 0.0001, with at most one update per four new transitions.

The original 39 inputs retain navigation, bomb/crate, escape-continuation, and opponent features. Twelve geometry inputs inherited from the #211 scaffold remain zero in both arms.

The remaining five inputs describe how often the current tile and its four neighbours appeared among the previous 16 observed positions, divided by 16. These five values are active only in the memory arm.

We reset history between rounds. We cache observations by step so action selection and replay construction use the same feature vector without counting an observation twice. The features describe past visits and do not prescribe an action.

We retain framework legality masking. No anti-loop rule overrides the network's selected action. The feature schema version is 221.

## Rewards and exploration

The inherited reward mapping remains in `config.py`. The memory arm adds a training-only penalty of −0.1 when all of the following conditions hold:

* The action is a successful directional move back to a recently occupied tile.
* The preceding 24 consecutive observations cover at most three positions.
* Those observations and the resulting state contain no bombs or explosions.
* The field, visible coins, agent score, and number of opponents are unchanged.
* The transition reports no coin collection, crate destruction, opponent kill, or invalid action.

BOMB, WAIT, and blocked movement receive no additional loop penalty. We exclude active bomb-escape situations.

Shaping applies with or without crates. The primary loop evaluation specifically measures the crate-free phase.

We retain the additional penalty exactly once when a surviving transition is later finalized as terminal. We add no extra penalty for a death transition without an observed next state.

One seeded episode in every block of five uses uniformly random legal actions. The remaining episodes are greedy. Evaluation is greedy with fixed agent seeds.

## Parent artifact and migration

Both arms start from the fixed Task 4 control-r1 milestone008000 fallback:

* SHA-256: `a5dadff5cc9f04a614caa828d3a0f014cad56f00c3644bb791275da85b35a8a6`
* Source checkpoint size: 3,436,279 bytes.
* Trained runtime commit: `c4ddfa4efadf0b3ec6d4380a4239b9cb3a097113`.
* Pilot implementation commit: `ca5b376ef7759283c29c85d7f06a55f11e499566`.
* Parent counters: 8,000 completed episodes and 1,485,852 optimizer updates.

Migration preserves online and target weights, target phase, Adam moments, and replay contents. The added input columns and corresponding Adam moment columns start at zero.

Historical replay has no recoverable visit history, so we explicitly zero-pad its new columns. These values do not represent reconstructed historical observations.

We use paired fresh random streams. The runner verifies initial greedy behaviour against the original fallback before scientific training.

There is no default trained `checkpoint.pt` in this repository agent directory. We store pilot checkpoints in isolated run outputs. They are not submission artifacts.

The local PC run root is `training_outputs/issue221-pc`. This is an operational location, not a durable evidence publication or retrieval location.

## Training and evaluation protocol

The authoritative executable registration is [`config.json`](../../experiments/2026-09-20-antiloop-pilot/config.json). Its seed audit records separation from previously registered seeds.

We run two paired replicas on the PC, with one single-thread worker per pair.

Both arms train on identical seeded Classic worlds against three `rule_based_agent` opponents, with rotating player slots. Fixed checkpoints occur after 25, 100, 250, and 1,000 new optimizer updates. The episode ceiling is 200 per arm. Missing the update endpoint makes the run incomplete.

The shared budget is two CPU-hours, including mechanics, training, evaluation, and recovery, with an absolute stop at 14:00 Europe/Berlin on 20 September 2026.

The workload RSS ceiling is 8 GiB. At least 2 GiB available memory is required. The watcher preserves resource usage across bounded technical retries and does not resume scientific or resource stops.

Each early evaluation uses 40 Classic games against three rule-based opponents, 20 solo coin-heaven games, and 20 solo loot-crate games.

Both arms must retain fallback performance within the registered score, survival, self-kill, collection, and invalid-action limits.

For a completed pair, the separate final development suites contain 80 Classic rule-based games, 20 solo coin games, 20 solo crate games, 20 Classic peaceful games, 40 Classic mixed-opponent games, and 10 serial latency games per policy.

Policies share world seeds, opponent slots, and evaluation conditions.

## Metrics and decision rule

The runner retains score, kills, survival, self-kills, coins, crates, invalid actions, and decision timings. Diagnostics include attack exposure, attributed bomb kill credits, generated and sampled replay origins, and loop-penalty counts.

The primary loop metric counts overlapping 24-step windows with no crates, hazards, or progress. A window counts as looping when it covers at most three positions.

We report eligible and looping counts together. Overlapping windows are not independent samples. An absent eligible denominator does not pass the screen.

The final memory arm must reduce pooled eligible Classic loop rate by at least 25% against both control and fallback. It must also meet the registered score, kill, survival, self-kill, collection, invalid-action, and latency gates.

In multiplayer suites, invalid-action increases are limited to 0.25 per game pooled and 0.5 per game in every replica against both baselines. Solo suites require zero invalid actions.

Latency-only games are excluded from the behavioural gate. Serial latency requires p95 below 50 ms and maximum below 100 ms.

Only fixed-final checkpoints are eligible for screening. Both replicas must complete. We do not select intermediate checkpoints based on attractive results.

Even a passing screen requires independent confirmation before we consider memory-r1 for submission. There is no automatic promotion.

## Dependencies and compatibility

The runtime uses NumPy and CPU PyTorch. `requirements.txt` declares the additional PyTorch dependency.

Runtime imports stay inside this agent directory. Checkpoint access is relative to the agent module. Evaluation uses one PyTorch thread without multiprocessing.

The pilot stages the agent in an isolated archived framework.

A selected candidate still needs validation of its exact exported package in the clean official framework, including Docker, latency, and memory checks. Successful tests of a different submitted fixture do not certify these weights.

## Results and limitations

No pilot checkpoint is currently promoted. We still need to add completed result tables, uncertainty, durable evidence links, and our interpretation after reviewing the generated observations.

The registered paired bootstrap reports 95% intervals. Two training replicas provide limited evidence about variation between independent training runs.

History covers only 16 observations. The shaping term is not potential-based and might change useful movement preferences despite its exclusions.

Preserved replay initially contains zero history columns and no new loop penalties. We need to assess these limitations, continuation stability, and retention from the pilot results rather than assume they are solved.
