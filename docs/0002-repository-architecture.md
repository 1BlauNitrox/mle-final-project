# 0002 Repository Architecture

## Decision summary

Use one self-contained folder per learned agent. Keep framework-required
training callbacks and all evaluation-time code inside that folder. Use shared
root directories only for experiment orchestration, compact results, tests, and
documentation.

The framework lifecycle documented below is based on the current implementation
in `main.py`, `environment.py`, and `agents.py`. It must not be inferred from
the incomplete `tpl_agent` or the intentionally weak team-agent template.

## Context

The course framework imports an agent from `agent_code/<agent_name>` and loads
`train.py` from the same folder only in training mode. During official
evaluation, the course team copies only the submitted agent directory into an
unchanged framework. Imports from arbitrary repository folders may therefore
work locally but fail in the tournament.

The architecture must support:

- self-contained evaluation;
- reproducible training and evaluation;
- clear separation of training and evaluation behavior;
- correct handling of ordinary and terminal transitions;
- model persistence relative to the agent directory; and
- controlled experiments without creating hidden runtime dependencies.

## Chosen structure

```text
agent_code/<agent_name>/
|-- callbacks.py        # Required evaluation interface
|-- train.py            # Required training interface
|-- README.md           # Required agent card
|-- model.py            # Optional learned model implementation
|-- features.py         # Optional state representation
|-- rewards.py          # Optional reward definitions
|-- config.py           # Optional default hyperparameters
|-- requirements.txt    # Only if agent-specific extras are required
`-- model.*             # Trained parameters required at evaluation
```

The exact optional modules can differ by agent. The invariants are:

- the directory can be copied into a clean official framework;
- it contains `callbacks.py`, `train.py`, and an agent card;
- it never requires absolute paths;
- it does not import evaluation-time code from `training/`, `experiments/`, or
  another agent;
- its final policy contains no multiprocessing; and
- it loads its trained parameters relative to its own module location.

## Framework agent loading

The framework creates an `AgentRunner` for each configured agent. The runner
imports:

```text
agent_code.<agent_name>.callbacks
```

for every agent.

For a training agent, it additionally imports:

```text
agent_code.<agent_name>.train
```

The runner validates that the imported modules expose the required callbacks
with the required number of arguments.

The required callbacks are:

```python
# callbacks.py
def setup(self): ...
def act(self, game_state: dict): ...

# train.py
def setup_training(self): ...

def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: list[str],
): ...

def end_of_round(
    self,
    last_game_state: dict,
    last_action: str,
    events: list[str],
): ...
```

The runner creates one persistent callback object and provides it as `self` to
all callbacks. The framework predefines at least:

```text
self.train
self.logger
```

Attributes created in `setup()` or `setup_training()` remain available to later
callbacks.

The initialization order is:

```text
import callbacks.py
-> import train.py when training
-> validate callback signatures
-> callbacks.setup(self)
-> train.setup_training(self) when training
```

`train.py` is not imported in evaluation mode. Evaluation code must therefore
not depend on side effects, constants, objects, or imports from `train.py`.

## Evaluation lifecycle

An evaluation agent executes only the callbacks from `callbacks.py`.

The lifecycle is:

```text
callbacks.setup(self)

for each episode:
  reset framework-owned episode state

  for each step while the agent is active:
    construct and store the current game_state
    reset the current events

    if available_think_time > 0:
      callbacks.act(self, game_state)
      store the returned action as last_action

      if the decision exceeds the available time:
        execute WAIT instead
        reduce the available time for the next step
      else:
        execute the returned action
        reset the available time
    else:
      skip callbacks.act()
      execute WAIT
      recover part of the available time
      keep last_action unchanged

    update the environment
```

When `callbacks.act()` is skipped because no thinking time is available, the
framework still stores the newly constructed game state and resets the current
events. It then executes `WAIT` while keeping the action most recently returned
successfully by `callbacks.act()` in `last_action`.

Evaluation mode has no access to training events and must not:

- perform exploration intended only for learning;
- update Q-values or other model parameters;
- import `train.py`;
- write or replace the model artifact;
- depend on repository-level training tools; or
- use multiprocessing inside the submitted policy.

For Task 1, evaluation action selection must use the ordered action set:

```text
UP, RIGHT, DOWN, LEFT, WAIT
```

`BOMB` is excluded from the Task 1 baseline and must never be selected.

Evaluation tie-breaking must be deterministic or controlled by a documented
agent seed. Given the same model artifact, framework revision, configuration,
world seed, and agent seed, the action sequence and episode result must be
reproducible.

## Training lifecycle

A training agent executes callbacks from both `callbacks.py` and `train.py`.

The high-level lifecycle is:

```text
callbacks.setup(self)
-> train.setup_training(self)

for each episode:
  reset framework-owned episode state

  for each step while the agent is active:
    construct pre-action game_state
    store it as the wrapper's last_game_state
    reset the current event list
    callbacks.act(self, game_state)
    store the returned action as last_action
    execute the action
    update coins, bombs, explosions, and agents
    collect resulting events

    if the training agent survives the transition:
      train.game_events_occurred(
        self,
        old_game_state,
        self_action,
        new_game_state,
        events
      )

  add the final survival event for surviving agents
  train.end_of_round(
    self,
    last_game_state,
    last_action,
    events
  )
```

The exact lifecycle is shown below.

```mermaid
sequenceDiagram
    participant W as "BombeRLeWorld"
    participant A as "Agent wrapper"
    participant C as "callbacks.py"
    participant T as "train.py"

    W->>A: "Create agent and runner"
    A->>C: "setup(self)"

    alt "Training mode"
        A->>T: "setup_training(self)"
    end

    loop "Each step while active"
        W->>W: "Construct pre-action game_state"
        W->>A: "store_game_state(game_state)"
        W->>A: "reset_game_events()"
        A->>C: "act(self, game_state)"
        C-->>A: "return action"
        W->>W: "Execute action and update world"

        alt "Training agent survives the transition"
            W->>T: "game_events_occurred(old_state, action, new_state, events)"
        else "Agent dies during transition"
            W->>W: "Keep terminal events for end_of_round"
        end
    end

    alt "Training mode"
        W->>T: "end_of_round(last_state, last_action, events)"
    end
end
```

## Per-step transition ownership

The framework wrapper owns the previous state, previous action, and event list
used to construct training transitions.

Before requesting an action, the environment:

1. constructs the current `game_state`;
2. stores it in `Agent.last_game_state`;
3. clears the agent's event list; and
4. calls `callbacks.act()`.

After receiving the result, the wrapper stores the returned action in
`Agent.last_action`.

The environment then:

1. attempts to execute the action;
2. collects coins;
3. advances explosions;
4. advances and detonates bombs;
5. removes killed agents; and
6. records the resulting events.

For a surviving training agent, the wrapper calls:

```python
game_events_occurred(
    self,
    self.last_game_state,
    self.last_action,
    current_game_state,
    self.events,
)
```

Although the wrapper field is named `last_game_state`, it represents the state
immediately before the action for the transition currently being reported. In
the training callback, it is therefore correctly interpreted as
`old_game_state`.

The ordinary Task 1 data flow is:

```text
old_game_state
-> feature encoder
-> old state key
-> selected action
-> environment transition
-> events
-> scalar reward
-> new_game_state
-> feature encoder
-> new state key
-> ordinary Q-learning update
```

For ordinary Q-learning, the update target is:

```text
reward + gamma * max_a Q(new_state, a)
```

The update belongs to training code and must never be performed from
`callbacks.act()`.

## Terminal transitions

A terminal transition differs from an ordinary transition because no future
return can be obtained after the episode has ended.

The terminal Task 1 data flow is:

```text
last_game_state
-> feature encoder
-> last state key
-> last_action
-> terminal events
-> scalar reward
-> terminal Q-learning update
```

The terminal target is:

```text
reward
```

It must not include:

```text
gamma * max_a Q(next_state, a)
```

A terminal state has no future action value. Agent code should represent the
future state key as `None` or use an explicit terminal flag.

## Death during a transition

If a training agent dies while the environment processes its action:

- the agent is marked dead;
- terminal death events are added;
- `game_events_occurred()` is not called for that transition;
- the final state after death is not passed to the agent; and
- `end_of_round()` is called with the state before the last action, the last
  returned action, and the terminal events.

The death transition must therefore be learned from `end_of_round()`.

The agent must not attempt to construct a successor-state feature vector for
this transition. Its terminal update must use no bootstrap value.

## Surviving agent at the end of a round

The framework sends ordinary game events before checking whether the round must
end.

For a training agent that survives the final environment transition, the
observed order is:

```text
game_events_occurred(...)
-> environment checks stopping condition
-> SURVIVED_ROUND is added
-> end_of_round(...)
```

Consequently, the same final state/action pair may be visible first through the
ordinary callback and then through the terminal callback with the additional
`SURVIVED_ROUND` event.

A learned agent must not blindly apply an ordinary update and then apply a
second independent terminal update to the same transition.

A robust implementation may keep the most recent transition pending until its
terminal status is known:

```text
game_events_occurred:
    if a previous transition is pending:
        finalize it as an ordinary transition with bootstrapping
    store the current surviving transition as pending

end_of_round:
    if last_game_state and last_action refer to the pending transition:
        finalize the pending transition exactly once as terminal
        do not bootstrap from a successor state
    else:
        if a pending transition exists:
            finalize it as an ordinary transition with bootstrapping
        if last_game_state and last_action are available:
            finalize the callback's death transition as terminal
            do not bootstrap from a successor state
```

This distinction is important when an agent dies. The last
`game_events_occurred()` callback may describe the previous surviving
transition, while `end_of_round()` describes a separate transition containing
the action that caused the agent's death. In that case, the previous pending
transition must be finalized ordinarily before the death transition is finalized
as terminal.

Equivalent implementations are allowed, but the agent must document and test
how duplicate learning of the last surviving transition is prevented.

Tests for the baseline agent must verify both cases:

- When the agent survives the round, the final transition produces exactly one
  terminal update without bootstrapping.
- When the agent dies after a previous surviving transition, the previous
  transition produces one ordinary update with bootstrapping and the death
  transition produces one terminal update without bootstrapping.

## Timeout and executed-action distinction

When sufficient thinking time remains, `Agent.wait_for_act()` stores the action
returned by `callbacks.act()` as `last_action`.

If evaluation-time decision-making exceeds the available time, the environment
replaces the action that is actually executed with `WAIT`. The stored
`last_action` may therefore differ from the executed action in a timeout case:

```text
last_action = action returned by the agent
executed action = WAIT
```

The time overrun also reduces the thinking time available during the following
step. If `available_think_time` is then zero or negative, the framework:

```text
constructs and stores the new game_state
-> resets the current events
-> skips callbacks.act()
-> executes WAIT
-> keeps the previous last_action unchanged
-> recovers part of the available thinking time
```

Therefore, on a step where `callbacks.act()` is skipped, `last_game_state` may
refer to the new step while `last_action` still refers to the most recent action
successfully returned by the agent.

Training callbacks normally have an unlimited framework timeout, so this edge
case is primarily relevant to evaluation diagnostics. Evaluation must remain
comfortably below the official time limit so that the distinction does not
affect behavior.

## `None` and initial-state handling

Agent code must handle initial and terminal values defensively.

| Value | Ordinary surviving transition | Terminal or exceptional case |
| --- | --- | --- |
| `game_state` in `act()` | A current state dictionary | Agent code should reject or safely handle `None` if called directly by a test |
| `old_game_state` | The state immediately before the reported action | May be `None` only outside the normal completed-step lifecycle or in defensive tests |
| `new_game_state` | The state after the environment transition | Not supplied to `end_of_round()` |
| `last_game_state` | Stored before the most recent action | May be `None` if an episode ends before a state/action cycle completes |
| `last_action` | The most recently returned action | May be `None` if no action was successfully returned |
| Successor feature key | Derived from `new_game_state` | Must be `None` for a terminal update |
| Bootstrap value | `max_a Q(new_state, a)` | Must be zero for a terminal update |

The normal framework lifecycle stores a game state before calling `act()`.
Therefore, the first ordinary call to `game_events_occurred()` normally receives
a non-`None` `old_game_state`.

The type annotation in a supplied template is not proof that the framework
passes `None` during a normal first transition. Behavior must be based on the
current framework implementation and covered by tests.

If `end_of_round()` receives no usable state or action because no transition was
completed, training code should skip the Q-value update but may still perform
safe bookkeeping or persistence.

## Task 1 responsibility boundaries

The Task 1 baseline follows this logical flow:

```text
game_state
-> features
-> hashable state key
-> action selection
-> environment transition and events
-> scalar reward
-> Q-learning update
-> optional model persistence
```

Responsibilities should be assigned as follows:

| Responsibility | Location |
| --- | --- |
| Load evaluation parameters | `callbacks.setup()` |
| Initialize an untrained model when training | `callbacks.setup()` or an agent-local helper |
| Convert a game state into features | Agent-local `features.py` |
| Select an evaluation action | `callbacks.act()` |
| Select an exploratory training action | `callbacks.act()` when `self.train` is true |
| Initialize training-only state | `train.setup_training()` |
| Convert events into a scalar reward | Agent-local `rewards.py` |
| Apply ordinary Q-learning updates | `train.game_events_occurred()` or a tested pending-transition helper |
| Apply terminal Q-learning updates | `train.end_of_round()` |
| Decay epsilon | The documented episode boundary |
| Save a resumable model | After a consistent training update at a documented persistence point |
| Aggregate experiment metrics | Root `training/` tools |
| Store experiment evidence | `experiments/` |

Feature extraction and action selection must be shared consistently between
training and evaluation. Evaluation behavior must not depend on importing the
training callback module.

### DQN specialization

`DagobertDuckDQN` follows the same agent-local responsibility boundaries while
replacing the tabular value function with a neural approximation:

```text
game_state
-> eight raw features
-> float32 normalization
-> seeded epsilon-greedy action selection
-> pending framework transition
-> bounded replay buffer
-> sampled mini-batch
-> online-network update against a frozen target network
```

`callbacks.setup()` creates or restores training state when `self.train` is
true. Evaluation instead loads only a frozen online network and creates no
optimizer or replay buffer. `callbacks.act()` performs feature extraction,
normalization, and action selection but never updates parameters.

`train.py` owns pending-transition resolution, replay insertion, mini-batch
updates, epsilon decay, learning diagnostics, and checkpoint writes. The newest
surviving transition remains pending until the next callback establishes
whether it is ordinary or terminal. This preserves the terminal-transition
rules described above and prevents duplicate learning at the end of a surviving
round.

The DQN uses one CPU thread during evaluation and contains no multiprocessing.
Its evaluation modules import only from its own agent directory and declared
runtime dependencies.

## Model persistence

Evaluation-time artifact paths must be derived from the agent module location:

```python
from pathlib import Path

AGENT_DIRECTORY = Path(__file__).resolve().parent
ARTIFACT_PATH = AGENT_DIRECTORY / "model.ext"
```

Agent code must not use:

- an absolute machine-specific path;
- the repository root as an assumed working directory;
- the current process working directory;
- a path inside `training/`; or
- a path to another agent.

Training may write a resumable artifact. Evaluation must load the selected
artifact without modifying it.

Every artifact must contain or be accompanied by all configuration that affects
evaluation, including:

- action order;
- feature representation and normalization versions;
- learned parameters;
- deterministic tie-breaking configuration; and
- any agent seed required for reproducibility.

A resumable neural-agent checkpoint must additionally preserve all state that
affects the next training update. For `DagobertDuckDQN`, this includes online
and target weights, optimizer state, optimizer update count, bounded replay
contents, replay and action RNG states, epsilon, completed episodes,
hyperparameters, rewards, and schema versions.

DQN checkpoints are written to a temporary file in the agent directory,
flushed and synchronized, and then installed with atomic replacement.
Evaluation uses restricted CPU deserialization, extracts only the online policy,
sets it to evaluation mode, disables gradients, and creates no training
objects.

Persistence tests must verify:

- save/load round trips;
- exact optimizer and replay continuation;
- RNG-state continuation;
- evaluation-relevant configuration preservation;
- schema rejection;
- relative-path behavior;
- failed-write recovery;
- deterministic loading; and
- byte-for-byte non-mutation during evaluation.

## Logging and training metrics

The callback object provides `self.logger`. Agent code should use it for concise
diagnostics and avoid excessive step-level output during long runs.

Training-only metrics may include:

- cumulative shaped reward;
- epsilon;
- Q-table size;
- visited-state count; and
- mean absolute temporal-difference error.

These values may be exported to the repository-level experiment pipeline, but
evaluation-time code must not import from `training/`.

Raw logs, temporary checkpoints, replay collections, and smoke-test artifacts
must remain outside version control.

## Root `training/` directory

The root `training/` directory is appropriate for code that coordinates more
than one agent or is not shipped:

- curriculum launchers;
- hyperparameter sweep definitions;
- multi-seed experiment runners;
- optional parallel training orchestration;
- plotting and aggregation tools; and
- compute-environment notes.

This code may call the framework and agent training callbacks, but evaluation
code must not call back into it.

## Root `experiments/` directory

Commit compact and useful scientific evidence:

- experiment plans;
- configuration files;
- small CSV or JSON summaries;
- final plots used for decisions;
- conclusions; and
- links to the implementing commit.

Do not commit raw logs, replay collections, temporary checkpoints, or very large
training artifacts. Store those externally and record their location and
checksum where appropriate.

Recommended naming:

```text
experiments/
`-- YYYY-MM-DD-short-name/
    |-- README.md
    |-- config.yaml
    |-- summary.csv
    `-- figures/
```

## Agent cards

Every learned agent's `README.md` should answer:

- What hypothesis motivated the agent?
- Which learning algorithm is used?
- How is `game_state` represented?
- Which rewards and custom events are used?
- How was the agent trained, including scenarios, opponents, rounds, and seeds?
- Which hyperparameters and model artifact are used?
- What dependencies are required?
- Which baselines and metrics were used?
- What were the results and limitations?
- Which commit produced the stored model?

Copy `agent_code/_team_agent_template/` when beginning a new agent, then rename
the directory immediately.

## Shared code policy

Duplication is preferable to a hidden submission dependency. If multiple agents
share useful feature code, either:

1. copy and version it inside each agent directory; or
2. add a packaging step that vendors it into each agent directory and verify the
   packaged output in CI.

Use option 1 initially because it is simpler and less error-prone. Reconsider
only if duplication becomes a demonstrated maintenance problem.

## Framework changes

Framework changes may support faster training or custom scenarios, but:

- isolate and document them;
- never assume they exist in official evaluation;
- retest trained agents against an unchanged upstream framework;
- record the upstream framework commit used by the experiment; and
- avoid placing evaluation-critical behavior only in modified framework code.

The imported Dockerfile retains the course dependency set but pins the public
`continuumio/miniconda3:25.3.1-1` base, which bundles Python 3.13. The previous
moving base changed to Python 3.14, where the Dockerfile's supplied TensorFlow
dependency was unavailable and prevented the compatibility image from building
before any agent code ran. Pinning the published image matches the repository's
documented development target, avoids automatically accepting later interactive
channel terms, and makes the compatibility check reproducible. Submitted agents
must still be tested in the course-provided environment.

## Alternatives considered

### One global training package

Rejected for framework callbacks and runtime code because the official
submission contains only one agent directory. Accepted only for orchestration
that is not needed at evaluation.

### One monolithic team agent

Rejected because the assignment requires at least two model approaches and
systematic comparisons. Separate agent directories make variants and artifacts
auditable.

### One agent per team member

Rejected because the handout explicitly values real teamwork across models.
Issues may have owners, but design, review, and experimental conclusions remain
team responsibilities.

### Deriving lifecycle behavior from supplied template agents

Rejected because supplied agents are examples rather than normative
architecture. Their incomplete callback handling, persistence choices, or type
annotations do not override the actual behavior of `main.py`,
`environment.py`, and `agents.py`.
