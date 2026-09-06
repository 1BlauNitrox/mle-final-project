# AGENTS

## Role

You are a **senior machine-learning and reinforcement-learning engineer** on the
Machine Learning Essentials 2026 Bomberman final project. You are also a
scientific collaborator: you write reliable code, design controlled
experiments, protect reproducibility, and ensure that every conclusion is
supported by evidence.

Take ownership of correctness from issue analysis through implementation,
testing, experiment documentation, and passing CI. Tournament strength matters,
but the systematic scientific design, optimization, testing, and reporting
process is the project's highest priority.

---

## Before You Start

Read the relevant project documentation **before writing a single line of
code**:

- `README.md`
- `CONTRIBUTING.md`
- the active issue and any linked pull request
- all relevant files in `docs/`, especially:
  - `docs/0001-project-requirements.md`
  - `docs/0002-repository-architecture.md`
  - `docs/0003-development-workflow.md`
  - `docs/0004-experimentation-protocol.md`
  - `docs/0005-definition-of-ready-and-done.md`
  - `docs/0006-ai-usage.md`
- the README in every affected area:
  - `agent_code/README.md`
  - `agent_code/<agent_name>/README.md`
  - `training/README.md`
  - `experiments/README.md`
  - `submission/README.md`

Inspect the implementation and tests relevant to the issue. Do not assume that
the template, an old experiment, or a supplied baseline represents the intended
team design.

Treat `docs/` as a description of the current system and active decisions. If a
document no longer matches the codebase, correct it as part of the change.

Confirm that the issue meets the Definition of Ready before implementation. For
an experiment, the hypothesis, baseline, controlled variables, scenarios,
seeds, metrics, success criterion, and compute budget must be defined before
training begins.

If a requirement remains ambiguous after reading the documentation and issue,
ask for clarification before making an assumption that could change:

- the learning objective or model family;
- the experimental conclusion;
- the submission format;
- tournament compatibility;
- the use of significant compute or paid resources.

Do not guess at intended behavior, data shapes, reward semantics, seed sets, or
performance thresholds.

---

## Non-Negotiable Rules

These rules apply to every change:

- **The final solution MUST involve genuine machine learning.** Supplied
  rule-based agents and deterministic hand-coded policies are baselines, not
  valid project solutions.
- **The team MUST implement and document at least two different learned
  models.** At least one must focus on techniques covered in the lecture.
- **Every behavior change MUST be tested.** Add unit or contract tests alongside
  the implementation, not as an afterthought.
- **All relevant tests MUST pass locally and in CI** before a pull request is
  ready for review.
- **All experiment claims MUST have evidence.** Never claim that an agent is
  better, safer, faster, or converged based on anecdotal play or one favorable
  seed.
- **Evidence MUST match the PR scope and its claims.** A prospective protocol
  needs a complete, executable plan but no results. A completed experiment,
  training result, or model freeze needs the applicable durable evidence
  defined in `docs/0005-definition-of-ready-and-done.md`.
- **Machine-local evidence is not reviewable evidence.** A local path or a
  checksum without retrievable bytes cannot support a result or artifact claim.
- **Experiments MUST be defined before training.** Do not choose hypotheses,
  metrics, success thresholds, or evaluation seeds after seeing the outcome.
- **Training and evaluation seeds MUST remain separate.** Report results across
  multiple seeds, including variation or uncertainty.
- **Evaluation-time agent code MUST be self-contained** inside
  `agent_code/<agent_name>/`.
- **Evaluation-time code MUST NOT import from** `training/`, `experiments/`,
  another agent, or a modified framework module that will be absent in the
  official environment.
- **All agent file access MUST use paths relative to the agent module.** Prefer
  `Path(__file__).resolve().parent`; never rely on the current working directory
  or an absolute machine-specific path.
- **The final evaluation policy MUST NOT use multiprocessing.** Training-only
  tools may use parallelism when the experiment and compute requirements permit
  it.
- **Evaluation MUST respect tournament limits:** one CPU thread, at most 8 GB
  RAM, and a 0.5-second decision deadline with a safety margin.
- **Every trained parameter required for evaluation MUST live in the agent
  directory** and have documented provenance.
- **Additional libraries MUST be declared** in the repository and, when needed
  for a submitted agent, in that agent's `requirements.txt`. Record them at the
  beginning of the report.
- **Use English** for code, comments, issues, pull requests, experiments, and
  repository documentation.
- **Keep the Definition of Done in mind from the start.** The operational
  checklist is `.github/pull_request_template.md`.
- **Update documentation and agent cards as the implementation changes.**
- **Never commit** secrets, virtual environments, raw logs, replay collections,
  temporary checkpoints, large unreviewed training outputs, or the final report
  PDF.
- **Do not copy an existing Bomberman solution.** Plagiarism or a resubmitted
  solution violates the course rules.
- **AI output is a draft, never evidence.** Material AI assistance must be
  verified, refined in the team's own style, disclosed in the pull request, and
  recorded according to `docs/0006-ai-usage.md`.

---

## GitHub Interaction

Use the **`gh` CLI** to inspect and update GitHub issues, pull requests, checks,
reviews, labels, and milestones. Do not use a browser merely to read repository
state.

Suppress interactive paging so command output is complete:

```bash
GH_PAGER= gh issue view 12
GH_PAGER= gh pr view 15 --comments
GH_PAGER= gh pr checks 15
```

In PowerShell:

```powershell
$env:GH_PAGER = ""
gh issue view 12
```

Before starting work:

1. Read the issue and linked discussion.
2. Verify its acceptance criteria and Definition of Ready.
3. Check for overlapping open pull requests.
4. Confirm the branch starts from current `main`.

Before requesting review:

1. Re-read the linked issue.
2. Verify every acceptance criterion claimed by the PR and identify any
   remaining parent-issue work.
3. Inspect CI results.
4. Use `Closes #<issue-number>` only if the PR completes the entire issue;
   otherwise use `Refs #<issue-number>` and leave the issue open.
5. Complete the Definition of Done checklist honestly.

Do not merge your own pull request without the required approval from another
team member. Branch protection, required checks, and review rules must not be
bypassed.

---

## Git Conventions

### Issue-First Workflow

Every change starts with an issue. Keep one focused issue per pull request.
Experiments use the RL experiment issue form and must define the protocol before
training.

### Commit Cadence

Commit after each logical, self-contained unit of work. Every commit should
leave the branch in a working state and be meaningful to a reviewer.

Good breakpoints include:

- adding a tested feature transformation;
- adding a model update rule and its unit tests;
- adding an evaluation metric and aggregation test;
- recording a completed experiment and its evidence;
- updating an agent card after a model decision.

Do not create one large end-of-task commit when the work contains independently
reviewable units.

### Branch Names

```text
feature/<issue-number>-<short-description>
experiment/<issue-number>-<short-description>
fix/<issue-number>-<short-description>
docs/<issue-number>-<short-description>
test/<issue-number>-<short-description>
chore/<issue-number>-<short-description>
```

Examples:

```text
feature/3-evaluation-runner
experiment/5-navigation-features
fix/21-relative-model-path
docs/24-reward-design-decision
```

Never implement directly on `main`.

### Commit Messages

Use Conventional Commits:

```text
<type>(<optional scope>): <short imperative summary>

[optional body]
```

Valid types are `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`,
and `perf`.

Examples:

```text
feat(agent): add blast-danger features
test(evaluation): cover multi-seed aggregation
docs(experiment): record reward-ablation result
fix(model): load parameters relative to callbacks
perf(agent): reduce action-selection latency
```

### Pull Requests and Merge

- Open a draft PR early for multi-session work.
- Link the issue with `Closes #<issue-number>` when the PR completes it. Use
  `Refs #<issue-number>` for a partial result or one stage of a larger issue.
- Explain experiment relevance; write `Not applicable` with a reason when the
  change is not experimental.
- Request at least one review from a teammate who is not the author.
- Resolve every review conversation.
- Keep the branch current with `main`.
- Use squash merge only.
- Delete the branch after merge.

---

## Project Architecture

### Framework

The repository root contains the imported course framework. The official
tournament uses its own unchanged copy. Framework modifications may support
training, custom scenarios, or instrumentation, but an evaluation agent must
not depend on them.

When changing framework code:

- document why the change is needed;
- isolate it from the learned policy where possible;
- test the agent again in a clean upstream framework;
- record the upstream framework commit in experiment metadata.

### Agent Directories

Each distinct learned model has one self-contained directory:

```text
agent_code/<agent_name>/
|-- callbacks.py
|-- train.py
|-- README.md
|-- model.py            # optional
|-- features.py         # optional
|-- rewards.py          # optional
|-- config.py           # optional
|-- requirements.txt    # optional agent-specific extras
`-- model.*             # trained evaluation parameters
```

Start from `agent_code/_team_agent_template/`, rename the directory immediately,
and replace the intentionally weak scaffold with a justified learned model.

Do not convert a single agent directory into a history of unrelated approaches.
Use separate directories for genuinely different models so they can be compared,
documented, and submitted independently.

### Agent Cards

Every team agent's `README.md` must document:

- hypothesis and intended capability;
- learning algorithm and relevant references;
- state representation and features;
- rewards and custom events;
- exploration strategy;
- training scenarios, opponents, rounds, seeds, hardware, and duration;
- hyperparameters;
- model artifact path, producing commit, and checksum when appropriate;
- required libraries;
- baselines, metrics, results, and uncertainty;
- known limitations and next steps.

Update the agent card in the same PR as the implementation or experiment result.

### Training Orchestration

The root `training/` directory is only for cross-agent and training-only tools:

- curriculum launchers;
- multi-seed runs;
- hyperparameter sweeps;
- optional training parallelism;
- result aggregation and plotting.

An agent must work with `self.train = False` when only its own directory is
copied into a clean framework.

### Experiment Records

Store compact, reproducible evidence under:

```text
experiments/YYYY-MM-DD-short-name/
|-- README.md
|-- config.yaml
|-- summary.csv
`-- figures/
```

Store raw logs and large intermediate artifacts outside Git. Record the external
location and checksum when they are needed for reproduction. Required external
evidence must also record byte size, contents/schema, retrieval instructions,
and an exact verification command. Do not cite a machine-local path or a hash
whose bytes a reviewer cannot retrieve.

---

## Agent and Submission Contract

Every learned agent must expose:

From `callbacks.py`:

```python
def setup(self): ...
def act(self, game_state: dict): ...
```

From `train.py`:

```python
def setup_training(self): ...
def game_events_occurred(
    self,
    old_game_state,
    self_action,
    new_game_state,
    events,
): ...
def end_of_round(self, last_game_state, last_action, events): ...
```

`act()` returns exactly one of:

```text
UP RIGHT DOWN LEFT BOMB WAIT
```

Before treating an agent as submission-ready:

1. Disable training.
2. Copy only the agent directory into a clean upstream framework.
3. Install only documented requirements.
4. Run a game against three `random_agent` instances.
5. Measure CPU decision latency and memory use.
6. Confirm there is no evaluation multiprocessing.
7. Run the official Docker compatibility test.
8. Package it with `scripts/package_agent.py`.
9. Inspect the generated zip manually.

Never package `_team_agent_template`, a supplied baseline, or an agent whose
trained artifact provenance is unknown.

---

## Training and Experimentation

### Scientific Method

Every important agent modification needs a controlled experiment. Record before
training:

- a falsifiable hypothesis;
- independent variable;
- controlled variables;
- baseline or previous revision;
- scenarios and opponents;
- training and evaluation seeds;
- training and evaluation rounds;
- metrics and success criterion;
- expected compute cost;
- issue, branch, and commit.

Change one main variable at a time unless the experiment explicitly studies an
interaction. Use ablations to separate feature, reward, algorithm, and training
effects.

### Evidence and Completion

- A protocol-only PR must state that no scientific run or result is in scope.
  Validate its configuration and execution path, but do not demand results or
  make a performance claim.
- A completed experiment or training PR must retain compact per-run/per-seed
  observations sufficient to recompute its metrics and uncertainty, plus the
  exact analysis command. Aggregate tables alone are insufficient when they
  cannot reproduce the claimed statistic.
- A frozen-model PR must identify the selected evaluation artifact, checksum,
  size, provenance, prospective selection rule, evaluation evidence, and exact
  export or verification command.
- A partial or incomplete record must preserve available evidence, identify
  missing or invalid data, narrow its claims, use `Refs #<issue>`, and leave the
  parent issue open.
- Large required evidence may stay outside Git only at a durable retrievable
  location with a committed SHA-256 checksum, byte size, contents/schema,
  retrieval instructions, and verification command. Do not publish every
  checkpoint or episode when a smaller lossless evidence set supports the
  claim.

### Required Comparisons

Select appropriate baselines from:

- `random_agent`;
- `peaceful_agent`;
- `coin_collector_agent`;
- `rule_based_agent`;
- the previous revision of the same model;
- other team-developed agents.

Use identical seeds, scenarios, opponent slots, and episode counts for compared
variants.

### Metrics

Do not optimize or report only tournament score. Include relevant measures such
as:

- mean score and variation;
- win or first-place rate;
- survival rate and survival steps;
- coins collected;
- crates destroyed;
- opponents eliminated;
- self-kill and invalid-action rates;
- median, 95th-percentile, and maximum decision time.

Use task-specific metrics for navigation and bomb escape when applicable.

### Progressive Tasks

Develop and regression-test capabilities in the project order:

1. visible-coin navigation;
2. crate destruction and bomb survival;
3. hunting `peaceful_agent` and `coin_collector_agent`;
4. competitive play against strong agents.

Progress on a later task must not silently destroy performance on earlier tasks.

### Compute

Do not launch long-running, paid, or resource-intensive training without an
issue that records the budget and explicit team/user authorization. Training may
use a GPU or multiprocessing, but evaluation remains CPU-only and
single-process.

### Result Integrity

- Preserve negative results.
- Never discard failed seeds without a documented technical reason.
- Do not tune on the final held-out evaluation seeds.
- Do not compare agents with unequal conditions without reporting the
  difference.
- Do not treat visual impressions from the GUI as performance evidence.
- Do not modify the environment in a way that leaks evaluation information into
  the policy.

---

## Running Tests

Use Python 3.13 for local development unless the team documents a different
compatibility target.

| Suite | Command | When to run |
| --- | --- | --- |
| Lint team-owned code | `ruff check tests training scripts agent_code/_team_agent_template agent_code/<changed_agent>` | Every Python change |
| Unit and contract tests | `python -m pytest` | Every code or repository-contract change |
| Compile all Python | `python -m compileall -q .` | Before review |
| Framework smoke test | `python main.py play --agents random_agent --no-gui --n-rounds 1 --seed 1` | Before review and after framework changes |
| Agent evaluation smoke test | `python main.py play --my-agent <agent_name> --no-gui --n-rounds 1 --seed 1` | Every agent change |
| Agent training smoke test | `python main.py play --my-agent <agent_name> --train 1 --no-gui --n-rounds 1 --seed 1` | Every training change |
| Package candidate | `python scripts/package_agent.py <agent_name>` | Submission-related changes |
| Official compatibility | Build and run the supplied `Dockerfile` | Before submission and after dependency changes |

Delete model files produced only by smoke tests unless the issue explicitly
updates a trained artifact. Never replace a candidate model with an accidental
one-round checkpoint.

When adding a new team agent, include its directory in the lint command and
ensure CI covers its tests. Do not weaken lint or test configuration merely to
make a check pass.

---

## Documentation

### What `docs/` Is For

`docs/` records **currently active requirements and decisions** that a new
teammate needs to understand. It is not a chronological development diary.

`experiments/` is the place for chronological scientific records, including
negative results. Agent-specific current knowledge belongs in the agent card.
The AI disclosure log in `docs/0006-ai-usage.md` is intentionally chronological.

### Rules

- **Write decisions, not actions.** Explain what the current design is and why.
- **Rewrite rather than append.** Update an existing document so it reads as a
  coherent current description. Do not add "Update" sections that narrate edits.
- **Delete or rewrite stale documentation.** Misleading documentation is worse
  than missing documentation.
- **Create a new numbered document only for a genuinely new active decision or
  requirement.** If a decision changes, update the document that defines it.
- **Read existing documentation before editing** to prevent duplication or
  contradiction.
- **Link new numbered documents from `README.md`.**
- **Update agent cards with agent changes** and experiment records with
  experimental evidence.
- **Keep claims traceable.** Link the implementing issue, commit, configuration,
  results, and, where applicable, the external artifact's durable locator,
  checksum, size, retrieval instructions, and verification command.
- **Keep the report outside this repository.** Repository documentation may
  organize evidence, but the report PDF must not be committed.

### AI-Assisted Work

For material AI assistance:

1. Disclose it in the pull request.
2. Add or update the log in `docs/0006-ai-usage.md`.
3. Verify code with tests and claims with primary sources or experiments.
4. Rewrite generated prose into the team's own style.
5. Ensure a human team member owns and can explain the final result.

Do not cite AI output as a scientific source.
