# Project Context and Session Handoff

This document is a reusable onboarding and handoff context for contributors and
coding agents working on the Machine Learning Essentials 2026 Bomberman final
project.

It has two different kinds of information:

1. **Durable project rules**, which summarize the assignment and the repository's
   active decisions.
2. **A dated status snapshot**, which becomes stale and must be refreshed at the
   start of every session.

The authoritative sources remain the supplied assignment PDF (currently
`../final_project.pdf` in this workspace), `AGENTS.md`, the active documents in
`docs/`, the relevant agent and experiment READMEs, and the current GitHub
issue/PR discussions. If this file conflicts with one of those sources, inspect
the conflict and follow the authoritative source rather than guessing.

## Task 3 branch handoff (2026-09-12)

Refresh GitHub before acting: PR #145 remains an unmerged roadmap proposal;
its accepted scheduling decisions guide this work. Completed #109 and #147
peaceful campaigns did not pass every gate, so no Task 3 replica is selected and
#137 / PR #140 cannot automatically continue. Task 2 remains cumulatively
incomplete; do not restart standalone DQN Task 2 optimization.

Read-only #146 diagnosis is in PR #151. It reproduces learned WAIT and legal
movement cycles without establishing the original GUI incident's cause. The
next prepared hypothesis is #150 / PR #154: matched standard versus Double DQN
training targets, fresh provisional #91 A/r3 initialization, unchanged rewards,
features and hunting/retention gates. See
[the protocol](experiments/2026-09-12-task3-double-dqn/README.md),
[server commands](experiments/2026-09-12-task3-double-dqn/SERVER.md), and
[independent device checks](experiments/2026-09-12-task3-double-dqn/DEVICES.md).
Preparation and short mechanics checks are complete; long-run execution still
needs the concrete protocol/budget decision and review or an explicit owner
execution exception. This is not scientific evidence of improvement. PR #149
under #126 remains parent-gated Task 4 preparation, not a launchable campaign.

## Mandatory startup behavior for every new session

Before changing code, documentation, experiments, issues, or pull requests:

1. Locate the intended repository and run `git status --short --branch` and
   `git worktree list`. Do not assume the first local checkout is on `main` or
   current.
2. Preserve all existing worktrees and uncommitted user work. Do not reset,
   delete, move, or reuse a worktree without verifying its purpose.
3. Run `git fetch origin --prune` and compare the selected checkout with
   `origin/main`.
4. Read `AGENTS.md` completely.
5. Read the active issue, its comments, every linked PR, PR description, review,
   and review conversation.
6. Read the relevant documents and area READMEs listed below.
7. Use `gh` to refresh all open PRs and issues. Treat the dated snapshot in this
   file only as a lead for investigation.
8. Check for overlapping branches and currently running training processes.
   A process on another team member's machine cannot be inferred from GitHub.
9. Confirm the issue meets the Definition of Ready and that any scientific run
   has explicit compute authorization before starting it.
10. State what is verified, what is inferred, and what remains unknown. Ask the
    user before making a choice that changes the model family, learning
    objective, experimental conclusion, submission contract, or significant
    compute budget.

Useful refresh commands:

```powershell
$env:GH_PAGER = ""
git status --short --branch
git worktree list
git fetch origin --prune
git log origin/main -15 --date=iso --pretty=format:'%h|%ad|%an|%s'
gh pr list --repo 1BlauNitrox/mle-final-project --state open --limit 100
gh issue list --repo 1BlauNitrox/mle-final-project --state open --limit 100
gh pr view <PR> --repo 1BlauNitrox/mle-final-project --comments
gh pr checks <PR> --repo 1BlauNitrox/mle-final-project
gh issue view <ISSUE> --repo 1BlauNitrox/mle-final-project --comments
```

On Windows, inspect local training commands when relevant:

```powershell
Get-CimInstance Win32_Process |
  Where-Object { $_.Name -match '^python(w)?\.exe$' } |
  Select-Object ProcessId, ParentProcessId, CreationDate, CommandLine
```

## Assignment and project goal

The team must develop reinforcement-learning agents for Bomberman. Tournament
strength contributes to the grade, but the systematic scientific process -
design, controlled optimization, testing, evidence, and reporting - carries
substantially more weight.

Non-negotiable course requirements:

- The solution must use genuine machine learning. Supplied rule-based agents
  are baselines, not valid submissions.
- Implement and document at least two different learned models. At least one
  must focus on techniques covered in the lecture.
- Work collaboratively across the models rather than assigning one isolated
  model to each team member.
- Divide the problem into manageable subgoals and use meaningful experiments to
  motivate subsequent modifications.
- Compare variants with previous learned versions and suitable supplied agents.
- Preserve and report successful, negative, mixed, and abandoned approaches.
- The report's Experiments and Results section is the most important section.
  It must show systematically whether changes improved performance and justify
  the final agent selection.

The repository currently contains the two required learned model families:

- feature-based tabular Q-learning;
- a neural Deep Q-Network (DQN).

Their existence does **not** mean the overall assignment is complete. Final
held-out evaluation, later-task performance, final model selection, packaging,
submission validation, and the report remain separate obligations.

### Deadlines

- Optional MaMPF compatibility submission: **17 September 2026, 21:00**.
- Final agent-code zip: **21 September 2026, 21:00**.
- PDF report: **28 September 2026, 21:00**.

The report should contain about 4,000 words per team member, identify the main
author after each heading, link the public repository, follow the prescribed
seven-section structure, and remain outside this repository. Do not use the
university logo.

### Runtime and submission contract

- Official evaluation uses the original, unchanged course framework.
- The submitted zip contains only the selected `agent_code/<agent_name>/`
  directory and everything it needs at evaluation time.
- Evaluation is CPU-only, limited to one AMD Ryzen 5 2600 thread and 8 GiB RAM.
- `act()` has a 0.5-second deadline; overruns reduce the following step's
  allowance.
- Evaluation-time multiprocessing is forbidden. Training may use it when the
  approved plan permits it.
- The agent must be self-contained, use agent-relative paths, and declare every
  additional dependency in its own `requirements.txt` when needed.
- Required callbacks are `setup()` and `act()` in `callbacks.py`, plus
  `setup_training()`, `game_events_occurred()`, and `end_of_round()` in
  `train.py` for training.
- Valid actions are `UP`, `RIGHT`, `DOWN`, `LEFT`, `BOMB`, and `WAIT`.
- All trained evaluation parameters must live inside the submitted agent
  directory.

## The four tasks are cumulative

The assignment explicitly says that the tasks are subsets of one another. A
Task 2 agent must retain Task 1 navigation; a Task 3 agent must retain Tasks 1
and 2; a Task 4 agent must retain Tasks 1-3.

1. **Task 1 - navigation:** collect visible coins quickly on `coin-heaven`,
   without crates, bombs, or opponents.
2. **Task 2 - bombing and survival:** on crate boards without opponents,
   destroy crates, reveal and collect hidden coins, and escape own bombs while
   retaining efficient navigation.
3. **Task 3 - hunting:** on `classic`, first hunt `peaceful_agent`, then the
   harder `coin_collector_agent`.
4. **Task 4 - competition:** compete for score against strong agents and team
   variants. Beating `rule_based_agent` is an important tournament-readiness
   target.

Later-task agents are therefore cumulative successors, not independent agents
that forget earlier tasks. At the same time, frozen earlier agents must remain
unchanged so the team can make honest regression comparisons.

## Repository architecture and ownership boundaries

The imported framework is based on `ukoethe/bomberman_rl` commit `0f55c1d`.
Framework changes may support training and instrumentation, but official
evaluation will not contain them.

```text
agent_code/<agent>/   self-contained policy, training callbacks, model,
                      features, rewards, persistence, requirements, agent card
training/             cross-agent orchestration, run plans, analysis, plotting
experiments/          prospective protocols and compact scientific results
tests/                unit, integration, contract, provenance, and regression tests
docs/                 current requirements and active architectural/process decisions
.github/              issue forms, PR template, CI and packaging workflows
submission/           submission packaging guidance
main.py               framework CLI entry point
```

Important boundaries:

- One self-contained directory represents each distinct learned model or frozen
  capability milestone.
- Freeze a baseline before extending it. Build the next task in a separately
  named successor directory.
- A successor records parent artifact checksums and performs an explicit,
  tested migration. It must never import runtime code from its parent.
- Evaluation code must not import `training/`, `experiments/`, another agent,
  or modified framework-only helpers.
- Controlled duplication inside agent directories is preferable to a hidden
  submission dependency. Repository-level differential tests may import both
  agents.
- Use a `features.py` module for a small agent or a `features/` package for a
  growing agent. Packages expose a stable API from `features/__init__.py`;
  `assemble.py` builds the complete vector, while files such as
  `navigation.py`, `bombs_and_crates.py`, and later opponent modules own focused
  feature groups.
- Root `training/` owns only orchestration that is not shipped: curricula,
  multi-seed plans, sweeps, aggregation, plotting, and optional training-only
  parallelism.
- Each agent README is its agent card and must document algorithm, features,
  rewards, hyperparameters, training, seeds, artifacts, dependencies, results,
  limitations, and provenance.

## Learned-agent lineages and accepted evidence

### Tabular Q-learning lineage

```text
DerKleineVermoegensumverteiler       frozen Task 1 parent
                 |
                 | checksum-verified migration
                 v
DerKleineSprengstoffkapitalist       cumulative Task 2 successor
```

#### `DerKleineVermoegensumverteiler`

- Frozen Task 1 development-selected baseline; never retrain or modify it for
  later tasks.
- Eight categorical features: four local movement flags and visible-coin
  direction/distance information.
- Five actions: `UP`, `RIGHT`, `DOWN`, `LEFT`, `WAIT`; `BOMB` is forbidden.
- Tabular Q-learning with `alpha=0.05`, `gamma=0.9`, epsilon `1.0`, decay
  `0.99`, and floor `0.1`.
- Development-selected campaign result: aggregate coin-collection fraction
  `0.9812` across five trained models. This is not final held-out Task 1
  evidence.
- Frozen artifact: `model.npz`, SHA-256
  `4e1da63a819ef8f51b112ffaf422ab251b853915375fe486538be8595b988307`.
- The final held-out Task 1 evaluation in
  `docs/0007-task-1-baseline-contract.md` has not been performed.

#### `DerKleineSprengstoffkapitalist`

- Cumulative tabular Task 2 successor; implementation merged through issue
  #45/PR #101.
- Six actions including `BOMB` and a sparse 17-feature Task 2 state. The first
  eight features retain the Task 1 projection; additions cover bomb
  availability, danger, safe directions, escape, crates, and useful blast
  targets.
- Unseen Task 2 states initialize their first five Q-values from the frozen
  Task 1 table and initialize BOMB conservatively.
- Task 2 training is enabled and retains Task 1 rewards plus crate/coin/death/
  survival signals.
- Issue #102/PR #105 is the accepted negative Task 2 development baseline:
  five replicas, 10,000 episodes each (`2,000 coin-heaven -> 2,000 loot-crate
  -> 6,000 classic`), 720 primary and 720 repeat evaluation episodes.
- Task 1 retention passed, but Task 2 feasibility failed: trained-minus-
  untrained classic collection difference `+0.00167`, 95% CI
  `[0.0, 0.005]`, only two of five replicas improved, and the absolute `0.30`
  classic collection gate was not reached.
- Durable Issue #102 evidence is attached to GitHub release
  `issue102-evidence-v1` and is reproducible with
  `training.analyze_tabular_task2_experiment`.
- The current agent card still contains older statements claiming no
  scientific Task 2 training. Treat the merged Issue #102 experiment record as
  the newer status source and reconcile the agent card in a focused change.

Current tabular follow-up work is described in the dated snapshot below. Do not
adopt an unmerged treatment as the default.

### DQN lineage

```text
DagobertDuckDQN                        frozen Task 1 reference
                 |
                 | zero-suffix migration, checksum-pinned
                 v
DagobertDuckDQNTask2                   cumulative Task 2 successor
                 |
                 | prospective selected-parent binding
                 v
DagobertDuckDQNTask3                   Task 3 successor in open PR #115
```

#### `DagobertDuckDQN`

- Frozen development reference from issue #42; training is disabled.
- Network: `8 -> 64 -> 64 -> 5` with ReLU hidden layers.
- Standard DQN: online/target networks, uniform replay, Adam, Huber loss,
  gradient clipping, hard target synchronization, seeded epsilon-greedy
  exploration.
- Main defaults: learning rate `0.001`, gamma `0.9`, batch `64`, replay
  capacity `10,000`, warm-up `256`, target sync every `250` updates, max
  gradient norm `10`, epsilon `1.0 -> 0.1` with decay `0.99`.
- Issue #41 produced an aggregate Task 1 collection fraction of `0.8334` but
  failed reproducibility/invalid-action and per-model gates.
- Issue #58 tested movement shaping and was negative/inconclusive. Per its
  stopping rule, no third Task 1 tuning experiment was run.
- Run 02 was selected mechanically as the median issue #58 model and frozen.
  It achieved development collection fraction `0.8165` and invalid-action rate
  `0.000315`; this does not make the overall DQN Task 1 experiment successful.
- Frozen evaluation artifact: `checkpoint.pt`, SHA-256
  `eb08e3f67b620ac2a253a2af4db3d5b4c6ea9e667a2aaf1d91e3fccf4ba8b05e`.

#### `DagobertDuckDQNTask2`

- Task 2 implementation and initial experiment infrastructure are merged.
- Network: 21 inputs, two 64-unit hidden layers, six actions. The Task 1
  eight-feature prefix is retained; Task 2 additions cover bomb availability,
  timed blast danger, safe moves, escape-after-bomb, reachable crate targets,
  and useful bomb placement.
- Current Task 2 defaults include learning rate `0.0005`, gamma `0.9`, epsilon
  decay `0.9997`, replay warm-up `500`, and target sync every `500` updates.
  These defaults were not individually validated as improvements.
- Issue #85 corrected migration by zeroing the 13 new input columns while
  copying inherited layers/actions. The corrected artifact is
  `checkpoint-issue85-zero-suffix.pt`, SHA-256
  `3edb2e7196030fcb52af6c7dc9ee69d9fc1259898ea674002fe06fbe93468015`.
- PR #92 merged the migration/protocol and PR #96 fixed the runner fingerprint
  mismatch. The first 60-episode migration-retention execution remains invalid
  and a clean repeat is still required. GitHub issue #85 is nevertheless
  currently closed. Do not infer a valid result from that state; reconcile or
  reopen tracking before making a migration-retention claim.
- Issue #46 remains open. Its corrected development evaluation is a negative/
  mixed, incompletely verifiable result: classic trained-minus-untrained
  collection `+0.0406` with 95% CI `[+0.0233, +0.0589]`, but the registered
  Task 2 absolute/improvement/self-kill/invalid-action gates and Task 1
  retention gates failed. Coin-heaven retention difference was `-0.6351`
  with CI `[-0.7006, -0.5658]`. Some raw training/resource evidence is still
  unavailable.
- Issue #86/PR #95 tested framework-legal action masking. It reduced invalid
  actions to zero but failed collection and survival non-regression gates, so
  the treatment was rejected and the default remains unmasked (`none`).

Do not silently replace frozen or historical artifacts. Every later DQN
experiment must identify which migration/checkpoint it starts from.

## Scientific experiment workflow

An experiment is not “train, inspect, then explain.” Follow this sequence:

1. **Create/read the experiment issue.** It must state one falsifiable
   hypothesis or a prospectively justified interaction.
2. **Meet the Definition of Ready.** Fix the baseline, independent variable,
   controls, artifacts, training/evaluation seeds, scenarios, opponents,
   budgets, metrics, uncertainty method, numeric success/non-regression gates,
   checkpoint-selection rule, owner, reviewer, hardware, and compute ceiling.
3. **Register the protocol before compute.** A protocol-only PR contains the
   executable plan/analyzer and successful dry runs, explicitly says that no
   scientific result is in scope, and makes no performance claim.
4. **Receive review and compute authorization.** Long-running or expensive
   training must not start merely because code exists.
5. **Execute an immutable clean revision.** Use fixed seed populations and the
   exact registered artifacts. Do not modify source/configuration mid-run.
6. **Use `training.run_plan`.** Inspect `--dry-run`, then execute. Resume an
   interrupted immutable plan with `--resume`; never replace a failed attempt
   or silently restart under the same plan ID.
7. **Evaluate mechanically selected checkpoints.** No checkpoint fishing,
   favorable-seed deletion, post-hoc thresholds, or opening held-out seeds for
   tuning.
8. **Retain claim-checkable evidence.** Keep all failures. Commit the smallest
   per-run/per-seed evidence needed to recompute aggregates and uncertainty.
9. **Publish required large evidence correctly.** Raw logs, replay buffers, and
   large checkpoints normally stay out of Git. If required for a claim, use a
   durable retrievable release/location and record SHA-256, byte size,
   contents/schema, retrieval steps, and an exact verification command. A local
   path or checksum without retrievable bytes is not evidence.
10. **Analyze with the registered analyzer.** Validate provenance, code/config/
    artifact fingerprints, seeds, deterministic repeats, and artifact
    immutability before producing results.
11. **Create a result PR.** Report aggregates, uncertainty, limitations,
    negative/mixed outcomes, and the adopt/reject/inconclusive decision. Use
    `Refs` if work or evidence remains; use `Closes` only when the complete issue
    is satisfied.
12. **Freeze only through a prospective rule.** A model-freeze PR records the
    selected evaluation artifact, hash, size, source lineage, producing
    code/config, selection rule, evaluation evidence, and export/verification
    command.

A smoke run proves only integration. It is never performance evidence. AI
output is also never scientific evidence.

### Experimental controls

- Change one main factor at a time unless a factorial interaction is explicitly
  preregistered with all necessary ablations.
- Compare arms using identical seeds, scenarios, opponents, slots, budgets, and
  checkpoint rules.
- Keep training, development, confirmation, and final seed populations
  disjoint.
- Report variation or confidence intervals across multiple replicas/seeds.
- Include earlier-task regression suites in later-task experiments.
- Preserve negative results and technical failures.
- Never widen a conclusion beyond what the retained evidence supports.

The Task 1 tabular contract in `docs/0007-task-1-baseline-contract.md` is more
specific: five training roots, 50 development world seeds, and 100 final world
seeds; final evaluation is one episode per final seed for every trained model;
the completion gates include aggregate collection fraction at least `0.80`, at
least four models at `0.75`, significant advantage over uniform random,
invalid-action rate below `0.01`, zero BOMB actions, deterministic immutable
evaluation, p95 below `50 ms`, and maximum below `100 ms`.

## GitHub development and review workflow

Every change follows:

```text
issue -> Definition of Ready -> branch from current main -> implementation,
tests and documentation -> draft PR -> CI -> non-author review -> resolve all
findings -> current with main -> squash merge -> delete remote branch
```

Rules:

- Never implement directly on `main`.
- One focused issue per PR.
- Branch names are
  `<feature|experiment|fix|docs|test|chore>/<issue>-<description>`.
- Use Conventional Commit messages.
- Open a draft PR early for multi-session work.
- Use `Closes #N` only when the PR completes every issue acceptance criterion.
  Otherwise use `Refs #N`, state remaining work, and keep the issue open.
- Complete `.github/pull_request_template.md` honestly, including every
  applicable evidence scope.
- At least one team member other than the author must approve. New commits
  dismiss stale approvals. Authors do not approve or merge their own work.
- Read PR descriptions, comments, all linked issues, review bodies, inline
  conversations, documentation, and CI before reviewing.
- Resolve every conversation, keep the branch current, and squash merge only.

### Definition of Done and evidence scopes

The DoD is conditional on the PR's claims:

- **Implementation/validation:** exact commands and results, tests for changed
  behavior, and relevant smoke/compatibility checks.
- **Prospective protocol:** complete falsifiable, executable or dry-run-validated
  plan; no results required and no success claim allowed.
- **Completed experiment/training:** registered protocol, immutable provenance,
  compact recomputable observations, aggregates/uncertainty, analysis command,
  conclusion, and decision.
- **Frozen/released model:** artifact, hash, size, provenance, prospective
  selection rule, evaluation evidence, and export/verification command.
- **Partial/incomplete result:** preserve available evidence, disclose missing
  or invalid data and unsupported claims, define follow-up, use `Refs`, and
  leave the parent issue open.

Do not demand all raw outputs in Git. Do demand enough accessible evidence to
verify every claim.

## CI/CD and verification

There is no production deployment. “Delivery” means a validated agent package.

CI runs once for PR updates and once for pushes to `main`; ordinary feature-
branch pushes do not cause a duplicate full run. Required jobs are change
detection, quality checks, and a framework smoke test. More expensive tabular,
DQN, successor, packaging, and official-Docker jobs are selected conservatively
from changed paths and run independently. Detection fails closed. Documentation
and retained result changes may skip irrelevant expensive jobs.

The official Docker image is rebuilt for relevant changes. A roughly 4 GiB
Buildx cache was measured and rejected because importing/exporting it made the
cold job much slower. Do not reintroduce caching without new measurements.

Typical local verification from a current clean branch:

```bash
python -m pip install -r requirements-dev.txt
ruff check tests training scripts agent_code/_team_agent_template agent_code/<changed-agent>
python -m pytest
python -m compileall -q .
python main.py play --agents random_agent --no-gui --n-rounds 1 --seed 1
python main.py play --my-agent <agent> --no-gui --n-rounds 1 --seed 1
python main.py play --my-agent <agent> --train 1 --no-gui --n-rounds 1 --seed 1
python -m training.run_plan <registered-plan.yaml> --dry-run
python scripts/package_agent.py <agent>
```

Use Python 3.13 unless the team prospectively changes the compatibility target.
Do not let a one-round training smoke overwrite a real candidate checkpoint.
Run the official Docker compatibility path before submission and after relevant
dependency or packaging changes.

## Documentation reading order

Always read the files relevant to the current task, not only this summary:

1. `AGENTS.md` - operational rules for agents and contributors.
2. `README.md` and `CONTRIBUTING.md` - setup, structure, and contribution flow.
3. `docs/0001-project-requirements.md` - assignment requirements.
4. `docs/0002-repository-architecture.md` - callback lifecycle, self-contained
   agents, transitions, persistence, frozen/successor decisions.
5. `docs/0003-development-workflow.md` - issue, branch, review, merge, and CI.
6. `docs/0004-experimentation-protocol.md` - prospective design and evidence.
7. `docs/0005-definition-of-ready-and-done.md` - readiness, merge, and evidence
   scopes.
8. `docs/0006-ai-usage.md` - AI policy and chronological disclosure log.
9. `docs/0007-task-1-baseline-contract.md` - normative Task 1 tabular gates.
10. `agent_code/README.md`, the affected agent card, `training/README.md`,
    `experiments/README.md`, and `submission/README.md`.
11. The active experiment README/config/run plans/analyzer and all related
    GitHub discussion.
12. The supplied assignment PDF (currently `../final_project.pdf`) when
    interpreting course requirements; it overrides a repository summary if
    they differ.

`docs/` describes current decisions, not a chronological diary. Update or
rewrite stale statements instead of adding “update” appendices. Chronological
experiment outcomes belong in `experiments/`; agent-specific current knowledge
belongs in its agent card. Material AI assistance must be disclosed in the PR
and recorded in `docs/0006-ai-usage.md`.

## Dated project snapshot - 2026-09-07 21:25 Europe/Berlin

This section is deliberately explicit but not durable. Refresh it before
acting.

### Remote `main`

- `origin/main`: `933a8fe11440e0f7645254390928da6af5dad46d`
- Most recent merges:
  - PR #105 / issue #102: accepted negative tabular Task 2 baseline.
  - PR #95 / issue #86: DQN legal-action-masking experiment; treatment rejected.
  - PR #101 / issue #45: tabular Task 2 implementation.
  - PR #96: Issue #85 runner/provenance correction.
  - PR #92: DQN Task 2 migration-retention implementation/protocol.
  - PR #94: clarified DoD scientific-evidence requirements.
  - PR #81: recorded the corrected but incomplete/negative Issue #46 result.
  - PR #80: staged experiment orchestration.

### Open pull requests

| PR | Purpose | Snapshot state |
| ---: | --- | --- |
| #115 | Cumulative `DagobertDuckDQNTask3` implementation for #108 | Ready, CI green, review required; provisional Task 2 parent must later be rebound after #107. |
| #113 | Protected Task 1 replay for #88 | Stacked on PR #112, CI green; no accepted scientific result. |
| #112 | Multi-step escape features for #87 | CI green, non-author review required; prerequisite for #107. |
| #111 | Tabular legal-action masking result for #110 | Reports a negative collection result and durable release evidence; changes are requested, so it is not accepted/merged. |
| #104 | DQN reward experiments for #103 | Reports both treatments rejected; reviewer identified missing durable/reproducible source evidence. Do not treat the decision as accepted while the blocker remains. |
| #100 | Safety-constrained DQN exploration proposal for #98 | Draft/backlog; deliberately not Ready and not authorized for training. |
| #99 | DQN staged-vs-direct curriculum for #97 | Changes requested; only the direct arm ran, the staged comparison and ratified decision rule are missing. |
| #84 | Dependabot PyTorch 2.14 CPU update | Quality check failing in this snapshot. |
| #83 | Dependabot PyYAML update | CI green, review required. |

Before working on any PR, refresh its head SHA, base, merge status, checks,
latest reviews, inline comments, and linked-issue state.

### Open issues and active plan

The open issue set is: `#46`, `#51`, `#52`, `#53`, `#87`, `#88`, `#89`,
`#90`, `#91`, `#97`, `#98`, `#103`, `#106`, `#107`, `#108`, `#109`,
`#110`, and `#114`.

The time-boxed DQN roadmap is issue #106:

1. Merge/test #87 (multi-step escape) and #88 (protected Task 1 replay).
2. Finalize #107's not-yet-Ready 2x2 factorial protocol: control, escape only,
   replay only, combined. It proposes 20 replicas and 200,000 training episodes,
   but exact decision rules, owner/reviewer, resources, and authorization must
   be fixed before execution.
3. Prepare Task 3 in parallel through #108/PR #115.
4. After #107 selects a development predecessor, bind it into the Task 3
   successor and execute a prospectively completed #109 peaceful-opponent
   protocol. Progress later toward the existing coin-collector issue #51.
5. If Task 2 gates still fail, Task 3 work is explicitly exploratory; it does
   not become evidence of Task 2 completion.

Older DQN ideas #89-#91 and #97-#98 are not on #106's 48-hour critical path.
Do not start extra Task 2 sweeps contrary to the accepted time box.

Tabular work:

- Issue #110/PR #111 completed a legal-action-mask comparison but is still under
  requested changes. Its branch reports zero invalid actions but no collection
  improvement; do not adopt until review is resolved and merged.
- Issue #114 has an active remote branch
  `experiment/114-tabular-useful-bomb-reward` at `e9554fe`. It prospectively
  compares `USEFUL_BOMB_PLACED: +1.0` with the unmasked Issue #102 baseline.
  No PR or accepted result existed at this snapshot. Its record claims a
  user-directed execution exception; verify the exact authorization and current
  process/result state rather than treating that exception as general policy.
- Issues #51-#53 retain future cross-agent/baseline comparisons: DQN versus
  `coin_collector_agent`, tabular successor versus `coin_collector_agent`, and
  frozen DQN versus frozen Q-learning for Task 1.

### Local machine and worktrees

- No local `python.exe`/`pythonw.exe` training process was running at the
  snapshot time. This says nothing about another team member's machine.
- The primary local checkout was on
  `feat/44-dqn-task2-bomb-crate-capability-v2` at `d22db8e`, ahead 9 and behind
  10 relative to current `origin/main`. Do not implement new work there merely
  because it is the first checkout.
- Dedicated worktrees exist for Issues #97 and #103 at `C:/run97` and
  `C:/run103`, plus multiple historical review/CI worktrees. Preserve them.
- Re-check process command lines, timestamps, run-plan `status.json`, retained
  outputs, GitHub comments, and remote branch commits before declaring an
  experiment running, complete, failed, or resumable.

## Immediate safe conclusion for a new agent

Do not start by inventing another algorithm or running training. First refresh
the state, select the exact requested issue, verify its readiness and
dependencies, and work from current `origin/main` in an appropriate focused
branch/worktree. The project's largest risks are currently not missing code but
scientific scope drift, unreviewable evidence, stale documentation, accidental
modification of frozen artifacts, overlapping experiment branches, and treating
partial or unmerged results as accepted decisions.
