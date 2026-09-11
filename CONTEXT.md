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

## Active DQN handoff - September 11, 2026

Read [the active delivery plan](docs/0009-dqn-deadline-plan.md) after this file.
It replaces the September 7 status snapshot and the expired 48-hour Task 2
roadmap. AGENTS.md remains the stable engineering policy; dated scheduling
belongs here and in the numbered plan, not in operational rules.

### Decision to preserve

Julius has ended standalone DQN Task 2 optimization for now. Status:
**development phase concluded with known limitations; cumulative Task 2
capability NOT complete**. Do not lower gates, close unmet capability issues,
replace frozen models, rerun D/r5, or start broad Task 2 sweeps to make the
status look complete. Task 3 may continue as explicitly exploratory.

The model families remain separate:

- DagobertDuckDQN -> DagobertDuckDQNTask2 -> DagobertDuckDQNTask3 (DQN).
- DerKleineVermoegensumverteiler -> DerKleineSprengstoffkapitalist (tabular).

This closeout makes no claim about tabular Task 2 completion. Both models learn
Q-values; DQN estimates them with a network, tabular learning uses a table.

### Verified evidence and current GitHub snapshot

Refreshed origin/main: `08ca233a4463a880b7e99d077cfc21e3538d22ed`.
This is a snapshot, not a command to reset or switch an active checkout.

- #107 / merged PR #122: negative/mixed factorial record; no cumulative passing
  cell. Escape helps survival; protected replay did not establish retention.
- #124 / PR #131 at `2252c23`: complete reduced evaluation (19 models, 4,880
  episodes), original D/r5 incomplete. Masked C passes Task 2-specific gates but
  fails Task 1 retention/no-bombs and the adoption criterion. No promoted model.
- #91 / PR #136 at `04e71e3`: complete negative gamma comparison. Keep gamma
  0.90; neither arm passes cumulative gates. Corrected execution began before
  peer approval; the disclosed deviation requires review, not retroactive approval.
- #131/#136: all nine CI checks pass; fresh non-author approval required.
- #129 and #130: approved, CI green, behind main; update and revalidate before
  merging. #120: current head approved/green, stacked on #130. #140: green,
  awaiting approval, stacked on #120. Required integration order is
  #130 -> #120 -> #140, retargeting after each dependency merge.
- PR #100 is closed without merge; #98 stays deferred backlog. No
  implementation/run exists. Do not spend the deadline window repairing its
  proposal-only merge conflict or claim the hypothesis was tested/rejected.

Detailed evidence, immutable result links, release retrieval and the remaining
issue dispositions are in [0009](docs/0009-dqn-deadline-plan.md).

### Task 3 running-state boundary

Julius reports the first exploratory Task 3 experiments are running. No local
Python process was visible during the September 11 inspection. Remote PIDs,
progress, available resources and exact running bindings are not independently
verified. Request/read status.json, resources.json, protocol/binding and source
metadata before giving an ETA or starting/resuming anything. A GitHub issue or
old preflight comment does not establish current process state.

#109 now registers #91 A/r3 as an explicitly owner-authorized provisional
parent. Approved Task 3 execution source is
`0c9b0c1add52e53269646a1b2ef2aec96fb297dd` (PR #120, includes #130).
Parent source is `cbd52be8392f5003a91c5600fda4efd544b48ec5`; checkpoint size
2,382,903 bytes; SHA-256
`c0b6081acf5a733258005d46b9430f3ef50e403470a3bb5428661729f9aeedf7`.
The #91 release and exact archive member are in #109 and 0009. Historical #107
A/r2 bindings must remain unchanged. Do not assume a running job uses the newest
registration until its binding is inspected. Never rebind a running campaign.

PR #130/#120 use the 39-input Task 3 successor retaining all 26 Task 2 inputs.
Main's older 34-input contract/fixture is not the execution source for this
campaign; let #130 update that implementation contract through review.
#137/#140 coin-collector execution requires a verified passing peaceful result,
review and separate authorization. All gates are conjunctive; the helper rejects
a failed predecessor. If it fails, register a distinct follow-up decision before
new compute; do not bypass the gate or claim #51 completion.

### Next work and dates

1. Review/merge Task 2 result records; preserve failed gates and historical data.
2. Finish/analyze current #109; prepare/review #137 in parallel, execute only if
   its predecessor gate passes. Diagnose reported loops on development data;
   a learned cycle is not automatically a code defect.
3. Prepare #126 Task 4 control/treatment/analyzer now. Main candidate idea:
   strong-opponent-only versus fixed opponent-mixture training under matched
   budgets, with frozen predecessor reference. Exact protocol is not Ready yet.
4. Use three independent machine allocations: server critical-path training,
   PC independent control/targeted optimization, laptop serial evaluation. Keep
   paired primary/repeat evaluation on one recorded environment and without
   competing training during latency measurement. No duplicate campaigns.
5. Sep 17 18:00 target for optional compatibility upload (hard 21:00).
6. Sep 18-19 bounded final tuning, then independent confirmation and candidate
   freeze by Sep 19 18:00. No training after final held-out inspection.
7. Sep 20 final audit/packaging; Sep 21 18:00 internal submission target,
   **21:00 hard agent deadline**. #127 owns release/fallback/receipt checks.
8. Sep 28 21:00 report deadline; start evidence tables/authorship now.

The detailed day-by-day plan, failure branches, machine scheduling, ranked
experiment ideas, readiness requirements and submission checklist are in 0009.
New proposals are not authorized experiments. Existing runs keep their approved
source/configuration/seeds/budgets. If time slips, drop optional tuning before
sacrificing final validation. Choose the strongest eligible learned team model
by a prospective comparison, not by assuming DQN must win.

### Local state to preserve

The main workspace was at `1f94a61`, behind current origin/main by ten commits,
with untracked `agent_code/DagobertDuckDQNTask2/preview-issue91-a-r3.pt` from the
user's GUI preview. Preserve it; never accidentally package it. The documentation
branch is `docs/106-dqn-deadline-plan` in the isolated `issue106-deadline`
worktree under the user's Temp directory. Numerous experiment/review worktrees
exist: enumerate them again rather than deleting or reusing one.

Task 2 downloads, published compact evidence and original training outputs are
retained. Do not delete or overwrite any source checkpoint, output root, working
copy, resource accounting or non-committed work. No training was launched by the
closeout/planning work. Course announcements and live server state are unknown.

## Authoritative reading map

- AGENTS.md, README.md and CONTRIBUTING.md: engineering and review rules.
- docs/0001: assignment; original ../final_project.pdf overrides summaries.
- docs/0002-0006: architecture, workflow, experiments, DoD and AI disclosure.
- docs/0007: Task 1 tabular contract; do not conflate it with DQN outcomes.
- docs/0008: Task 3 implementation contract at the chosen execution revision.
- docs/0009: current DQN closeout, priorities and deadline/failure plan.
- Agent, training, experiment and submission READMEs: relevant implementation,
  protocol and evidence; read at the actual selected commit.
- Live #106/#109/#137/#126/#127 and linked PR reviews/comments: dependencies,
  readiness and authorization. Refresh rather than trusting this dated snapshot.
