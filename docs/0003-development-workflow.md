# 0003 Development Workflow

## Purpose

The workflow keeps `main` reproducible, ensures knowledge sharing, and ties every
change to an agreed outcome.

## Issue states

Use these conceptual states on the GitHub board:

1. **Backlog** - captured but not yet ready.
2. **Ready** - meets the Definition of Ready.
3. **In Progress** - assigned and has an active branch.
4. **In Review** - represented by an open pull request.
5. **Done** - merged after satisfying the Definition of Done.

## Issue types

- **Experiment:** a testable RL hypothesis with baselines, metrics, seeds, and
  acceptance criteria.
- **Feature:** a user- or developer-visible capability.
- **Bug:** reproducible incorrect behavior.
- **Task:** documentation, setup, research, maintenance, or coordination.

Use the matching issue form. An experiment should be defined before code or
long-running training begins.

## Branches

Create branches from current `main`:

```text
feature/<issue>-<description>
experiment/<issue>-<description>
fix/<issue>-<description>
docs/<issue>-<description>
test/<issue>-<description>
chore/<issue>-<description>
```

Examples:

- `experiment/14-tabular-q-learning-rewards`
- `feature/21-danger-map-features`
- `fix/34-relative-model-path`

Keep branches short-lived and scoped to one issue.

## Pull requests

Open a draft PR early for visibility. The PR must:

- explain what changed and why;
- contain `Closes #<issue-number>`;
- include experiment results or explicitly state that experiments do not apply;
- list validation performed;
- identify documentation and dependency changes;
- complete the Definition of Done checklist.

When the PR becomes ready, request review from at least one teammate who is not
the author.

## Reviews

Reviewers check:

- correctness and clarity;
- experimental validity and absence of data leakage;
- reproducibility;
- runtime and submission compatibility;
- tests and CI;
- documentation;
- whether conclusions are supported by evidence.

Review comments should explain risk or desired behavior. Authors resolve every
conversation before merge.

## Merge policy

- Direct pushes to `main` are prohibited.
- Required CI checks must pass.
- At least one non-author approval is required.
- Stale approvals are dismissed when new commits are pushed.
- The branch must be current with `main`.
- Use squash merge to keep one focused commit per issue.
- Delete the remote branch after merge.

Because the PR body contains `Closes #<issue>`, merging into `main`
automatically closes the linked issue.

## Commit messages

Use concise, imperative conventional-style messages:

```text
feat(agent): add bomb-danger feature
test(evaluation): compare policy across fixed seeds
docs(experiment): record reward sweep outcome
fix(model): load parameters relative to callbacks
```

## Small changes

Typos and tiny documentation corrections still use a PR. The repository's
review protection applies consistently so no ambiguous bypass category is
needed.
