# 0005 Definition of Ready and Definition of Done

## Definition of Ready

An issue may move from Backlog to Ready only when:

- [ ] The title states a single clear outcome.
- [ ] The problem, motivation, or hypothesis is understood.
- [ ] Acceptance criteria are observable and testable.
- [ ] Dependencies, blockers, and required decisions are resolved or listed.
- [ ] The issue is small enough for one focused pull request, or has been split.
- [ ] Relevant framework and tournament constraints are identified.
- [ ] For experiments, the baseline, controlled variables, scenarios, seeds,
      metrics, and success criterion are defined before training.
- [ ] Required compute, data, and external resources are available.
- [ ] The issue has an owner and an appropriate type/label.

## Definition of Done

A pull request may be merged only when:

- [ ] All acceptance criteria claimed by the PR are met. If a parent issue
      remains incomplete, the PR states the remaining work and does not claim
      to complete that issue.
- [ ] The PR links its issue. It uses `Closes #<issue>` only when merging the PR
      completes the entire issue; otherwise it uses `Refs #<issue>` and leaves
      the issue open.
- [ ] Team-owned code is clear and appropriately commented.
- [ ] New behavior and important logic are covered by tests.
- [ ] Local checks and all required GitHub Actions checks pass.
- [ ] At least one other team member has reviewed and approved the change.
- [ ] All review conversations are resolved.
- [ ] Documentation, agent cards, dependencies, and numbered decisions are
      updated where applicable.
- [ ] The PR declares every applicable evidence scope and meets the
      corresponding requirements below.
- [ ] Every claim is no broader than the retained evidence supports.
- [ ] Evaluation-time code uses relative paths, is self-contained, avoids
      multiprocessing, and respects CPU/time/memory constraints.
- [ ] No secrets, report PDF, raw logs, replays, or unnecessary large artifacts
      are committed.
- [ ] AI-assisted material is reviewed, tested, rewritten as needed, and logged.
- [ ] The branch is current with `main` and is ready for squash merge.

The pull-request template repeats this checklist at the decision point and is
the operational single source of truth for merging.

## Evidence scopes

Evidence is conditional on what a pull request changes and claims. A PR can have
more than one scope and must satisfy every applicable row.

| Scope | Evidence required before merge |
| --- | --- |
| Implementation or validation | Exact validation commands and results, automated tests for changed behavior, and relevant smoke or compatibility checks. A smoke run proves integration only; it is not performance evidence. |
| Prospective experiment protocol | A falsifiable hypothesis, variables and controls, baselines, scenarios, training and evaluation seeds, budgets, metrics, success criterion, artifact plan, and executable or dry-run-validated commands. Results are not required. The PR must say that no scientific result is in scope and must not claim that the experiment succeeded. |
| Completed experiment or training | The prospectively registered protocol, immutable code/configuration/artifact references, compact results and conclusions, sufficient per-run or per-seed observations to recompute reported metrics and uncertainty, and a documented analysis or verification command. |
| Frozen or released model | The selected evaluation artifact, its SHA-256 checksum and byte size, producing code/configuration and source-artifact provenance, the prospective selection rule, evaluation results, and a reproducible export or verification command. |
| Partial or incomplete result record | All evidence that is still available, an explicit account of missing or invalid data and unsupported claims, and concrete follow-up work. It uses `Refs #<issue>` and leaves unmet acceptance criteria and the parent issue open. |

A protocol-only PR is therefore complete when it makes a future experiment
executable and reviewable. A results PR is complete only when it records the
outcome. Separating those changes is encouraged for long-running experiments
because the protocol can be reviewed before compute is spent.

## Evidence retention

Commit the smallest durable evidence package that is sufficient to check the
claims. This normally includes the protocol and configuration, compact
per-run/per-seed observations, aggregate tables, final figures used for a
decision, conclusions, and the analysis or verification code. It does not mean
committing every episode, intermediate checkpoint, replay buffer, or verbose
log. Retain those only when they are necessary to reproduce a reported value or
investigate a disclosed failure.

Large required evidence may remain outside Git only when the experiment record
or artifact manifest provides all of the following:

- a durable, retrievable location or immutable release identifier;
- the SHA-256 checksum and byte size;
- a description of the contents and schema;
- exact retrieval instructions; and
- an exact command that verifies the download and reproduces or validates the
  claimed result.

A machine-local path, a statement that files existed, or a checksum without
retrievable bytes is not reviewable evidence. If required evidence is
unavailable, narrow the claim, mark the result unverified or incomplete, use
`Refs` rather than `Closes`, and keep the relevant issue or acceptance criteria
open.
