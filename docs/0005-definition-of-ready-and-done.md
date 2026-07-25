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

- [ ] All linked-issue acceptance criteria are met.
- [ ] The PR contains `Closes #<issue>`.
- [ ] Team-owned code is clear and appropriately commented.
- [ ] New behavior and important logic are covered by tests.
- [ ] Local checks and all required GitHub Actions checks pass.
- [ ] At least one other team member has reviewed and approved the change.
- [ ] All review conversations are resolved.
- [ ] Documentation, agent cards, dependencies, and numbered decisions are
      updated where applicable.
- [ ] Experiment changes record configuration, seeds, baselines, metrics,
      results, and conclusions.
- [ ] Claims in documentation are supported by committed evidence or a clearly
      identified external artifact.
- [ ] Evaluation-time code uses relative paths, is self-contained, avoids
      multiprocessing, and respects CPU/time/memory constraints.
- [ ] No secrets, report PDF, raw logs, replays, or unnecessary large artifacts
      are committed.
- [ ] AI-assisted material is reviewed, tested, rewritten as needed, and logged.
- [ ] The branch is current with `main` and is ready for squash merge.

The pull-request template repeats this checklist at the decision point and is
the operational single source of truth for merging.
