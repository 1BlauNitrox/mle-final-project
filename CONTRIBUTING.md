# Contributing

All changes go through a GitHub issue, a short-lived branch, a pull request,
automated checks, and peer review.

## Before starting

1. Confirm that the issue meets the
   [Definition of Ready](docs/0005-definition-of-ready-and-done.md).
2. Assign the issue to yourself.
3. Pull the latest `main`.
4. Create a branch:

   ```bash
   git switch main
   git pull --ff-only
   git switch -c <type>/<issue-number>-<short-description>
   ```

Allowed branch types are `feature`, `experiment`, `fix`, `docs`, `test`, and
`chore`.

## While working

- Keep the branch focused on one issue.
- Prefer small commits with clear messages such as
  `feat(agent): add danger-map features`.
- Add tests for behavior and contract changes.
- Update the relevant numbered document or agent card when a decision changes.
- For experiments, record hypotheses and evaluation settings before training.
- Never commit secrets, local virtual environments, raw logs, large temporary
  checkpoints, or the final report PDF.
- Do not introduce multiprocessing into evaluation-time agent code.

## Pull requests

Open a draft pull request early for work that takes more than one session. Use
the pull-request template. Write `Closes #<issue>` only when the PR completes
the entire issue; use `Refs #<issue>` when parent-issue work remains.

A pull request is ready for review when:

- its acceptance criteria are met;
- local checks pass;
- the applicable evidence scope is declared and its requirements are met;
- documentation is current;
- the author has completed the Definition of Done self-review.

At least one other team member must approve the pull request. Authors must not
approve their own changes. Resolve requested changes and conversations before
squash merging.

## Local checks

```bash
ruff check tests training scripts agent_code/_team_agent_template
python -m pytest
python main.py play --no-gui --n-rounds 1
```

## Experiment changes

Follow [`docs/0004-experimentation-protocol.md`](docs/0004-experimentation-protocol.md).
A result is not reproducible unless the agent revision, configuration, seeds,
opponents, scenarios, number of rounds, and metrics are recorded.

A prospective protocol PR registers those decisions and validates the execution
path before compute is spent; it does not require results. A completed result
PR retains compact evidence sufficient to recompute its claims. Large required
artifacts may remain outside Git only with a durable retrieval location,
SHA-256 checksum, byte size, contents/schema, retrieval instructions, and a
verification command. Local paths are not reviewable evidence. Incomplete
results use `Refs`, disclose the missing evidence, and leave the parent issue
open.

## Documentation changes

Documentation uses English and Markdown. Add chronological knowledge-base files
as `docs/NNNN-short-title.md`. Link new documents from the root README.
