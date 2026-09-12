# Issue 157: retain the original Task 3 authorization during resume

Refs #157 and #150. Implementation and recovery mechanics only: no scientific
result, new protocol, budget extension, or checkpoint selection is claimed.

The owner reported successful PR #156 storage recovery: 2,479 completed input
copies verified, 2,471 storage replacements, 8,154,880,737 estimated bytes
reclaimed, 21 protected checkpoint files verified, and a second audit with zero
replacements. Free space was 8.2 GiB. The server audit journals still need to be
retrieved with the eventual scientific evidence; this is an owner-supplied status.

The next resume failed before any jobs started. The confirmed record values are:

```text
authorization.json: 2026-09-12T11:21:26.793224+00:00
resources.json:     2026-09-12T11:21:26.793224Z
```

The shared monitor compares timestamp strings, although its own persistence
changes the UTC suffix to `Z`. The focused test reproduces that round-trip defect
using the actual monitor. The timestamps denote exactly the same instant.

## Bounded adapter

Download `scripts/resume_issue150.py` outside the existing campaign checkout.
It requires source `6f014485a3026cc3707fa2cc3a379880dd0b74bd`, the original protocol
and binding, ten completed training jobs with intact final/live checkpoints,
no campaign lock or recorded workers, unchanged resource limits, and at least
8 GiB available memory and disk. It defaults to a read-only preflight. This
capacity requirement applies to the already deduplicated, partially completed
campaign; it does not validate PR #154's disproven fresh-run storage estimate.

During `--execute`, the adapter replaces only the orchestrator's monitor factory
in its own process. It proves timezone-aware instant equality, passes the stored
timestamp spelling to the original monitor, and delegates all checks and usage
accounting to that unchanged monitor. The original CLI still validates complete
authorization identity, source, binding, plans, artifacts, seeds and limits.
The runtime adapter is explicitly recorded; an unchanged Git checkout alone
must not be described as the complete executed orchestration provenance.

The adapter does not rewrite authorization/resource records to enable resume.
The legitimate resumed monitor continues normal resource accounting. Original
usage and downtime remain charged: 24 CPUh / 10 wallh / 8 GiB, ending September 12
at 21:21:26 UTC / 23:21:26 Europe/Berlin. A breached budget remains a blocker.
No Task 2 worktree, code, campaign record or experiment process is changed.

Before execution, a new flushed `resume-<uuid>.jsonl` journal records helper SHA,
source, original authorization/resource timestamps, resource state, remaining
budget, job counts and record/checkpoint hashes. Failures retain this journal.
Only successful completion creates the usual completion marker, then invokes
the original supervisor for analysis, verification and compact export. The
original runner skips completed jobs and retains interrupted attempts.

## Server handoff

Use the exact helper revision and SHA supplied in the issue/PR handoff; verify
the download with `sha256sum` before running. Keep the campaign checkout pinned.
After downloading the helper as `~/task3-issue150/resume.py`:

```bash
cd "$HOME/task3-issue150"
PY="$HOME/task3-issue109/repo/.venv/bin/python"
"$PY" resume.py --root .
```

Only if the preflight succeeds and the campaign is stopped, execute the same
authorized remaining evaluations in a fresh tmux session:

```bash
export TASK150_AUTHORIZED=yes
CMD="TASK150_AUTHORIZED=yes '$PY' resume.py --root . --execute"
CMD="$CMD >> supervisor.log 2>&1"
tmux new-session -d -s task3-150-timefix -c "$PWD" "$CMD"
tail -n 30 supervisor.log
```

On success, retrieve `issue150-evidence.tar.gz`, its adjacent manifest, **and**
all `resume-*.jsonl` / `recovery-*.jsonl` journals and both downloaded helpers.
The original exporter does not automatically include these external helpers or
journals; publish them as a companion recovery archive with its own checksum.
Preserve all originals until evidence review. Do not reset budgets, mark jobs
complete manually, or rerun training if recovery fails.

## Validation and ownership

`python -m pytest -q tests/test_resume_issue150.py` covers the real monitor's
round-trip failure, equivalent-time resume, retained CPU usage/deadline, changed
or naive times, changed limits, corrupted models/bindings, incomplete training,
active locks/workers, expired/exhausted budgets, read-only preflight, restored
runtime symbols on failure, and success-only analysis/export handoff. Synthetic
fixtures do not establish scientific performance or certify server completion.

AI assistance: Codex diagnosed the defect and prepared the adapter, tests and
handoff. Human review is separate from PR #156's storage-only approval. No merge
or scientific execution is performed during implementation.
