# Issue 150 disk exhaustion recovery

The owner-reported campaign stopped with `ENOSPC` on 2026-09-12: the shared
49 GiB filesystem had zero available space, while run outputs occupied 8.9 GiB.
Reference had 320 completed jobs; control 1,605; Double 564 completed, one left
marked running and 1,040 pending. All training jobs precede evaluation in each
arm, so the report indicates ten completed training jobs. Checkpoint integrity
still must be verified before accepting any scientific result.

The 8 GiB free-space recommendation was incorrect. Compact exported evidence
does not measure the working storage used by thousands of retained input
snapshots, each including the checkpoint/replay state. Warning-only logging
does not remove those copies. Codex prepared the incorrect estimate. Do not
repeat an unmodified full campaign based on that estimate.

Keep scientific source `6f014485a3026cc3707fa2cc3a379880dd0b74bd`, bindings,
all seeds, job records, artifacts and resource authorization unchanged. This
storage-only helper is downloaded outside the campaign checkout. It does not
modify the running campaign's source or relax a scientific gate.

`scripts/deduplicate_evaluation_inputs.py` defaults to a read-only audit. It
selects only checkpoint.pt input snapshots of completed evaluation attempts,
checks their hashes against recorded immutable artifacts, and groups identical
bytes with matching permissions, modification time, ownership and filesystem.
Application uses atomic hard links, preserving every path, byte, permission and
mtime. Inode/link counts and ctime change, and identical archived copies then
share storage. Neither users nor tools should later edit these archived inputs
in place. The runner never modifies an earlier completed attempt's input copy.

Training snapshots, incomplete attempts, temporary copies, live replica workspaces
and canonical final artifacts are excluded. Application holds the normal
campaign lock, flushes an audit record before each replacement, rechecks bytes
and attributes before/after, and verifies status/resource records remain intact.
An existing lock or audit is not overwritten. No model needs to be deleted.

Use only with a confirmed stopped campaign. The reported process listing had
no experiment process, and resource active_root_pids was empty. An interrupted
job can remain marked running; do not manually mark it completed. The unchanged
runner records abandoned attempts as interrupted when legitimately resumed.

Example commands, with a checksum-verified helper downloaded to the task root:

```bash
ROOT="$HOME/task3-issue150"
PY="$HOME/task3-issue109/repo/.venv/bin/python"
"$PY" "$ROOT/recover-inputs.py" --root "$ROOT/runs"
"$PY" "$ROOT/recover-inputs.py" --root "$ROOT/runs" --apply --audit "$ROOT/storage-recovery-1.jsonl"
df -h "$ROOT"
```

The original authorization started at 11:21:26.793224 UTC. Its ten-hour wall
ceiling expires at **21:21:26 UTC / 23:21:26 Europe/Berlin on September 12**.
Downtime still counts. Never reset authorization/resource records or increase
that ceiling. Before resuming, verify all ten final artifacts against completed
training records, unchanged replica artifacts, readable metadata and at least
the original launcher's 8 GiB available disk/RAM. The report leaves 1,041
evaluation jobs; use measured evaluation timing to assess the remaining window.

If prerequisites hold, use the existing external start.sh and `--resume`, with
a new tmux session name, from the unchanged scientific checkout. Completed jobs
are skipped and the abandoned evaluation gets a retained new attempt. If the
budget is no longer sufficient, preserve/export partial evidence and register
a separate owner-approved completion protocol; do not silently extend this run.

Validation: five unit tests cover dry-run, byte/path/attribute preservation,
idempotence, hash mismatch, active locks/workers, path escape and link failures.
A disposable copy of real 26-episode smoke outputs retained all 1,104 files'
hashes, sizes, modes and mtimes after 21 replacements; estimated savings were
17,616,375 bytes. Original smoke outputs were untouched. This validates storage
mechanics, not scientific efficacy or the server campaign's integrity.
