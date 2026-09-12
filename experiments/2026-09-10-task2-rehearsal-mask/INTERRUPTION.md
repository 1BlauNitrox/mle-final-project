# Interrupted campaign: September 10, 2026

Refs #124. This is an operational partial record, not a completed experiment.
No scientific selection or Task 2 completion is supported.

## Verified outcome

At 21:38:05 Europe/Berlin the campaign stopped because system-wide free RAM
fell below the registered 1 GiB floor. The campaign's own recorded peak was
1,350,107,136 bytes (about 1.26 GiB), below its 8 GiB aggregate ceiling.
This establishes a system-memory guard breach; it does not identify which
other process consumed the memory. CPU consumption was 45.44 hours and elapsed
wall time 12.64 hours, within the amended 96 CPU-hour / 24-hour ceilings.

| Arm | Curriculum | Legal mask | Completed 500-episode blocks |
| --- | --- | --- | --- |
| A | blocked | off | 100/100 |
| B | interleaved | off | 95/100 |
| C | blocked | on | 100/100 |
| D | interleaved | on | 90/100 |

385/400 blocks completed: 192,500/200,000 planned training episodes. Partial
attempt episodes are not counted as completed blocks. B/r5/block16 and
D/r5/block11 stopped; subsequent blocks remain pending. The earlier Windows
file-lock failure at B/r2/block16 was successfully recovered by attempt 003,
which finished at 17:41:20 Europe/Berlin. Failed attempts remain retained.

All 5,120 planned evaluation episodes remain unexecuted: the supervisor's
training barrier prevented evaluation until every training arm completed.
Therefore there are no held-out collection, survival, self-kill, latency,
reproducibility or treatment-effect results from this campaign.

## What was tested

The registered 2x2 design asks whether interleaving visible-coin practice
preserves navigation and whether legal-action masking improves crate-board
collection when timed escape features are active. All arms enable escape
features and uniform replay; rewards, architecture, optimizer, initial weights
and total scenario budgets are fixed. Each arm has five paired replicas with
2,000 coin-heaven, 2,000 loot-crate and 6,000 classic training episodes each.
This tests practice order and masking, not the value of escape features alone.

The four registered primary contrasts require at least a 0.10 collection
fraction improvement with positive multiplicity-adjusted bootstrap lower
bounds, plus non-regression guards. Existing absolute Task 1/2 gates still
apply. Training reward cannot replace these held-out comparisons.

## Analysis and evidence

The amendment-aware registered analysis was invoked without launching games:

```powershell
python C:\path\to\issue124\resume_issue124_amendment.py --execution-root C:\path\to\issue124-execution --campaign-root C:\path\to\issue124 --analyze-only
```

It exited with `ValueError: Resource record incomplete, changed or breached`.
No result or selected artifact was produced. `interruption-evidence.json`
retains compact per-training-job statuses, attempt histories, artifact hashes,
source-status hashes and the resource/status records supporting this report.
It is operational evidence only; it does not contain checkpoint bytes or
scientific observations and cannot establish model performance.

## Recovery and decision

Preserve all campaign files, successful checkpoints and original ledgers.
Do not clear `limit_reached`, reset the clock or bypass the shared supervisor.
A documented technical recovery amendment must preserve the breach/history
and cumulative accounting, define how resource verification distinguishes
interrupted and resumed segments, and retain the original experiment design.
The original amended wall deadline is September 11 at 08:59:45 Europe/Berlin;
any extension beyond it requires an explicit budget amendment.

At the follow-up check around 23:16, only about 1.41 GiB system RAM was free,
below the required 4 GiB startup floor. Free memory before attempting recovery.
The remaining work is 15 complete blocks (7,500 episodes), plus all evaluation
and analysis; discarded partial attempts must be replayed from completed stage
checkpoints. Completion before the deadline is not established.

Task 2 remains open. No optimization is selected or applied. If complete valid
evaluation later passes the registered treatment and absolute gates, adopt the
mechanically selected checkpoint/configuration and bind Task 3 to it. If gates
fail, retain that negative result and use only an explicitly exploratory Task 3
predecessor while deciding whether a focused Task 2 follow-up is warranted.
