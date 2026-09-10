# Issue #124 runtime extension

On September 10, after observing that interleaved practice takes longer than
estimated, Julius requested: "can you not hard stop the runs at 6PM? let them
run longer". The operational extension uses a 24-hour total wall ceiling from
the original 08:59:45 Europe/Berlin start, and a 96 CPU-hour ceiling. Four
workers and 8 GiB aggregate RAM remain unchanged. The new deadline is
September 11 at 08:59:45 Europe/Berlin. These ceilings were selected to allow
the registered matrix to finish, not to add training episodes.

This is an amendment after training began, not the original prospective
budget. Preliminary training summaries had been inspected; no evaluation
result or treatment selection was available. Seeds, five replicas per arm,
20 blocks per replica, scenarios, rewards, episode budgets, decision thresholds
and checkpoint selection are unchanged. Report the budget deviation with the
scientific results.

The live supervisor cannot reload its deadline. An operational restart
preserves completed stages and interrupted attempt records. At most one
in-progress block per arm is replayed from its preceding saved checkpoint;
these attempts must not be silently excluded from execution/resource history.
The original authorization is preserved byte-for-byte. Resource snapshots,
process identities, conservative CPU accounting for the handover interval,
and the owner's instruction are retained in `budget-amendment.json` and
`budget-amendment-history/`. The original deadline epoch is not reset.

Execution source remains `1ce18c8736b6a50773b60b95ec0f11dd4c901028` in the
unchanged `issue124-execution` worktree. The separately committed
`scripts/resume_issue124_amendment.py` is copied to the output directory and
checksum-bound to the amendment. It changes only effective runtime limits and
normalizes the pinned monitor's UTC timestamp spelling for resume. It invokes
the original resume and analysis implementations. The analysis result and
evidence manifest include the amendment, wrapper and preserved ledgers.

Use the retained wrapper for subsequent resume or analysis, since the original
launcher correctly rejects the amended resource ceiling:

```powershell
python C:\path\to\issue124\resume_issue124_amendment.py --execution-root C:\path\to\issue124-execution --campaign-root C:\path\to\issue124
python C:\path\to\issue124\resume_issue124_amendment.py --execution-root C:\path\to\issue124-execution --campaign-root C:\path\to\issue124 --analyze-only
```

Do not launch a second supervisor while one is active. No peer approval or
merge is implied by the owner's execution/budget exception.

## Windows resource-ledger recovery

A transient `PermissionError` replacing `resources.json` interrupted B/r2's
block 16 at the first supervisor handover. The checkpoint and retained
preceding stage were not the failing files. The operational resume monitor
retries only `PermissionError`, up to 20 attempts separated by 0.1 seconds,
while retaining the same CPU/memory values and monitor lock. Persistent errors
and unrelated I/O errors remain fatal. This changes no learning code.

A subsequent campaign-only supervisor restart installs this fix and uses the
original resume path to recover incomplete stages. Completed stages are
skipped. Resource ledgers, interrupted attempts, the prior amendment wrapper
and amendment metadata remain in the evidence history. The 24-hour wall,
96 CPU-hour, four-worker and 8 GiB ceilings are unchanged.
