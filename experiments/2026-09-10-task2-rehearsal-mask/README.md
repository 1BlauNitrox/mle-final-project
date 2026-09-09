# Escape-enabled Task 2 rehearsal and masking preparation

Refs #124; prerequisite #123. **Draft, not ready to execute.** The user requested
local Windows execution on September 10. No scientific run has started.

| Arm | Task 1 practice | Legal-action masking |
| --- | --- | --- |
| A | blocked | none |
| B | interleaved | none |
| C | blocked | framework_legal |
| D | interleaved | framework_legal |

All arms use escape-continuation inputs and uniform replay. Rewards, network,
optimizer, epsilon schedule and source weights are fixed. This does not adopt
masking: #86 rejected it; the combination with active escape inputs is a new
hypothesis. #107's negative protected-replay result remains unchanged.

## Concrete proposed matrix

Each arm has five paired replicas and twenty 500-episode blocks: 2,000
coin-heaven, 2,000 loot-crate and 6,000 classic episodes per replica. Blocked
arms run four coin blocks, four loot blocks and twelve classic blocks.
Interleaved arms use coin blocks at positions 1, 6, 11 and 16, loot blocks at
positions 2-5, and classic for the other twelve. Every arm has identical process
boundary counts. World-seed offsets pair the same scenario occurrence across
arms; they do not replay the same initial seed at every later practice block.
The intervention changes experience order, including the exploration level at
which a scenario is encountered; it does not claim identical trajectories.

Training root pairs are 124001-124005 / 224001-224005. The fresh development
world/agent ranges are 324001-324040 / 424001-424040 (classic),
325001-325040 / 425001-425040 (coin-heaven), and
326001-326040 / 426001-426040 (loot-crate). Repeats reuse these pairs and are
not independent observations. Frozen Task 1 and untrained migration references
receive matching suites. Final and confirmation seeds stay unopened.

Total proposal: 200,000 training and 5,120 evaluation episodes, 400 training
stage jobs. Select only the final-budget checkpoint. The new control is a
fresh matched run, not a claim that the reblocked curriculum equals historical
#107 execution. The source is the committed fresh 26-input migration,
SHA-256 `4ad409472e7ca008dfcc82aa26017017c90259b65f9e593020d1b63921430f60`.

## Decision and launch work still required

- Confirm four arms or reduce to A/B before execution, with a correspondingly
  revised prospective budget and multiplicity family.
- Carry every #107 absolute Task 1/2 gate unchanged. Ratify the new efficacy,
  non-regression, multiplicity, ranking and fallback rules; implement and test
  their analyzer before training. Existing #107 factor-specific analyzer must
  not be run on these newly defined treatments.
- Confirm wall/CPU/RAM ceilings and reviewer. The PC has 16 logical CPUs and
  about 16 GiB RAM, but only about 2 GiB was free during setup. Do not assume
  four workers fit without a fresh memory check. Proposed ceiling: four
  workers, 8 GiB campaign RAM, ten wall hours; not yet ratified.
- Implement a campaign-wide resource monitor and detached Windows supervisor
  with persistent PID/status/log paths, clean reviewed SHA checks and resume
  validation. Enforce one global worker limit; do not run all four plans with
  four workers each. Keep scientific evaluation serial and uncontended.
- Validate a short isolated integration smoke and all CI; obtain non-author
  review. No training defaults or final agent artifact are replaced.

Only a dry-run command is currently valid:

```powershell
python -m training.run_plan training/run_plans/issue124-cell-a.yaml --dry-run
```

Do not omit `--dry-run`: the generic runner does not enforce the proposed
shared campaign budget. A detached launch command will be recorded only after
the missing runner/analyzer and resource decisions are complete.

## Deadline priorities

Finish this bounded Task 2 attempt and select mechanically; do not chase an
unlimited sequence of sweeps. Prepare #125's escape-preserving Task 3 migration
concurrently, then complete existing #109/#120 and progress to #51. Target
Task 3 work on September 10-12, Task 4 (#126) on September 13-16, optional
compatibility on September 17, candidate freeze by September 19, and final
verification/submission (#127) before September 21 at 21:00 Europe/Berlin.
These are targets, not a promise of completed learning milestones.
