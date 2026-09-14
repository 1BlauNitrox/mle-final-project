# Task 3 escape-preserving successor and exploratory baseline

Refs #125, #108 and #109. The 39-input implementation is complete in this
preparation branch; the final #124 parent binding and scientific Task 3
experiment are not complete.

## Implemented contract

- All 26 Task 2 inputs are preserved, including active timed escape features.
- The existing thirteen opponent descriptors form suffix indices 26-38.
- Both parent networks preserve all six Q-values, with zero new columns.
- Parent action masking, escape mode and applicable hyperparameters persist
  into acting, Bellman targets, checkpoint loading and fresh training.
- Explicit mode mismatches fail. Legacy 21-input and 26-input parents are
  supported; 34-input Task 3 artifacts are rejected under schema 4.
- Migration requires a parent hash, prohibits source overwrite, and resets
  optimizer, replay, exploration/RNG and Task 3 episode/update counts.
- The committed checkpoint remains a reproducible #85 compatibility fixture.
  Frozen Task 1/2 agents and the running #124 checkout are unchanged.

## Baseline available before #124 finishes

The verified #107 selection was A/r2, with Task 2 incomplete. It is suitable
only as an explicitly exploratory historical baseline while #124 runs:

- Parent SHA-256: `d571b199bc00a0419d16706cc5f7844364a184384058268b2cc8de5b79bb479d`
- Parent size: 2,382,903 bytes.
- Producing source: `69e2a931e4ebfd358cc0128049516b1a12adc2a8`.
- Durable source: [issue107-evidence-v1 release](https://github.com/1BlauNitrox/mle-final-project/releases/tag/issue107-evidence-v1).
- Archive SHA-256: `ef2c0fac3db0e5b72a9ea9ce89599f726a495a7a84ba33bb9d9b3f4342fc5748`.
- Archive member: `run-plans/issue107-cell-a-control/artifacts/r2/classic-crates/checkpoint.pt`.

Prepare into a new directory (the tool refuses an existing output directory):

```powershell
python scripts/prepare_task3_baseline.py --parent C:\path\to\parent.pt --parent-sha256 d571b199bc00a0419d16706cc5f7844364a184384058268b2cc8de5b79bb479d --parent-source-commit 69e2a931e4ebfd358cc0128049516b1a12adc2a8 --output-dir C:\path\to\task3-baseline
```

The directory contains the preserved Task 2 parent, the 39-input Task 3
migration, `binding.json`, matched `candidate.yaml` / `reference.yaml`, and
`smoke.yaml`. The migration's two new-network states retain their corresponding
parent online/target values. Every binding is labeled exploratory and does not
claim Task 2 completion or authorize a scientific campaign.

For bounded integration only, `smoke.yaml` runs one peaceful training episode,
one coin-collector training episode, then four evaluation episodes. It uses
temporary agent workspaces and never trains the baseline file in place:

```powershell
python -m training.run_plan C:\path\to\task3-baseline\smoke.yaml --output-root C:\path\to\new-smoke-output
```

The locally prepared baseline is under the main checkout's ignored
`training_outputs/issue125-task3-baseline/`. Required scientific evidence must
be published before a performance claim; this path alone is not evidence.

## Final handoff

After #124 finishes, verify its complete analysis and mechanical selection.
Use the selected artifact, never a visually promising checkpoint, and preserve
its failed Task 2 gates. Rebuild/bind both Task 3 and frozen-reference plans,
including mode flags and checksums, from the same parent. #109/#120 remains the
single peaceful-opponent protocol; do not silently run the historical baseline
as the #124 successor comparison.

Scientific launch still needs agreed numerical elimination/score/self-kill/
retention rules, the corresponding verified analyzer, fresh/collision-audited
seed populations, a shared resource budget and explicit compute authorization.
The current #109 templates retain five replicas x 10,000 episodes; no shorter
pilot or altered thresholds are silently adopted. The coin-collector template
remains gated on the peaceful-stage decision. No Task 3 scientific run starts
in this implementation issue.

## Implementation validation

On 2026-09-10, the complete local suite passed: 758 tests and 11 subtests.
Scoped Ruff checks, Python compilation and agent packaging passed. The generated
historical baseline completed the six-episode smoke plan (two training and four
evaluation episodes). These checks establish execution compatibility, not game
strength. Official Docker compatibility, clean upstream execution and measured
latency/memory remain required before a submission-readiness claim.
