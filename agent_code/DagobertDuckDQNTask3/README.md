# DagobertDuckDQNTask3

Status: schema-4 implementation and provisional compatibility fixture under
#125 / PR #130. No Task 3 scientific training result is claimed.

## Model and hypothesis

A self-contained CPU DQN extends the Task 2 policy with public opponent
positions and attack context to learn hunting, while retaining navigation,
crate destruction and escape behavior. Architecture: `39 -> 64 -> 64 -> 6`.
Actions: `UP RIGHT DOWN LEFT WAIT BOMB`. Evaluation uses one CPU thread and
agent-relative paths; no cross-agent imports or multiprocessing.

Inputs 0-25 exactly match Task 2, including all five escape-continuation
features. Inputs 26-38 are the existing thirteen opponent descriptors.
The full order, normalization and attribution rules are in
[the Task 3 contract](../../docs/0008-task-3-opponent-awareness-contract.md).
Parent masking and escape settings are preserved in acting, training and
persistence; explicit environment mismatches fail. Fresh training resets replay,
optimizer, epsilon, RNG and update counts, while preserving both parent networks.

Inherited rewards are unchanged; attributable `KILLED_OPPONENT` adds +5 per
native event. Third-party elimination events add no reward. Exploration is
seeded epsilon-greedy (default 1.0, decay 0.9997, floor 0.1); Adam learning rate
0.0005, gamma 0.9, batch 64, replay 10,000, warm-up 500, target interval 500,
gradient clip 10. Parent migration preserves applicable parent hyperparameters.
Dependencies: NumPy and CPU PyTorch, declared in `requirements.txt`.

## Provisional fixture and reproducibility

The committed `checkpoint.pt` is a fresh compatibility migration of the
historical corrected #85 Task 2 checkpoint, not a trained Task 3 candidate.
Its SHA-256, byte size and lineage are in `artifact.json` and
`parent-artifact.json`. Regenerate in an isolated worktree:

```powershell
python scripts/migrate_task3_dqn_successor.py --parent-sha256 3edb2e7196030fcb52af6c7dc9ee69d9fc1259898ea674002fe06fbe93468015
```

Migration supports 21 and 26 parent inputs, including active escape features.
All available columns and both online/target networks are preserved; new
columns are zero. Old 34-input Task 3 artifacts require explicit migration or
regeneration and cannot be loaded under the new schema.

## Baselines and next steps

The next scientific parent comes from #124's registered selection, with failed
Task 2 gates retained explicitly. A historical #107 A/r2 exploratory baseline
can be prepared separately; it does not certify Task 2 completion or replace
#124 selection. Existing #109/#120 provides matched peaceful, classic,
coin-heaven and loot-crate evaluation templates, followed later by #51's
coin-collector study. Numeric decisions and compute authorization remain
scientific launch requirements.

Unit/contract tests cover prefix equivalence, active-feature Q preservation,
separate online/target weights, masking consistency, reward attribution,
source checksums and read-only evaluation. Integration smoke and packaging
checks do not establish game strength, learning convergence or tournament
latency on the official hardware. There are no Task 3 means, confidence
intervals or success claims yet.
