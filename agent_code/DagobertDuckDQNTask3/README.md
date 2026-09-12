# DagobertDuckDQNTask3

Status: schema-4 implementation under #125 / PR #130; the completed #109
exploratory campaign failed the peaceful-stage gates. No trained Task 3
replica is selected. The committed checkpoint remains a compatibility fixture.
See the [result and durable evidence](../../experiments/2026-09-08-dqn-task3-peaceful-opponent/RESULTS.md).

Five 10,000-episode replicas from explicitly authorized provisional #91 A/r3
achieved 20.5% elimination (parent 17.5%; paired improvement 95% CI
[-12.5, 18.5] percentage points), below the required 60%. Self-kills were
11.5%, invalid actions 6.79%, and classic collection retention failed.
The registered decision prohibits coin-collector continuation. Runtime passed;
this neither completes Task 2 nor establishes tournament readiness.

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
python scripts/migrate_task3_dqn_successor.py --parent-sha256 3edb2e7196030fcb52af6c7dc9ee69d9fc1259898ea674002fe06fbe93468015 --output training_outputs/task3-fixture-check.pt
```

Migration supports resumable checkpoints with 21 and 26 parent inputs, including
active escape features. Existing outputs are refused; evaluation-only exports
cannot supply the inherited target network and are rejected explicitly.
All available columns and both online/target networks are preserved; new
columns are zero. Old 34-input Task 3 artifacts require explicit migration or
regeneration and cannot be loaded under the new schema.

## Baselines and next steps

The Task 2 parent is explicitly configurable. Historical #107 A/r2 can support
the exploratory #109 study now; a later registered Task 2 selection requires
a new binding, with failed gates retained explicitly. Neither choice certifies
Task 2 completion. The [#109 protocol](../../experiments/2026-09-08-dqn-task3-peaceful-opponent/README.md)
registers user-approved numerical gates, five 10,000-episode replicas, matched
40-pair peaceful/classic/coin-heaven/loot-crate evaluations with repeats, a
paired analyzer and guarded launcher. Fresh review and separate compute
authorization remain required. The [#137 continuation](../../experiments/2026-09-11-task3-coincollector/README.md)
requires the passing peaceful decision and its mechanically selected training
checkpoint, retains both networks, and resets training state. It uses the common
launcher/analyzer with `--protocol coincollector`; #51 retains its validated-parent scope.

Unit/contract tests cover prefix equivalence, active-feature Q preservation,
separate online/target weights, masking consistency, reward attribution,
source checksums and read-only evaluation. Integration smoke and packaging
checks do not establish game strength, learning convergence or tournament
latency on the official hardware. There are no Task 3 means, confidence
intervals or success claims yet.
