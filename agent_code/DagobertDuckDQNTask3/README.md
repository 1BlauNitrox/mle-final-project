# DagobertDuckDQNTask3

Task 3 remains exploratory. No trained replica passes the cumulative hunting
and Task 1/2 retention gates, and no replica is selected. The committed
checkpoint is a compatibility fixture, not a trained Task 3 candidate.

## Experiment findings

| Experiment | Main result | Decision |
| --- | --- | --- |
| [#109 peaceful training](../../experiments/2026-09-08-dqn-task3-peaceful-opponent/RESULTS.md) | 20.5% elimination versus 17.5% for the parent; improvement CI [-12.5, +18.5] pp | Hunting and retention gates failed. |
| [#147 legal masking](../../experiments/2026-09-12-task3-legal-mask/RESULTS.md) | 21% masked versus 18% unmasked; difference CI [-14.5, +20.5] pp | Adoption and collection/crate-retention gates failed. |
| [#150 Double DQN](https://github.com/1BlauNitrox/mle-final-project/pull/161) | 23.5% elimination in each arm; difference CI [-18.5, +16.0] pp | No established Double DQN benefit; cumulative gates failed. |
| [#163 safe-attack penalty](../../experiments/2026-09-13-task3-safe-attack-results/README.md) | 11% elimination in each arm versus 20% for the parent; treatment difference CI [0, 0] pp | No observed treatment effect; hunting and retention gates failed. |

Intervals are registered 95% intervals from each experiment. Different studies
use different seeds and budgets; their percentages are not matched comparisons.
The linked records contain per-seed results, provenance and every gate.

In #163, all five paired models have identical learned weights and training
state apart from the treatment flag. The retained replay contains no qualifying
safe, crate-free attack actions; it covers only the trailing buffer, not the
whole training history. One of 880 evaluation repeat pairs failed, with a
1,184.5 ms control decision-time spike. That failure remains part of the result.
No default or checkpoint changes follow from either #150 or #163.

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

By default, inherited rewards are unchanged; attributable `KILLED_OPPONENT` adds +5 per
native event. Third-party elimination events add no reward. Exploration is
seeded epsilon-greedy (default 1.0, decay 0.9997, floor 0.1); Adam learning rate
0.0005, gamma 0.9, batch 64, replay 10,000, warm-up 500, target interval 500,
gradient clip 10. Parent migration preserves applicable parent hyperparameters.
Dependencies: NumPy and CPU PyTorch, declared in `requirements.txt`.

Two training-only experiment options default to false:
`double_dqn` selects the next legal action with the online network and evaluates
it with the target network; `neutral_safe_attack_bombs` omits the -0.5
wasteful-bomb penalty for confirmed safe, crate-free opponent attacks. Both flags
are persisted. Neither introduces an evaluation-time policy override.

## Provisional fixture and reproducibility

The committed `checkpoint.pt` is a compatibility migration of the
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

## Next steps and limitations

First measure attack opportunities and actual reward exemptions in a bounded,
prospectively registered diagnostic before another long reward experiment.
Keep Task 1/2 regression gates and the exact parent configurable. The completed
campaigns used explicitly provisional #91 A/r3; this does not certify Task 2
completion. #137 coin-collector continuation remains blocked by the failed
peaceful-stage decisions. #146's reported looping cause is still unknown.

Unit and contract tests cover migration, masking, reward attribution,
persistence and read-only evaluation. Passing these checks does not establish
successful hunting or tournament readiness. Seeds, hyperparameters, artifact
hashes and execution records remain in the linked experiment evidence.
