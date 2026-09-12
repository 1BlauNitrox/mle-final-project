# Reproduced stalls: a diagnostic finding, not a treatment result

At source `94a3acca16944ca95f57ba13b70bc8f7f1307c15`, all 22 registered
diagnostic episodes completed in 179 seconds of supervisor wall time. Two
additional exact repeats matched every recorded state, action, feature,
Q-value and reward component. One earlier episode at `3676d5c` completed its
game but failed export because the diagnostic passed `save_stats=None`;
the corrected explicit output path is instrumentation-only. Its output directory
was preserved; no scientific seed was discarded or model changed. Total new
games including that failed export: 25. No training occurred.

Thirteen of the 22 episodes met the prospectively defined conservative stall
criterion: parent 1/2, control 4/10, masked 8/10. These tiny diagnostic samples
are not estimates of population frequency or a causal masked-control comparison.
The exact GUI round is unknown, so reproducing similar behavior does not establish
that these are the user's original states. All model hashes match the registered
#91 parent/#147 inventory; originals were checked unchanged after each episode.

## Concrete observations

- Parent, world1460002/agent2460002, solo classic: repeated WAIT at (3,14),
  beginning at step100. At step101, WAIT Q=3.6889 exceeds UP Q=3.2856 although
  UP is legal. There are no active board hazards and WAIT receives -0.1.
  This is a learned stationary stall, not necessary bomb waiting. Exact repeat passed.
- Control r1, world1460001/agent2460001, solo: RIGHT is the greedy action into
  a blocked boundary from (15,1), producing INVALID_ACTION throughout the round.
  RIGHT Q=2.64394 versus legal DOWN Q=2.60142. No implementation defect in the
  unmasked policy is implied; this is precisely the invalid-action limitation.
- Masked r1, world1460001/agent2460001, solo: legal DOWN/UP alternation between
  (15,4) and (15,5), stall window112-135 and continued afterward. The board is
  hazard-free and unchanged, there are no visible coins, and each movement has
  zero immediate reward. Exact repeat passed. Other masked replicas also show
  legal movement cycles and one stationary WAIT case.
- Different positions sometimes share feature vectors, but the representative
  two-tile cycle also has different local feature vectors at its endpoints.
  Feature aliasing alone is therefore not established as the cause.
- Crate-free bombs are classified WASTEFUL_BOMB_PLACED even if an opponent is
  in the blast footprint: this is an inherited source-code property. This
  diagnostic did not establish that it causes the observed stalls.

The absence of visible coins in representative cycles gives no support for
reviving #90 coin-distance shaping as a cure for those states. Positive Q-values
on zero-immediate-reward cycles suggest inspecting value learning. Offline
inspection of final online/target networks found action disagreement on some
recorded states, so standard and Double DQN targets can differ. Neither that
disagreement nor a positive Q-value proves overestimation relative to optimal
return: other future actions, training exploration and approximation matter.

The proposed bounded follow-up is #150: standard versus Double DQN targets,
keeping masks, features, rewards, optimizer, budgets and every capability gate
fixed. It is a falsifiable hypothesis about target learning, not a confirmed
fix, an anti-loop override or a change of learned-model family. A failure must
remain negative and cannot select a favorable checkpoint.

## Retained evidence

[summary.json](summary.json) records each input/source hash, seeds, first detected
window and complete overlapping-window count. Full public-state trajectories,
all detected windows and framework statistics (including two repeats) are in
[issue146-evidence-v1](https://github.com/1BlauNitrox/mle-final-project/releases/tag/issue146-evidence-v1).
Archive `issue146-evidence-v1.tar.gz`: 808,976 bytes, SHA-256
`57d7271a9238b18eab3d44a364cd2ddac3a2a91720bee0aa4adee80dc4a63dd6`.
The adjacent manifest maps every member to SHA-256 and size. Model inputs remain
retrievable from issue147-evidence-v1; no model is selected by this diagnosis.

```bash
gh release download issue146-evidence-v1 --repo 1BlauNitrox/mle-final-project --dir inputs-146
echo '57d7271a9238b18eab3d44a364cd2ddac3a2a91720bee0aa4adee80dc4a63dd6  inputs-146/issue146-evidence-v1.tar.gz' | sha256sum -c -
```

Use `training.diagnose_dqn_stalls.stall_windows` on the decompressed trajectory
list to reproduce each recorded window. Each gzip trajectory contains a JSON list
of per-action records; `state` is the full public pre-action state, `features`
and `normalized_features` are network inputs, `q_values` follow UP/RIGHT/DOWN/
LEFT/WAIT/BOMB order, and `legal_mask`, `action`, `events`, `reward_components`
describe the actual policy/environment transition. No heuristic changes actions.
Instrumented wall time is not tournament latency evidence. Global CPU accounting
was not instrumented; the measured serial wall time and per-episode checks
document the bounded diagnostic workload, not scientific campaign accounting.

Validation: 875 tests and 11 subtests passed locally. Human review must assess
the retained data and interpretation. Refs #146: the specific original GUI event
and the ultimate learned cause remain unresolved; no corrective behavior is adopted.
AI assistance is disclosed in docs/0006-ai-usage.md.
