# Task 3 safe-attack penalty result (#163)

## Decision

All 50,000 training episodes and 1,760 evaluation episodes completed. The
registered decision is **fail / exploratory mixed or negative**. No arm or
replica is selected, and no automatic coin-collector continuation is allowed.
Task 2 capability remains cumulatively incomplete.

The treatment produced **no observed improvement over the matched control**.
Both arms eliminated at least one peaceful opponent in 11% of primary games,
versus 20% for the frozen parent. Their paired elimination difference was
0 percentage points, with registered 95% bootstrap interval [0, 0]. This
interval describes this observed matched experiment; it does not establish
that the reward change can never matter in other training conditions.

## Registered comparison and provenance

- Issue: [#163](https://github.com/1BlauNitrox/mle-final-project/issues/163);
  campaign/result PR: [#165](https://github.com/1BlauNitrox/mle-final-project/pull/165).
- Executed source: `1e18b9c659041b4044d04e9ec6563b98c8c471dc`.
  The archive preserves exact working-tree bytes, including recorded Windows
  line endings, and verifies their relationship to this immutable Git commit.
- [Original protocol](../2026-09-12-task3-safe-attack-penalty/config.yaml):
  five paired replicas per arm, exactly 5,000 training episodes each,
  standard DQN, shared legal-action mask and escape-preserving continuation.
- Control retains the inherited -0.5 wasteful-bomb penalty. Treatment sets
  `neutral_safe_attack_bombs=True`, omitting that penalty only for a
  framework-confirmed safe, crate-free opponent attack. No positive attack
  bonus, feature change, opponent change or Double DQN intervention was used.
- Both arms use fresh migration of provisional #91 A/r3. Parent SHA-256:
  `c0b6081acf5a733258005d46b9430f3ef50e403470a3bb5428661729f9aeedf7`;
  2,382,903 bytes; producing commit
  `cbd52be8392f5003a91c5600fda4efd544b48ec5`.
- Final checkpoint after exactly 5,000 episodes for every replica; no
  favorable checkpoint or seed was substituted. All ten artifact hashes,
  sizes and paths are in [result.json](result.json).
- Training world seeds: 1630001-1630005; agent seeds: 2630001-2630005.
- Evaluation: 20 common world/agent pairs per suite, each repeated exactly.
  World seeds: 1631101-1631120 (peaceful), 1631201-1631220 (solo classic),
  1631301-1631320 (coin heaven), 1631401-1631420 (loot crate).
  Agent seeds equal world seed + 1,000,000. Protected ranges remain unused.
- Frozen reference: 160 jobs. Control and treatment: 805 jobs each.
  Total: 1,770 completed jobs. Primary observations determine gameplay
  estimates; repeats are integrity checks, not independent samples.
- Uncertainty uses the registered crossed replica/world-pair bootstrap:
  10,000 resamples, seed 163, percentile 95% intervals. The shared reference
  is not counted as five independently trained models.

## Performance and retention

Control and treatment have identical primary gameplay outcomes. The table
therefore shows one value for both. Intervals are candidate minus reference;
collection rates are fractions of available coins, not raw coin counts.

| Metric | Frozen reference | Each trained arm | Difference, 95% interval |
| --- | ---: | ---: | --- |
| Peaceful elimination rate | 20% | 11% | -9 pp [-26, +6] |
| Peaceful first-place rate | 90% | 76% | -14 pp [-31, +2] |
| Peaceful mean score / score margin | 4.35 | 2.71 | -1.64 [-2.98, -0.29] for margin |
| Peaceful self-kill rate | 15% | 4% | Descriptive: -11 pp |
| Coin-heaven collection | 46.60% | 37.34% | -9.26 pp [-30.50, +12.64] |
| Solo-classic collection | 27.22% | 15.78% | -11.44 pp [-23.56, -0.78] |
| Loot-crate collection | 27.70% | 8.64% | -19.06 pp [-30.46, -8.30] |
| Solo-classic crates per episode | 39.80 | 28.98 | See scaled retention contrast below |
| Loot-crate crates per episode | 40.15 | 15.31 | See scaled retention contrast below |

[summary.csv](summary.csv) contains score, survival, coins, crates, bombs,
invalid actions and pooled primary decision-time statistics for every suite.
[replicas.csv](replicas.csv) preserves between-replica variation. Peaceful
elimination rates were 15%, 10%, 5%, 10%, 15% in both arms. Full per-episode
observations, including repeats, are in [evaluation.csv](evaluation.csv).

[contrasts.csv](contrasts.csv) contains every retention interval. Its crate
contrasts are **candidate minus 0.9 times reference**, as prospectively registered,
not ordinary differences. Solo-classic crate contrast: -6.84
[-19.27, +5.44]; loot-crate: -20.825 [-32.91, -9.30].

## Every gate, without excluding failures

[gates.csv](gates.csv) lists every arm/suite gate separately.

| Gate | Control | Treatment |
| --- | --- | --- |
| Elimination at least 60% | Fail | Fail |
| Elimination improvement at least 20 pp, positive lower CI | Fail | Fail |
| Treatment benefit at least 10 pp, positive lower CI | Not applicable | Fail |
| First place at least 60%; positive mean score margin | Pass | Pass |
| Positive lower CI for first-place and margin improvement | Fail | Fail |
| Peaceful self-kill at most 10% | Pass | Pass |
| Task 1 coin-heaven collection retention (2 pp margin) | Fail | Fail |
| Task 2 classic collection retention (3 pp margin) | Fail | Fail |
| Task 2 loot-crate collection retention (3 pp margin) | Fail | Fail |
| Classic and loot-crate crate retention (90% of reference) | Fail | Fail |
| Survival retention, every regression suite (5 pp margin) | Pass | Pass |
| Self-kill retention, every regression suite (2 pp margin) | Pass | Pass |
| Invalid-action rate below 1%, every suite and replica | Pass | Pass |
| Every episode p95 below 50 ms and maximum below 100 ms | Fail | Pass |

The deterministic-repeat gate fails for **1 of 880 pairs**: control r5,
loot-crate world seed 1631403, agent seed 2631403. Primary/repeat
`attempted_actions` are 400/399 and `action_wait` 390/389. Other non-latency
fields, including executed-action sequence hash and native game outcomes,
match. Primary maximum latency is 7.17 ms; repeat maximum is **1,184.50 ms**.
This is consistent with a missed callback and default WAIT after the decision
deadline, but the cause of the delay is unproven. It remains a runtime and
repeat failure. No seed was removed or replaced.

The original analyzer correctly stopped on that mismatch. The result verifier
retains the diagnostic discrepancy and computes the unchanged registered
metrics with all observations. Default strict repeat validation is unchanged;
the diagnostic result cannot select or promote a checkpoint.

Source fingerprints, parent binding, checkpoint modes/counts, seed matrix,
raw-statistics/CSV agreement and completed-job preservation passed. These
checks do not turn a failed deterministic-repeat gate into an integrity pass.

## Treatment activation diagnosis and proposed follow-up

All five paired final payloads differ only in
`config.neutral_safe_attack_bombs`: online/target parameters, optimizer, replay,
RNG and counters are identical. All 25,000 paired training episode records
match after excluding arm/alias/artifact identifiers and timing measurements.
This includes shaped rewards, losses and executed-action hashes.

The five retained control replay buffers contain 50,000 transitions in total,
with zero safe, crate-free attack states and zero eligible bomb actions.
Only three states had the safe-attack feature set at all. These are bounded
trailing replay buffers, not all training transitions. No complete-run
counter of actual reward exemptions was recorded. The evidence supports
lack of observed effective treatment exposure; it does not establish the
full-run exemption count or a universal causal explanation.

**Recommended next step:**
first run a bounded attack-opportunity and reward-activation diagnostic.
Instrument eligible public-state opportunities, framework-confirmed eligible
placements and actual penalty exemptions. Use forced eligible-state contract
checks to verify the reward difference, then a prospectively fixed short pilot
to measure whether the training setup provides those opportunities. Register
its seeds, episode count, activation decision rule and complete compute budget
before any scientific pilot; do not choose them after looking at the outcome.
A separate reproduction of the failing timing pair may diagnose latency, but
cannot replace this experiment's failed observation.

Do not repeat this reward sweep with a larger budget without evidence that
it activates. For the next substantial training comparison, prioritize the
existing [#126 / PR #149](https://github.com/1BlauNitrox/mle-final-project/pull/149)
strong-opponent distribution preparation under the unmerged
[#145 roadmap](https://github.com/1BlauNitrox/mle-final-project/pull/145): fixed
rule-based opponents versus a fixed opponent mixture. Its current parent
requirement is unmet. An exploratory parent choice needs an explicit owner
decision and prospective registration, together with a bound launcher, raw
verification, regression suites and a measured full-workload allocation.
Do not silently select one of these failed #163 checkpoints as its parent.

We have a reproducible exploratory Task 3 reference and matched control.
We do not yet have a Task 3 model that passes the cumulative hunting and
Task 1/2 retention criteria. No anti-loop rule, lowered gate, extra favorable
seed or claim of Task 2 completion follows from this result.

## Evidence and reproduction

[evidence.json](evidence.json) records the compact archive checksum, size,
contents and publication state. Both assets are public at
[issue163-evidence-v1](https://github.com/1BlauNitrox/mle-final-project/releases/tag/issue163-evidence-v1).
A fresh download passed archive/manifest and all 6,016 member checks.

From the result revision, install the documented development dependencies and
retrieve `issue163-evidence-v1.tar.gz` plus its `.manifest.json` into one folder.
The following commands verify every archived file, extract to a new directory,
validate the exact executed source against Git, and recompute all findings:

```bash
python scripts/package_issue163_evidence.py --archive issue163-evidence-v1.tar.gz --extract evidence163
python -m training.verify_issue163_results --root evidence163/campaign-root --source evidence163/source --output reproduced163
```

Do not run the original `task3_attack_campaign verify` on this diagnostic
wrapper: it deliberately has a different schema and retains a failed repeat.
The full archive includes the native per-episode statistics and all recorded
latencies, exact executed source, ten final training artifacts, parent binding,
original/amended authorizations and recovery evidence. Recomputed result JSON,
scalar observations and tables must match the committed technical outputs.
Raw logs, replay payloads and large artifacts stay outside Git.
