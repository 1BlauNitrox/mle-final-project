# Discount-horizon result (#91)

The full registered comparison completed: two arms, five paired replicas per arm,
100,000 training episodes, 1,360 primary evaluation observations and 1,360 matched
repeats. The decision is **retain gamma 0.90**. Gamma 0.97 did not improve this
agent under the registered curriculum. Neither arm completes cumulative Task 1/2
requirements. This is a completed negative experiment, not a promoted model.

## Results

| Gamma | Classic collection | Coin-heaven collection | Loot-crate collection | Classic self-kills |
| --- | ---: | ---: | ---: | ---: |
| A: 0.90 | 33.44% | 16.21% | 18.42% | 9.0% |
| B: 0.97 | 19.06% | 6.81% | 13.04% | 11.0% |

Collection uses all board coins, including initially hidden coins. Each
model/scenario contributes forty primary world seeds. Replicas have equal weight;
repeats are not independent observations. Per-replica metrics and all action,
survival, coin/crate and latency diagnostics are retained in server-results/summary.csv.

The registered primary B-A classic contrast is **-14.39 percentage points**,
95% paired hierarchical bootstrap interval **[-23.00, -4.83]**, 10,000 resamples.
It fails the +10-point minimum and positive lower bound. Coin-heaven collection
also falls by 9.40 points (95% interval [-19.70, -1.22]); its safety/retention guard
fails. Retain negative outcomes rather than try another gamma post hoc.

A meets the 30% classic collection threshold but fails aggregate and per-replica
invalid-action gates. B fails the classic collection threshold and a per-replica
invalid-action gate. Both fail Task 1 retention, invalid-action and no-bombs
requirements. The frozen Task 1 reference collects 76.50% on these coin-heaven
seeds, versus A's 16.21% and B's 6.81%. All deterministic and latency checks pass.
No favorable subset or replica can override the cumulative decision.

Mechanical selection is A/r3, the median classic replica among A's five models.
Its final checkpoint is 2,382,903 bytes, SHA-256
`c0b6081acf5a733258005d46b9430f3ef50e403470a3bb5428661729f9aeedf7`.
It is an exploratory predecessor only; no committed agent default or submission
checkpoint has been replaced. This run does not rank A against #124's A or C:
those studies use different seeds and execution details.

## What this tells us

The tested larger discount factor increases value scale without improving
collection. Mean absolute Q on the fixed final-stage probes averages 12.39 for A
and 19.82 for B. These are diagnostic values, not policy-quality metrics. All
stage gamma settings match the assigned arm; online and target initial weights
are identical, the initial replay buffers are empty, and only gamma differs in
initial configuration. This rules out the suspected silent gamma-reset failure.
It does not establish a universal optimal gamma or prove the mechanism behind
poor navigation. The controls still forget much of Task 1 and often choose
invalid actions. Raising gamma is not a demonstrated fix for those problems.

Together with #124's negative rehearsal result, this supports stopping these two
interventions and progressing with explicitly provisional Task 3 work and strict
regression checks. Masking from #124 remains an exploratory diagnostic candidate,
not a retrospectively adopted winner. Any further navigation intervention needs
a new controlled protocol; do not prolong this completed experiment.

## Integrity and execution

Exact execution commit: `cbd52be8392f5003a91c5600fda4efd544b48ec5`.
Server metadata records **Python 3.14.4**, not the recommended local 3.13 target.
The actual completed run is verified; this is not a blanket compatibility claim
for other 3.14 versions. Exact dependencies and source fingerprints are retained
in the protocol and every job's metadata in the external evidence package.

The first launch failed before jobs because the training-only runner API was
missing. The corrected campaign retained the original authorization time, CPU
usage and failure records. First corrected training jobs began about 08:09 UTC
on September 11. Peer approval of the execution SHA was recorded at 08:49 UTC,
after training had started. This is a disclosed pre-run-review workflow deviation;
it is not retroactively described as prior approval. The owner launched the run.
No outcome-dependent seed, threshold, treatment or budget change is made here.

All 2,750 scientific jobs completed without failed attempts. Final ledger:
36,153.16 CPU seconds, 22,412.36 wall seconds (6h 13m 32s including startup recovery),
1,029,009,408-byte peak campaign RAM, no active workers and no breached limit.
Every retained evidence-manifest hash matches, all 32 checkpoint hashes match,
all ten final models contain exactly 10,000 training episodes, and all stage
loss/TD means reproduce from episode diagnostics. Primary/repeat columns match
for every seed and every episode passes latency checks.

The portable verifier reproduces summaries, bootstrap intervals, all absolute
gates and selection. Q probes are recomputed with rtol=1e-5 and atol=1e-5 for
cross-platform floating-point differences; maximum observed absolute difference
is 1.90735e-5. Gamma, hashes, budgets and deterministic game observations are not
compared with relaxed tolerances. Git blob hashes, rather than Windows checkout
line endings, bind the Linux protocol configuration.

## Durable evidence and exact verification

[Evidence release](https://github.com/1BlauNitrox/mle-final-project/releases/tag/issue91-evidence-v1)
contains the lossless required episode CSVs, metadata, protocol, resource and
startup-recovery records, original server analysis, two initial and thirty stage
checkpoints. Redundant per-episode agent copies and raw logs are omitted.
Archive: `issue91-complete-evidence-v1.tar.gz`, **19,134,828 bytes**.
SHA-256: `9e1ade26006539fec7fa92d20da0ba7a16922b86174b06d71946e00c08e53efe`.

```bash
gh release download issue91-evidence-v1 --repo 1BlauNitrox/mle-final-project --pattern issue91-complete-evidence-v1.tar.gz --dir training_outputs/issue91-evidence
tar -xzf training_outputs/issue91-evidence/issue91-complete-evidence-v1.tar.gz -C training_outputs/issue91-evidence
python -m training.verify_issue91_results --campaign-root training_outputs/issue91-evidence/issue91-fixed --output training_outputs/issue91-evidence/verification.json
```

Run this on the results PR checkout with documented development dependencies;
it performs no training or evaluation games. It verifies checkpoint bytes against
the server manifest and reproduces the reported metrics and uncertainty.
Committed server-results/ contains the primary observations, per-replica summaries,
training diagnostics, original result and independent verification report.
The external package retains repeat observations and complete provenance.

The experiment needs no further compute. Final result review and acceptance of
the disclosed workflow deviation remain owner/reviewer work; this PR retains
Refs #91 pending that decision. Keep source data until peer review is complete.
AI assistance: Codex performed artifact verification, statistics reproduction,
portable-verifier implementation and documentation; evidence is the retained run.
