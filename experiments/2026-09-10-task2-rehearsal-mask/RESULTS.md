# Issue #124 reduced evaluation result

Refs #124 and PR #131. The September 11 Windows laptop evaluation completed all
4,880 registered reduced-scope episodes: 2,440 primary observations and 2,440
matched deterministic repeats. A/B/C have five trained replicas; D has four and
is exploratory only. Training completed 395/400 blocks (197,500 episodes); D/r5
lacks its last five blocks. This record does not complete the original full
factorial experiment and does not complete Task 2 for this DQN.

## Findings and decision

| Arm | Practice / masking | Classic collection | Classic self-kills | Coin-heaven collection |
| --- | --- | ---: | ---: | ---: |
| A | Blocked / off | 29.56% | 11.0% | 21.68% |
| B | Interleaved / off | 17.94% | 6.5% | 4.69% |
| C | Blocked / on | 34.89% | 0.5% | 42.68% |
| D (exploratory, n=4) | Interleaved / on | 32.64% | 1.25% | 13.25% |
| Frozen Task 1 reference | Reference | Not evaluated | Not evaluated | 74.80% |

Collection is the fraction of all board coins collected, including initially
hidden coins. Each trained-model/scenario group has forty primary world seeds;
repeats are never additional independent observations. Per-replica metrics,
including survival, action counts and latency, are in laptop-results/summary.csv.

The registered primary rehearsal contrast B-A on coin-heaven is -16.99 percentage
points (95% interval -24.77 to -9.03; four-test corrected 98.75% interval -26.82 to
-6.82). This schedule harmed retention. Reject this rehearsal treatment; do not
claim that all possible rehearsal schedules have been disproved.

The primary masking contrast C-A on classic is +5.33 points (95% interval -5.06
to +15.33; corrected interval -7.86 to +17.33). It fails the registered +10-point
minimum and positive corrected lower-bound requirements. C's descriptive safety
and collection results justify further diagnosis, but do not establish adoption
under this protocol. In particular, zero invalid actions alone is not success.

C passes all Task 2-specific absolute gates but fails Task 1 retention and the
coin-heaven no-bombs gate. Its 42.68% navigation collection remains below the
frozen reference's 74.80%. A fails the 30% classic collection threshold and a
per-replica invalid-action limit, as well as Task 1 gates. B also fails both task
groups. No A/B/C candidate passes cumulative completion requirements.

Neither B nor C is eligible, so the registered fallback is A, with median
classic replica r5. This is a mechanical exploratory predecessor, not a promoted
submission artifact. SHA-256:
`370c1a2c66aac5d9e02c5599006f8d8f3648ff987e00865daa84ad266f516a54`,
2,382,903 bytes. No committed agent checkpoint or default has been replaced.

D is excluded from confirmatory selection. Its conditional comparisons and
interaction are reported separately using matched r1-r4 from all four arms;
they cannot override the A/B/C decision or silently complete the missing seed.

## Integrity and limitations

All 4,880 jobs are complete. Primary/repeat deterministic columns match for every
pair, all decision-time checks pass, and all final evaluation checkpoint bytes
match their completed block-20 training records. Metadata identifies clean
execution commit `1ce18c8736b6a50773b60b95ec0f11dd4c901028`, Python 3.13.15 and
a common dependency/source environment across cells. The retained metadata lists
exact dependency versions. The source PC used a different development environment;
these are laptop results, not pooled observations from both hosts.

Laptop migration reran all evaluation games to keep comparisons, repeat checks
and timing on one host. The old partial evaluation remains historical only.
The original authorization time and cumulative resource usage are preserved.
Final evaluation ledger: 14,530.25 CPU seconds including earlier attempts,
21,896.25 wall seconds since original authorization, 291,573,760-byte peak
campaign RAM, no active workers and no limit breach. Wall time includes downtime;
it is not a pure laptop throughput measurement. Training interruptions remain
in the retained final-training status records and earlier interruption documents.

The verifier checks registered YAML conditions/seeds, exact resolved jobs,
metadata, single-worker immutable evaluation, training artifact linkage, repeats,
resources and migration hashes before statistics. It deliberately rejects the
original full-matrix completion claim. Checkpoint verification can additionally
be rerun against the retrievable asset below.

## Reproduce and retrieve evidence

Compact lossless observations and metadata, training provenance and verification
records are committed in `laptop-results/evidence.json.gz`. Both primary and
repeat rows are also in `episodes.csv`; `result.json` contains uncertainty,
eligibility, every gate and exploratory D results. The original 1.29 GiB input ZIP
is unnecessary for recomputing these claims: redundant agent snapshots and raw
logs were omitted from the compact evidence. Preserve local originals until
peer review is complete.

```bash
gh release download issue124-reduced-evidence-v1 --repo 1BlauNitrox/mle-final-project --pattern issue124-final-checkpoints-v1.zip --dir training_outputs/issue124-evidence
python -m training.analyze_issue124_reduced --evidence experiments/2026-09-10-task2-rehearsal-mask/laptop-results/evidence.json.gz --checkpoints training_outputs/issue124-evidence/issue124-final-checkpoints-v1.zip --output training_outputs/issue124-recomputed
```

[Checkpoint evidence release](https://github.com/1BlauNitrox/mle-final-project/releases/tag/issue124-reduced-evidence-v1)
contains 19 trained checkpoints, two reference checkpoints, a per-file manifest
and training provenance. ZIP size: 3,852,548 bytes. SHA-256:
`dbdf7d81fb61bef9de6882c4fcfa59610c0ed9957da1aa345210599b9b327e59`.
The analysis command hashes all 21 extracted checkpoint byte streams and checks
them against retained evaluation/training hashes before recomputing conclusions.
The release is evidence, not an approved frozen-model release.

## Next decisions

Do not spend another day filling D/r5 merely to finish this matrix. Preserve the
negative rehearsal result, leave #124 open for the original unmet scope, and use
this PR as a completed reduced-result record. Await the independent #91 gamma
result before deciding whether another DQN training run is warranted. Prepare
Task 3 against an explicitly provisional predecessor while retaining Task 1/2
regression checks. The main unresolved capability is navigation retention and
productive coin collection, not simply avoiding self-kills. C is a useful
exploratory diagnostic candidate, not a retrospectively selected winner.

AI assistance: Codex implemented the reduced verifier and documented results.
Claims are derived from retained observations; tests and peer review are required.

## Later cross-experiment seed overlap

Current main's subsequently added #139 tabular loot-crate development seeds
144001-144005 overlap #124 training seed values. Those plans were absent from
#124's pinned execution revision. The historical seed inventory is retained with
source hashes and a pinned inventory digest; historical validation checks that
inventory. The default live launch audit remains strict and rejects the overlap.
This does not change #124 evaluation seeds or its within-study paired evidence.
Do not claim all present-day cross-model development populations are globally
independent, and do not repurpose these values as unseen final evaluation seeds.
