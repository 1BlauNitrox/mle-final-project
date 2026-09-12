# Issue 147: legal masking did not establish successful hunting

The registered decision is **exploratory_mixed_or_negative**. No arm or replica
is selected. Masking passed the invalid-action and self-kill guards but failed
the hunting benefit and cumulative collection/crate-retention requirements.
Do not launch #137 / PR #140 from these checkpoints or pick the best-looking
replica. Task 2 remains cumulatively incomplete.

## Execution and process

- Executed source: `15bedeed078d073e4acefea378e1a06cd41ca7df`.
- Original #91 A/r3 parent: SHA-256
  `c0b6081acf5a733258005d46b9430f3ef50e403470a3bb5428661729f9aeedf7`,
  2,382,903 bytes; producing source
  `cbd52be8392f5003a91c5600fda4efd544b48ec5`.
- Both arms freshly migrated that parent, independently preserving online and
  target networks and zeroing the opponent columns. The initialization payloads
  differ only in `action_masking`. Final artifacts retain 10,000 completed
  episodes, gamma 0.9, active escape features and their registered masking mode.
  [checkpoints.json](checkpoints.json) inventories all ten finals; none is promoted.
- Unchanged registration: [config.yaml](config.yaml), SHA-256
  `95cbff14a8613f37e57b83fa2426e501499c33d5a84db3c8529661d70e4c6e4b`.
  Its prospective authorization text is historical. The runtime authorization
  and [owner's explicit pre-review execution exception](https://github.com/1BlauNitrox/mle-final-project/issues/147#issuecomment-5641563536)
  document permission to run, not permission to merge or evidence of peer approval.
  The runtime field `reviewed_commit` is a legacy name for the executed SHA.
  The requested exception marker was absent from hardware_description; the
  original record is preserved, with the exception disclosed here instead.
- Server: Linux workbench, Intel i5-6500T, four exposed CPUs, Python 3.13.15,
  CPU PyTorch 2.13.0+cpu, NumPy 2.5.3. Full hardware/dependency records are in
  `campaign/authorization.json`. Both arms and reference used this environment;
  the recorded matrix limits training to two workers and evaluation to one.
  Process records do not prove absence of unrelated host workloads.
- UTC interval: 2026-09-11 23:13:22 to 2026-09-12 05:49:36
  (Berlin: September 12, 01:13 to 07:49). Wall 23,773.72 s (6h36m),
  CPU 31,945.50 s (8h52m), peak monitored process memory 847,527,936 bytes
  (0.79 GiB). Caps were 10 wall-hours, 24 CPU-hours and 8 GiB; no breach.
- Complete: reference 320 jobs, control 1,605, masked 1,605; 3,530 attempts,
  all completed, zero failures or retries. Exactly 100,000 training episodes
  and 3,520 evaluation episodes, including 1,760 primary/repeat pairs.
  All registered seeds and immutable model bindings matched; reserved seeds
  were unused. Repository seed collision validation passed against executed source.
- The parent is explicitly provisional and owner-authorized. PR #136 remains
  changes-requested over its separate evidence packaging (missing bound YAMLs).
  That review gap does not change the verified parent bytes included here, but
  prevents treating the upstream result as fully accepted.

## Hunting and the treatment decision

Primary evaluation: 40 common development seed pairs per replica, five replicas
per trained arm. Reference is one fixed checkpoint evaluated on those same 40
pairs, not five independent reference models. Exact repeats are integrity checks,
not additional independent samples. Intervals use the unchanged 10,000-draw
crossed bootstrap, seed 147; treatment comparisons preserve matched replicas.

| Peaceful metric | Frozen parent | Control | Masked |
| --- | ---: | ---: | ---: |
| Elimination | 20.0% | 18.0% | 21.0% |
| Strict first place | 85.0% | 84.5% | 85.5% |
| Mean score | 4.075 | 3.800 | 4.470 |
| Mean score margin | 4.075 | 3.785 | 4.465 |
| Self-kill | 35.0% | 10.5% | 3.5% |
| Invalid actions / attempted actions | 5.915% | 8.838% | 0.00382% |

Masked-minus-control elimination is **+3.0 percentage points**, 95% CI
**[-14.5, +20.5]**. It fails both the >=10-point effect and positive lower-bound
requirements. Against the frozen parent, masked elimination improves by only
1 point (CI [-18.5, +21.5]), below the required >=20 points, and its absolute
21% is far below 60%. High first-place rates against peaceful_agent do not
establish hunting: elimination is a separate required endpoint.

Masked replica elimination rates were 45%, 35%, 10%, 2.5%, 12.5%; control rates
were 15%, 27.5%, 12.5%, 25%, 10%. This variation is retained, not filtered.
The strongest individual replica cannot replace the failed aggregate selection rule.

## Every registered gate

| Gate | Control | Masked |
| --- | --- | --- |
| Elimination >=60% | FAIL | FAIL |
| Elimination improvement >=20 points and positive CI lower bound | FAIL | FAIL |
| Strict first place >=60% | PASS | PASS |
| Strict first-place improvement CI lower >0 | FAIL | FAIL |
| Score margin >0 | PASS | PASS |
| Score-margin improvement CI lower >0 | FAIL | FAIL |
| Peaceful self-kill <=10% | FAIL | PASS |
| Peaceful invalid actions <1%, aggregate and every replica | FAIL | PASS |
| Classic collection retention | FAIL | FAIL |
| Classic survival retention | PASS | PASS |
| Classic self-kill retention | PASS | PASS |
| Classic crate retention | FAIL | FAIL |
| Classic invalid actions <1% | FAIL | PASS |
| Coin-heaven collection retention | FAIL | FAIL |
| Coin-heaven survival retention | PASS | PASS |
| Coin-heaven self-kill retention | PASS | PASS |
| Coin-heaven invalid actions <1% | FAIL | PASS |
| Loot-crate collection retention | FAIL | FAIL |
| Loot-crate survival retention | FAIL | PASS |
| Loot-crate self-kill retention | FAIL | PASS |
| Loot-crate crate retention | FAIL | FAIL |
| Loot-crate invalid actions <1% | FAIL | PASS |
| Every primary/repeat episode p95 <50 ms and max <100 ms | PASS | PASS |
| Masked-control elimination treatment benefit | N/A | FAIL |
| Source, parent, checkpoint, seed and raw/CSV integrity | PASS | PASS |
| Exact repeat identity, including executed action sequences | PASS | PASS |
| Completion and registered resource ceilings | PASS | PASS |

Masked collection-difference CI lower bounds are -18.22 points (classic,
must exceed -3), -15.40 (coin heaven, must exceed -2) and -8.59 (loot crate,
must exceed -3). Classic collection means were 23.0% masked versus 31.11%
parent. Crate contrasts use candidate minus 0.9 times reference; lower bounds
are -18.94 classic and -7.81 loot crate, failing the >=0 rule. These failures
mean retention was not established; they do not prove that every metric worsened.

Across all 3,520 primary/repeat episodes, the worst episode p95 was 13.630 ms
and maximum decision 21.138 ms. This passes the registered server latency gate;
it is not an official-hardware compatibility certification.

[decision.json](decision.json) retains all arm summaries, contrasts and gates;
[summary.csv](summary.csv) contains aggregate and per-replica primary metrics;
[gates.csv](gates.csv) is the machine-readable decision table. Per-seed raw and
compact observations are in the retrievable archive below.

## Evidence and reproduction

Release: [issue147-evidence-v1](https://github.com/1BlauNitrox/mle-final-project/releases/tag/issue147-evidence-v1).
The original downloaded archive is preserved without repacking:

- `issue147-evidence.tar.gz`: 125,853,763 bytes; SHA-256
  `328b0cf69e99caedd1f60d33eb529f9271c270f8db7ffde571d601f6b30c828a`.
- `issue147-evidence.tar.gz.manifest.json`: 2,490,045 bytes; SHA-256
  `a60fe23c5f1bd1e620f1e934ce628cba11710b3f7dbbdfdae08776cab29ccdf3`.
- All 10,635 regular members match the manifest. It maps member paths to
  SHA-256 and byte size, and records the outer archive hash/size.
- `campaign/`: original authorization/resources, resolved plans, complete job
  status/metadata, raw framework JSON including decision timings, episode CSVs,
  and trained/frozen checkpoints. `binding/`: original parent, verified fresh
  migration, masked initialization and bound plans. `analysis/`: original full
  result, source-manifest and gzip JSON with evaluation/training observations.
  Duplicate staged source and verbose logs are intentionally omitted; executed
  source is retrievable at the exact Git commit above.

From this result PR's checkout with documented Python dependencies installed,
run each line as one command. Use new output directories; do not overwrite evidence.

```bash
gh release download issue147-evidence-v1 --repo 1BlauNitrox/mle-final-project --dir inputs-147
echo '328b0cf69e99caedd1f60d33eb529f9271c270f8db7ffde571d601f6b30c828a  inputs-147/issue147-evidence.tar.gz' | sha256sum -c -
echo 'a60fe23c5f1bd1e620f1e934ce628cba11710b3f7dbbdfdae08776cab29ccdf3  inputs-147/issue147-evidence.tar.gz.manifest.json' | sha256sum -c -
python -m training.verify_task3_mask_results --archive inputs-147/issue147-evidence.tar.gz --manifest inputs-147/issue147-evidence.tar.gz.manifest.json --extract-to training_outputs/issue147-import --output training_outputs/issue147-recomputed
python -m training.task3_mask_campaign verify --analysis-dir training_outputs/issue147-import/analysis
```

The historical verifier checks Linux Git-blob fingerprints and recorded server
dependencies separately from the current analysis installation. It relocates
parent paths only in temporary copies, validates the original binding hash and
every job's original configuration fingerprint, checks all final checkpoint
modes/counts and reruns the registered raw analyzer. No games are launched.
The Windows reproduction matched all 100,000 training and 3,520 evaluation
observations, every decision field and the full source-manifest exactly.
Gzip container timestamps may differ; decompressed observations are identical.

Local review checks: Python 3.13 `python -m pytest -q` passed 872 tests and
11 subtests; Ruff on tests/training/scripts/affected agent, `compileall` and
the one-round random-agent framework smoke passed. Ten focused verifier tests
cover relocation preservation, tampering, archive/member integrity and refusal
to overwrite imports. These are implementation checks, not additional training.
Current remote checks and human review are recorded on PR #148.

## Next decision

The tested mask addresses observed invalid actions, but this experiment does not
establish a hunting improvement. The smallest justified next step is bounded
diagnosis of the existing training summaries and #146 looping report, followed
by one prospective intervention only if a concrete mechanism is supported.
The aggregate action counts alone cannot identify loops or their cause; do not
add an untested anti-loop rule or assume more episodes will solve the problem.

No further training is authorized by this result. #137 remains blocked; #126
Task 4 protocol/analyzer preparation can continue independently, but dependent
training needs a registered available parent. A distinct exploratory continuation
requires an explicit owner decision and new prospective registration, retaining
these failed gates. Do not wait for Task 2 to become retrospectively complete.

AI assistance: Codex verified archive/raw data, implemented and tested the
portable verifier, reproduced the registered analysis and drafted this record.
Human review of the evidence and interpretation remains required before merge.
