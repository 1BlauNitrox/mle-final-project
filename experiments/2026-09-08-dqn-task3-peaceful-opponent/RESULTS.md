# Completed exploratory peaceful-stage campaign

The registered campaign completed on 2026-09-11. Its decision is
`exploratory_mixed_or_negative`, with **no selected replica** and
`stop_no_automatic_continuation`. Task 2 is not complete. All five training
replicas and all evaluation pairs are retained; no failed scientific seed was
discarded. PR #120 records this result; it does not close #109 or authorize #137.
The immutable `config.yaml` retains its pre-execution status fields; completed
status and the separate execution authorization are recorded here and in the
retained campaign evidence, without rewriting the registered configuration.

## Execution and provenance

Julius authorized and ran reviewed source
`0c9b0c1add52e53269646a1b2ef2aec96fb297dd`, with the previously approved
[protocol](config.yaml): five 10,000-episode training runs, four suites with
40 common development seed pairs, and exact repeats. There are 960 primary
evaluation episodes and 960 repeats. Repeats verify determinism and are not
additional independent observations. Confirmation and final held-out seeds
were not used.

The parent was prospectively changed from historical #107 to #91 A/r3,
produced by `cbd52be8392f5003a91c5600fda4efd544b48ec5`, SHA-256
`c0b6081acf5a733258005d46b9430f3ef50e403470a3bb5428661729f9aeedf7`
(2,382,903 bytes). Julius explicitly permitted use before PR #136 result peer
review. This is a provenance exception, not a declaration of Task 2 success.
The server's migrated starting artifact is 64,237 bytes, SHA-256
`55c060bdaebbf7c0ad275718e2d1e5d0f82a90f030b30814a1338715f7ae5621`.
The retained binding validates both networks, preserved input columns, zero
opponent columns, fresh state and inherited configuration (gamma 0.9, escape
features enabled, action masking disabled).

Server: Intel Core i5-6500T, four cores, approximately 14 GiB RAM, Python
3.13.15 and CPU PyTorch 2.13.0+cpu. Exact dependency records are retained in
authorization and every run's metadata. Wall time was 11,956.24 seconds
(3 h 19 min); measured CPU time 15,765.85 seconds; peak process-tree memory
843,542,528 bytes. No resource ceiling was reached.

## Results and uncertainty

Candidate means pool five equally sized replicas; reference means use one
fixed parent and the same 40 pairs. Intervals use the registered 10,000 crossed
bootstrap draws (seed 109), resampling model and common pair indices. They
condition on this parent and development population. All gates are conjunctive.

| Peaceful metric | Candidate | Parent | Registered decision |
| --- | ---: | ---: | --- |
| Episodes with an elimination | 20.5% | 17.5% | Fail: required >=60% |
| Elimination improvement | +3.0 pp | — | Fail: required >=20 pp and positive lower bound; 95% CI [-12.5, 18.5] pp |
| Strict first place | 89.5% | 85.0% | Absolute pass; improvement fails, CI [-8.0, 17.5] pp |
| Mean score | 4.430 | 3.725 | Descriptive |
| Score margin over opponent | 4.415 | 3.700 | Absolute pass; improvement fails, CI [-0.665, 2.170] |
| Self-kill episodes | 11.5% | 40.0% | Fail: required <=10% |
| Invalid-action ratio | 6.79% | 3.25% | Fail: every candidate replica and aggregate must be <1% |

Replica elimination rates were 15%, 20%, 15%, 32.5%, and 20%. Even the
highest observed replica fails the absolute hunting gate; selecting it after
seeing results would also violate the registered selection rule. A high
first-place rate against this opponent does not establish reliable elimination.

| Retention suite | Candidate / parent collection fraction | Collection difference, 95% CI | Other failed gates |
| --- | --- | --- | --- |
| Classic | 0.2556 / 0.3917 | -0.1361 [-0.2472, -0.0333] | Survival noninferiority, self-kill noninferiority, crates, invalid actions |
| Coin heaven | 0.4372 / 0.3660 | +0.0712 [-0.1117, 0.2942] | Invalid actions |
| Loot crate | 0.2141 / 0.1750 | +0.0391 [-0.0498, 0.1223] | Crates, invalid actions |

All three collection-retention gates fail. Positive point estimates in coin
heaven and loot crate do not prove noninferiority. Classic collection shows
a negative interval. Classic crate retention contrast (candidate minus 0.9
times parent) is -11.10 [-24.80, 1.16]; loot crate is +4.52 [-6.26, 14.65].
Invalid-action ratios are 10.15%, 26.84%, and 15.92% respectively. Survival and
self-kill retention pass in coin heaven and loot crate. Runtime passes every
primary and repeat episode's registered latency checks, resource checks,
artifact immutability and exact repeated actions/outcomes.

[results.json](results.json) contains every gate and interval;
[summary.csv](summary.csv) includes per-replica metrics, action-independent
outcomes and latency; [contrasts.csv](contrasts.csv) contains all contrasts.
The external analysis retains action counts, per-call times and training
diagnostics. Neither aggregate results nor the apparent best seed replace it.

## Evidence and exact reproduction

[evidence.json](evidence.json) records the durable release URL, SHA-256, byte
size and original transfer hash. The release contains all retained raw episode
statistics and normalized CSVs, every job's metadata, status/resolved plans,
authorization/resources/hardware, original parent binding and five final
training/evaluation artifacts. Redundant staged source copies and text logs
are omitted. `evidence-manifest.json` identifies every retained source file by
relative path, byte size and SHA-256. `analysis/` contains lossless compressed
observations, the full result and input manifest.

Run from this PR's checkout after fetching its Git history (including executed
commit 0c9b0c1). These commands perform analysis only:

```bash
mkdir -p training_outputs/issue109-review
python -c 'import json,urllib.request; r=json.load(open("experiments/2026-09-08-dqn-task3-peaceful-opponent/evidence.json")); urllib.request.urlretrieve(r["url"], "training_outputs/issue109-review/evidence.tar.gz")'
python -c 'import hashlib,json,pathlib; r=json.load(open("experiments/2026-09-08-dqn-task3-peaceful-opponent/evidence.json")); p=pathlib.Path("training_outputs/issue109-review/evidence.tar.gz"); assert p.stat().st_size==r["size_bytes"] and hashlib.sha256(p.read_bytes()).hexdigest()==r["sha256"]'
tar -xzf training_outputs/issue109-review/evidence.tar.gz -C training_outputs/issue109-review
python -m training.analyze_task3_campaign --verify-evidence training_outputs/issue109-review/analysis
python -m training.verify_task3_results --campaign-root training_outputs/issue109-review/evidence/runs-parent91 --binding-dir training_outputs/issue109-review/evidence/binding-parent91 --output training_outputs/issue109-review/recomputed
```

The historical verifier checks Git source bytes with Linux path ordering and
the recorded execution dependencies independently of the local analysis
environment. It relocates parent paths in a disposable binding copy, preserving
the original evidence. Full verification passed before publication.

One analyzer defect was fixed during analysis: framework round keys include
timestamps and are serialized lexically, so positional lookup mismatched
training round latencies after round 99. Numeric round identity now selects
the corresponding raw timings, with regression tests for lexical order,
timestamped keys and duplicate identities. No observations, gates, bootstrap
parameters or scientific policy were changed.

## Validation

Python 3.13: 857 tests and 11 subtests pass, including the new verifier and
round-identity regression tests. Scoped Ruff and Python compilation pass.
Both complete raw-evidence verification and independent compact-observation
recomputation produce the same negative decision. No scientific games were
launched during analysis. Current PR checks and fresh peer review remain the
GitHub acceptance record.

## Next decision

Do not launch the registered coin-collector continuation or export a selected
Task 3 model. Review this negative result first. Useful work can continue
without waiting for Task 2: diagnose invalid actions and poor attack conversion
on development evidence, then register a focused intervention with the same
retention safeguards before another run. The current study changes opponent
features, reward and training together; it cannot identify their separate
causal effects. More episodes, a different parent or masking would each require
a new explicit controlled protocol, budget and authorization. Do not tune on
reserved confirmation/final seeds. Task 2 validation and peer review remain
separate requirements; they are not the only blocker to continuation.
