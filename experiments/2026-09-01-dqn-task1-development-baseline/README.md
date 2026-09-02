# Task 1 DagobertDuckDQN development baseline

## Status and lineage

Issue #41 is complete with a mixed, overall negative decision. Training used
commit `e7e8b52f50b2acb46ccad04905d76c6948304e21` and agent fingerprint
`56938f004403b056aef6df07079e0f6d94f0e0c7093ae693a3cad5c131e2da4e`.
Evaluation used commits `573443a` and `255f627`; the latter only adds retry
handling for transient Windows manifest locks.

Five fresh DQN runs used world seeds `12001`--`12005`, agent seeds
`22001`--`22005`, 10,000 `coin-heaven` episodes each, no opponents, and final
checkpoint selection. Evaluation used only seeds `31001`--`31040`. Reserved
seeds `31041`--`31050` and final held-out seeds remain unused.

## Training and retained failure

Runs 1--4 completed normally. Run 5 first failed after 616 episodes when
Windows denied an atomic checkpoint replacement. Its raw output and partial
checkpoint (`d778e033896d02c11f4e7e25dfc936fd45bcd0e6df010a9e9bb7b1f6a735057a`)
are retained. A prospective issue amendment authorized one fresh retry and a
60,000-episode attempted budget. The unchanged retry completed. Successful
training took 37,482 seconds (10.41 hours); 50,616 episodes were attempted.

| Run | World / agent seed | Duration [s] | Final artifact SHA-256 |
| ---: | --- | ---: | --- |
| 1 | 12001 / 22001 | 7092.31 | `144540fd2d99067bb010c25583a004f1bba559023becf26d38c92095e2df5cd0` |
| 2 | 12002 / 22002 | 7607.56 | `89859bd01bcb4e8fe87b614861c696cfbf1260c0cd07185adb781f48ba7c316e` |
| 3 | 12003 / 22003 | 7843.45 | `9f2beac25df249dc2650f180045e0e826626d5eee3a2dd6d96c6ac3b74c8ff84` |
| 4 | 12004 / 22004 | 7520.98 | `f1a09fbea1587e55e22375608467959f6261f63252212cb765e01a551de72d55` |
| 5 | 12005 / 22005 | 7418.15 | `1dc1fd50b2477896a3d80dd04e8e803a895777ce74494d0bc9210b4c0de7a862` |

Each final artifact is 862,778 bytes. Raw logs, checkpoints, per-episode rows,
and the manifest remain under the ignored series
`training_outputs/issue-41-dqn-task1-baseline/20260831T235351723566Z/`.

## Development results

All 200 primary episodes and 200 exact repeats completed. Deterministic outcome
fields matched and every artifact remained byte-identical. Every `act()` time
is retained in raw statistics. Observed process memory was about 215 MB for the
agent plus 67 MB for the orchestrator, below 8 GB.

| Model | Mean fraction | SD | Full clears | Invalid rate | WAIT | BOMB | p95 / max [ms] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| run-01 | 1.0000 | 0.0000 | 40 | 0.000000 | 10 | 0 | 0.502 / 2.115 |
| run-02 | 0.9965 | 0.0221 | 39 | 0.001456 | 2 | 0 | 0.493 / 2.786 |
| run-03 | 0.4830 | 0.3152 | 1 | 0.000254 | 6487 | 0 | 0.752 / 17.426 |
| run-04 | 0.7395 | 0.3006 | 13 | 0.608521 | 6 | 0 | 0.393 / 1.655 |
| run-05 | 0.9480 | 0.1657 | 32 | 0.001087 | 0 | 0 | 0.746 / 27.601 |
| **Aggregate** | **0.8334** | **0.2875** | **125** | **0.164158** | **6505** | **0** | **0.752 / 27.601** |

`summary.csv` contains the compact values and `result.json` the mechanical
criterion decisions. The aggregate performance gate passed, but only three
models reached 0.75 and the aggregate and run-04 invalid-action gates failed.
No-bomb, immutability, and latency gates passed.

### Determinism is not yet verified for these artifacts

The recorded determinism result was produced by a check that compared episode
outcomes and per-action **totals** only. Two runs can execute the same moves in
a different order and still produce identical totals, so that check could not
establish the reproducibility the repository contract requires.

The check has since been strengthened to compare a digest of the ordered
executed action sequence, and the framework now records that digest. Verifying
these five artifacts under the stronger check requires re-running the 400
evaluation episodes against them, which has not been done. Until then the
determinism gate for this experiment is **unverified**, not passed. The
strengthened check applies to every subsequent experiment.

PR #37 contains aggregate tabular rows but not the 200 model-index/world-seed
pairs; its five original artifacts are unavailable locally and in GitHub
Actions. The paired bootstrap interval therefore cannot be computed without
inventing evidence. Descriptively, its reported aggregate is 0.8995 and DQN is
0.0661 lower; this is not a paired confidence interval.

## Decision

Do not freeze a DQN candidate. Preserve this mixed result and prospectively
register one controlled follow-up focused on invalid-action handling and
run-to-run instability, changing one factor only (for example legal-action
masking). Do not use reserved or final held-out seeds in that follow-up unless
specified prospectively.
