# Issue 225 campaign and submission comparison

## Scope

This record covers our loop-reduction and attack-improvement campaign from 20 September 2026 noon through the final selection benchmark on 21 September 2026. We preserved failed runs instead of treating visual play as evidence. `campaign-inventory.json` lists each completed issue-225 run and the SHA-256 of each registered experiment configuration.

Our initial protected candidate is the `broad_persistent` anti-loop policy from the V7 experiment lineage. Its runtime-bound reference checkpoint is `training_outputs/issue225-pc-v2/initial/reference.pt` with SHA-256 `85f2d9a436cccae99ceefb181d9e759c82c50718872b91723d21b923b3c87f6f`.

The protected ZIP remains `dest/final-project-agent-code.zip` with SHA-256 `b72722395ad597c8a36c1123f9ab9ecd3289362ad04e330c562edc91961adbd1`.

## Campaign chronology and findings

The broad anti-loop search consisted of V1, V2/V2b, V3, V4, V5/V5b, and V7.

V1, V2b, and V4 failed their registered screens. V3, V5b, and V7 were locally eligible, but only V7 was independently confirmed and retained as our protected fallback.

Its confirmation covered 160 multiplayer games plus 20 coin, 20 crate, and 10 latency games. Relative to its control, it had a score difference of +0.0125, unchanged kills, survival −0.00625, self-kills +0.00625, and invalid actions −0.00625 per multiplayer game.

The registered loop count was zero among 6,948 eligible candidate windows versus 1,255 among 7,664 control windows under the registered 24-step, hazard-free loop definition. This metric does not rule out shorter or visually obvious oscillations such as right-left alternation.

Coin retention was unchanged. Crate collection changed by +0.022. Latency was 17.97 ms p95 and 29.15 ms maximum.

The teacher-loop continuation, `issue225-pc-v2`, compared a teacher-loss arm with the frozen fallback after a 5,000-transition warm-up. It did not meet the registered final screen and therefore did not replace the fallback.

For the attack work, we changed one attack mechanism at a time. The safe attack, trapped-attack, bomb-head, learned attack-gate, endgame-pursuit, movement-only pursuit, bounded-pursuit, corridor-pursuit, capped corridor, and paired-confirmation variants all remained non-promoted.

Several showed a pilot signal but failed a safety, loop, or independent-confirmation criterion. We therefore did not allow any attack candidate to overwrite the protected ZIP.

V18 shortened the loop history early during pursuit. Its fresh confirmation showed a kill difference of +0.0179/game, below the +0.025 registered gate, and an increase of +0.243 loop windows/game. We rejected it.

V19 limited the shorter history to the period after a learned pursuit bomb and tested a post-kill cap. The selected `targeted` pilot passed, with score +0.1667 and kills +0.0333/game versus the protected fallback.

Its fresh confirmation failed despite a +0.0286 kill difference. Self-kills increased by +0.0536/game, above the +0.05 gate. Loop counts increased by +0.2643/game and loop-game incidence increased by +0.00357.

We include V19 in the final benchmark only as an explicitly ineligible comparison candidate.

## Final common-seed benchmark

`config.json` registers the same 52-game benchmark for each checkpoint:

* 16 Classic games against three `rule_based_agent` opponents
* 12 mixed-opponent games
* 12 coin-heaven games
* 12 loot-crate games

We run the agents in their recorded source trees because the fallback and warm-lineup checkpoints use different feature schemas.

Per-game rows are written to `training_outputs/issue225-submission-selection-benchmark/games.jsonl`. The aggregated output is `analysis.json`.

Candidates:

| Candidate              | Frozen runtime                                                  | Eligibility before this benchmark                 |
| ---------------------- | --------------------------------------------------------------- | ------------------------------------------------- |
| `fallback`             | `DagobertDuckDQNAntiLoop` from `issue225-pc-v2/source`          | Eligible and protected                            |
| `warm_lineup`          | `DagobertDuckDQNTask3` from local `C:/task4-warm-lineup/source` | Requires this comparison                          |
| `targeted_unqualified` | `DagobertDuckDQNAntiLoop` from V19 restart runtime              | Ineligible: failed fresh loop and self-kill gates |

The prospective screen keeps the fallback unless warm-lineup has non-negative paired score, kills, survival, and collection effects, no self-kill or invalid-action increase, and at least +0.05 coin-collection fraction on the Coin suite.

The failed V19 candidate does not become eligible through this benchmark.

## Final benchmark result and proposed decision

The benchmark completed 156/156 games without a runner error.

The compact per-game evidence is `per-game-results.jsonl` with SHA-256 `1500169342D305811B9FC0916D9E0432E7A18D8DD4C0F835219B50BA37352482`.

The aggregate is `benchmark-analysis.json` with SHA-256 `FECAF28350B2156A3239B8BD44CD8ACBDEFB37FC1C6031ACB10FF75F335F0782`.

| Comparison versus fallback               | Classic (16) | Mixed (12) | Coins (12) | Crates (12) |
| ---------------------------------------- | -----------: | ---------: | ---------: | ----------: |
| Warm score difference                    |       +0.625 |     +2.417 |      0.000 |       0.000 |
| Warm kills difference                    |       +0.125 |     +0.417 |      0.000 |       0.000 |
| Warm self-kill difference                |       +0.125 |     −0.250 |      0.000 |       0.000 |
| Warm survival difference                 |       −0.125 |     +0.250 |      0.000 |       0.000 |
| Warm invalid-action difference           |       −0.438 |     −1.333 |      0.000 |       0.000 |
| Warm coin-collection-fraction difference |        0.000 |     +0.037 |      0.000 |       0.000 |

Warm-lineup failed the prospective screen because its Coin-suite collection-fraction difference was 0.000, below the required +0.05.

It also lost 12.5 percentage points of Classic survival and added 0.125 self-kills per Classic game. Under this registered screen, it does not replace the protected fallback.

The targeted V19 candidate had larger Coin, +0.060 fraction, and Crate, +0.110 fraction, differences and positive score differences in all suites. It remains explicitly ineligible because its pre-registered confirmation failed the loop and self-kill gates.

Under the registered selection rule, we retain `dest/final-project-agent-code.zip`, the protected broad-persistent fallback.

It is the only candidate in this comparison with an independent result under the registered loop metric and without a failed prospective gate.

The benchmark does not establish that this candidate eliminates visually apparent loops or performs best on every opponent composition. It supports retaining the fallback under the registered selection rule.

## Reproduction

```powershell
cd .worktrees/issue225-teacher-loop
..\..\.venv\Scripts\python.exe scripts/benchmark_submission_candidates.py
```

The command resumes from existing JSONL rows.

We run the focused summary contract test with:

```powershell
..\..\.venv\Scripts\python.exe -m pytest tests/test_benchmark_submission_candidates.py -q
```
