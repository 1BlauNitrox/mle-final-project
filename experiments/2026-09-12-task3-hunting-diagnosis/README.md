# Issue 162 hunting diagnosis - DRAFT for human review

## Question and setup

After the negative standard-versus-Double-DQN comparison in #150, did the frozen
standard-DQN agents encounter safe attacks, choose bombs at those opportunities,
and receive the inherited crate-free bomb penalty? This was a bounded diagnosis,
not a training experiment or a test of a remedy.

All five #150 standard final replicas completed the registered matrix on
12 September 2026: two classic worlds against one peaceful_agent
(1501101/1501102), plus one solo-classic world (1501201). Agent seeds were world
seeds +1,000,000. The 15 episodes and 6,000 recorded steps are complete, with
no optimizer updates, checkpoint selection, policy override or held-out testing.

[config.json](config.json) pins the five original checkpoint hashes and the
unchanged agent/framework source `6f014485a3026cc3707fa2cc3a379880dd0b74bd`.
The executed diagnostic driver was `c7db7e081f0495b28777bbec0d4c5dac9ffe3576`;
its adapted instrumentation came from `e960bcc21dbe8b03343b85587a3af725c415616d`.

## Findings

| Diagnostic observation | Count |
| --- | ---: |
| Recorded states with a living opponent | 3,372 |
| Geometric attack states | 14 |
| Attack states with a predicted safe escape | 14 |
| Safe attack states with BOMB available and legal | 2 |
| Bombs placed at those available opportunities | 2 |
| Safe attack bombs receiving the crate-free penalty | 1 |
| Recorded opponent eliminations | 4 |
| Episodes containing a registered stall window | 8 / 15 |

The key example is r3, world 1501101, step 173: the agent placed a bomb
threatening the opponent, with a predicted escape and no crates in its footprint.
The reward instrumentation recorded WASTEFUL_BOMB_PLACED = -0.5. At step 177,
the native events recorded KILLED_OPPONENT (+5). Both full public-state excerpts
are in [attack-example.json](attack-example.json). Because training was disabled,
these are reconstructed reward components, not optimizer updates during diagnosis.

Opponent Manhattan distance had median 16 and range 1-28 over the recorded
opponent-present states. The registered stall predicate found 1,817 **overlapping**
24-step windows across eight episodes: five opponent episodes and three solo
ones. Each window had a fixed board/coin signature, no bombs/explosions or progress
events, and a position cycle of period 1-4. These windows are not independent
samples or 1,817 separate failures.

## Interpretation and follow-up

The penalty can apply to an actual successful, apparently escapable attack.
That supported a controlled reward ablation; it does not show that the penalty
caused poor hunting or loops. Two available opportunities across only two shared
opponent worlds cannot establish a general opportunity rate, efficacy estimate
or causal effect. No confidence interval or performance pass is claimed.

The resulting ablation was registered separately as
[#163](https://github.com/1BlauNitrox/mle-final-project/issues/163). Its completed
[PR #165](https://github.com/1BlauNitrox/mle-final-project/pull/165) found no observed
benefit and no qualifying reward activations in retained replay. Review that
result before proposing further training. No model/default is adopted here;
cumulative Task 2 and Task 3 requirements remain unmet.

## Evidence and limitations

[trajectory-evidence.zip](trajectory-evidence.zip) retains the exact original
compressed trajectories, native summaries, registration and source hash manifest
(32 members; 511,562 bytes). [evidence-manifest.json](evidence-manifest.json)
records its SHA-256. The verifier checks every trajectory hash, reproduces all
saved diagnostic metrics and stall windows, compares all 6,000 compact CSV rows,
and verifies both public-state excerpts without playing games:

```powershell
python -m scripts.verify_issue162_evidence --record experiments/2026-09-12-task3-hunting-diagnosis
```

Refs [#162](https://github.com/1BlauNitrox/mle-final-project/issues/162)
and [PR #164](https://github.com/1BlauNitrox/mle-final-project/pull/164).
AI assistance was used for instrumentation, verification and drafting.
