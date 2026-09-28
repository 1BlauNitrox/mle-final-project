# DQN hunting geometry and revisit observations

Issue #211 compares three fixed-budget continuations of the same trained fallback: unchanged observations, an additional hunting-geometry block, and a separate revisit-memory block. We wanted to test whether obstacle-aware hunting information improves kill conversion and whether short movement history reduces late-game loops without damaging the existing policy.

## Experimental design

We keep the original 39 inputs unchanged. All three networks use 56 inputs, with inactive feature blocks set to zero. We preserve the online and target weights, initialize the newly added input columns to zero, and reset replay and optimizer state equally in all arms.

The geometry block adds five obstacle-aware distances to reachable attack positions, corresponding reachability flags, and two bounded estimates of an opponent's escape options. The memory block adds visit counts for the current tile and four neighbouring tiles over the previous 16 decisions. Rewards and one-step DQN updates stay unchanged.

We trained three paired replicas, each running control, geometry, and memory for 1,000 episodes. Laptop 1 ran replica 1, laptop 2 ran replica 2, and the PC ran replica 3. Within each replica, all arms used the same training-world stream. Independent evaluation worlds covered Classic against three rule-based opponents, mixed opponents, peaceful hunting, visible-coin collection, and solo bombing. We rotated competitive starting slots. The executable protocol, exact seeds, resource limits, and prospective gates are stored in `config.json`. The seed collision audit is stored in `seed-audit.json`.

For the exploratory screen, we required either at least +0.05 kills per game for geometry or a 25% relative reduction in eligible loop windows for memory. Point estimates for score, self-kills, survival, collection retention, and latency still had to pass their registered limits. We treated wide score confidence intervals as warnings during screening rather than automatic vetoes. Promotion required a separate confirmation experiment.

## Evidence validation

All nine training jobs completed their 1,000-episode budgets. Every replica also completed all 20 registered fixed-final evaluation cells and four serial latency cells. Each exported archive contains 67 manifested files. File size and SHA-256 verification succeeded for every entry:

* replica 1: 6,789,999 bytes,
  `ab1c184b2ec6651147140aeef5f543d1baedf3591e8a6ca726f924fda9dbc69f`
* replica 2: 7,210,855 bytes,
  `7b56d6209adbf58c86fbcd61f622e94bebf2c01eb2a62b8066324a46f63533f7`
* replica 3: 7,583,559 bytes,
  `c581b3f6ac1224dbafa118d58b87f49ff8c93d8a4962f491d695c672a1b02f63`

All replicas record configuration SHA-256
`b1881b18e5eb44ffa31c456f0a59efe9d68e046381a90d7c4ae109530676a216`
and launch commit `1959e7ebaa9cc188dcf28dfbbfa8a2a167482f71`.

Our initial combined analysis stopped because six source hashes differed across Windows and Linux checkouts. We compared the preserved files directly. The differences came only from Git line-ending conversions. Normalizing the CRLF copies to LF produced the exact recorded hashes, and `git diff --ignore-space-at-eol` showed no content change. Analyzer correction `5657e65` accepts only these six verified LF/CRLF hash pairs. Fifteen focused tests pass. This correction changes evidence validation only. No model, episode, metric, gate, or scientific result changes.

The combined `analysis.json` has SHA-256
`bd78143ae7ed71c7d80071a3b4c5e774863914698f53d80ca92e171aad1dc740`.

## Results

The primary Classic results contain 360 games per artifact: 120 shared worlds from each of the three replicas, against three rule-based opponents.

| Artifact                       | Score/game | Kills/game | Survival | Self-kills/game | Coins/game |
| ------------------------------ | ---------: | ---------: | -------: | --------------: | ---------: |
| Frozen reference               |     3.2167 |     0.0833 |   39.17% |          0.5250 |     2.8000 |
| Unchanged continuation control |     0.1139 |          0 |    0.28% |          0.9722 |     0.1139 |
| Hunting geometry               |     0.0778 |     0.0028 |    0.28% |          0.9917 |     0.0639 |
| Revisit memory                 |     0.0583 |          0 |    0.56% |          0.9917 |     0.0583 |

The repeated frozen-reference rows use the same worlds and behaviour and serve as a consistency check. They are not 360 independent reference worlds.

Compared with the unchanged continuation, geometry changed score by -0.0361 (95% paired hierarchical bootstrap interval [-0.1667, +0.1056]), kills by +0.0028 [0, +0.0167], survival by 0.0000 [-0.0111, +0.0139], and self-kills by +0.0194 [-0.0056, +0.0472]. Memory changed score by -0.0556 [-0.1917, +0.0528], kills by 0.0000 [0, 0], survival by +0.0028 [-0.0111, +0.0222], and self-kills by +0.0194 [-0.0083, +0.0472].

All trained continuations performed substantially worse than the frozen reference. Geometry lost 3.1389 score and 0.0806 kills per game relative to the reference. Survival fell by 38.89 percentage points and self-kills increased by 0.4667 per game. Memory lost 3.1583 score and 0.0833 kills per game. Survival fell by 38.61 percentage points and self-kills increased by 0.4667 per game. Complete intervals and secondary endpoints remain in `analysis.json`.

## Looping and runtime

Control and memory produced no eligible 24-step hazard-free late-game windows because the agents usually died before reaching this phase. We therefore have missing loop evidence, not evidence of no looping. Geometry produced 151 eligible overlapping windows, all classified as short-period cycles. The frozen reference produced 15,301 looping windows among 15,403 eligible overlapping windows. These overlapping counts are descriptive and are not statistically independent observations.

All feature arms passed the registered serial latency limits in every replica. Geometry's maximum decision time was 3.734 ms and memory's was 3.062 ms, well below the 500 ms tournament deadline. Runtime was therefore not the limiting factor.

## Decision and interpretation

Our registered analyzer classifies both geometry and memory as `reject_for_this_deadline`. Geometry did not reach the required +0.05 kills per game. Memory did not demonstrate a 25% loop reduction because neither memory nor control produced eligible late-game windows. Neither arm passed the safety and retention comparison with the frozen reference. No arm is eligible for confirmation, and we selected no new submission artifact.

These results do not establish that the additional observations are generally unhelpful. The unchanged continuation collapsed alongside both feature arms, while the unchanged frozen reference remained effective on the same evaluation worlds. We therefore did not test the feature hypotheses with a stable continuation baseline. Read-only audits of replicas 1 and 2 indicate that degradation began when dense gradient updates started with fresh replay and optimizer state. We treat this as a supported diagnostic inference, not a separately controlled causal conclusion.

We will not promote or extend the geometry and memory checkpoints before the deadline. The frozen reference remains our fallback. Issue #213 separately tests whether preserving mature replay and Adam optimizer state stabilizes a short continuation. We do not reuse or overwrite the failed #211 checkpoints.

## Reproduction and retained evidence

From a checkout containing analyzer correction `5657e65`, we extract the three immutable archives into separate directories and run:

```powershell
.venv\Scripts\python.exe .worktrees\issue211-screen\scripts\analyze_observation_screen.py `
  --roots training_outputs\issue211-combined\r1 `
          training_outputs\issue211-combined\r2 `
          training_outputs\issue211-combined\r3 `
  --output training_outputs\issue211-combined\analysis-recomputed.json
```

We expect `reject_for_this_deadline` for both geometry and memory. We compare the recomputed file with the retained analysis rather than assuming byte-identical output across analyzer revisions.

Replica 1 is published as GitHub release `issue211-r1-evidence-v1`. The exact verified archives for replicas 2 and 3 remain local and still need a durable published locator before this PR is review-ready. The external opponent bundle is not part of the evidence archives.

## References

Hausknecht and Stone, *Deep Recurrent Q-Learning for Partially Observable MDPs*, https://arxiv.org/abs/1507.06527. We use this paper as motivation for temporal information. The experiment uses explicit visit counts and does not implement DRQN.
