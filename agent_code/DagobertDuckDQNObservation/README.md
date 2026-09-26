# DQN observation-screen runtime

For issue #211, we preserve the trained 39-input DQN prefix and test either 12 descriptive hunting-geometry inputs or five bounded revisit-memory inputs.

All arms use a 56-64-64-6 network. Inactive input blocks and newly introduced weights start at zero. We retain the inherited reward mapping, one-step update, discount 0.9, batch size 64, replay capacity 10,000, and target update interval 500. The registered learning rate is 0.0002. One randomly selected episode per five-episode block explores uniformly over framework-legal actions. All other episodes use the learned greedy policy. Evaluation is greedy.

We initialize all arms from the same pinned control-r1 milestone-008000 fallback. Its SHA-256 is
`a5dadff5cc9f04a614caa828d3a0f014cad56f00c3644bb791275da85b35a8a6`.

We retain the online and target weights while initializing optimizer, replay, and paired random streams afresh. The checkpoint's `observation_mode` selects control, geometry, or memory. Feature schema 211 rejects incompatible checkpoints. We do not commit a default trained checkpoint in this directory.

We register training opponents, seeds, hardware budgets, evaluation suites, and decision rules in
`experiments/2026-09-19-observation-screen/config.json`. The runner copies this directory into an isolated pinned framework. Runtime imports stay within the agent directory. NumPy and CPU PyTorch are declared in `requirements.txt`.

The geometry describes current obstacles and possible escape routes, not future opponent decisions. Memory records the previous 16 observed positions and resets each round. Neither feature block overrides the learned action.

## Experimental result

All three registered replicas completed 1,000 training episodes for each of the control, geometry, and memory arms. We evaluated 360 paired Classic games per artifact against three rule-based opponents. We also ran the mixed-opponent, peaceful-survival, collection, bombing, looping, and serial-latency suites.

The unchanged frozen reference averaged 3.2167 score, 0.0833 kills, 39.17% survival, and 0.5250 self-kills per Classic game. All three trained continuations deteriorated sharply:

| Artifact             | Score/game | Kills/game | Survival | Self-kills/game |
| -------------------- | ---------: | ---------: | -------: | --------------: |
| Frozen reference     |     3.2167 |     0.0833 |   39.17% |          0.5250 |
| Continuation control |     0.1139 |          0 |    0.28% |          0.9722 |
| Hunting geometry     |     0.0778 |     0.0028 |    0.28% |          0.9917 |
| Revisit memory       |     0.0583 |          0 |    0.56% |          0.9917 |

Compared with the simultaneously trained control, geometry changed kills by +0.0028 per game, with a registered 95% paired hierarchical bootstrap interval of [0, +0.0167]. Its score difference was -0.0361 [-0.1667, +0.1056]. Memory changed kills by 0.0000 [0, 0] and score by -0.0556 [-0.1917, +0.0528].

Geometry did not reach the registered +0.05 kills-per-game screening target. Memory did not demonstrate the required 25% reduction in eligible late-game loop windows because the control and memory agents usually died before reaching the measured phase. Both variants also failed the registered safety and retention comparisons with the frozen reference.

Serial latency passed on every replica. Maximum decision times were 3.734 ms for geometry and 3.062 ms for memory.

Our analyzer classifies both feature variants as `reject_for_this_deadline`. We do not select or package any checkpoint from this experimental agent. The frozen reference remains our submission fallback. We record the complete intervals, secondary metrics, evidence hashes, and reproduction instructions in
`experiments/2026-09-19-observation-screen/README.md`.

## Limitations and follow-up

These results do not establish that hunting geometry or revisit memory are generally ineffective. The unchanged continuation collapsed alongside both feature variants, so we did not test the observation hypotheses with a stable continuation baseline.

Our audits indicate that degradation began after dense updates resumed with empty replay and fresh Adam state. We treat this as a diagnostic inference rather than a controlled causal conclusion. Issue #213 separately tests whether preserving replay and optimizer state stabilizes continuation.

We keep this directory as an experimental runtime for reproducing issue #211. It has no default trained checkpoint and is not a submission candidate.

## Reference

Hausknecht and Stone, *Deep Recurrent Q-Learning for Partially Observable MDPs*, https://arxiv.org/abs/1507.06527. We used this work as motivation for testing temporal information. This agent uses explicit visit counts and does not implement a recurrent network.
