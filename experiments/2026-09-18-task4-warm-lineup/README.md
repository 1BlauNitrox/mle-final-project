# Warm-started continuation against the tournament line-up

## Question

The final-training agent collected more coins than `rule_based_agent`, but performed worse on eliminations. We therefore tested whether continuing from the fixed episode-8,000 fallback against three `rule_based_agent`s improves hunting without losing earlier capabilities.

The fallback is control-r1 at episode 8,000 from the final training run. We fixed it in advance as the first alphabetically named job at the newest shared milestone instead of selecting it by score. Its SHA-256 is `a5dadff5cc9f04a614caa828d3a0f014cad56f00c3644bb791275da85b35a8a6`.

## Design

We used two arms, each with three replicas and 12,000 training episodes per replica:

| Arm     | Training opponents                                     | Purpose                                                           |
| ------- | ------------------------------------------------------ | ----------------------------------------------------------------- |
| control | rule_based_agent, coin_collector_agent, peaceful_agent | Continue the soft line-up from final training.                    |
| hard    | Three rule_based_agents                                | Test whether regular combat exposure improves tournament hunting. |

All six runs started from the same migrated fallback. We copied the online and target-network weights and checked them tensor by tensor. We initialized replay and Adam state afresh. The learning rate was 0.0001, while reward mapping and architecture stayed unchanged. One of every five episodes used uniform legal-action exploration. The only planned factor was the training opponent line-up.

We evaluated the final checkpoints against the frozen fallback on fixed, disjoint Classic worlds with three `rule_based_agent`s. We also used classic-peaceful, coin-heaven, and loot-crate retention suites. The registered configuration, seeds, thresholds, and resource limits are in `config.json`. The seed audit is in `seed-audit.json`.

## Completion and integrity

All six training jobs completed their 12,000-episode budgets. The run recovered from the pinned framework's file-handle exhaustion without discarding observations. Recovery cost at most 25 episodes per affected job. Training, evaluation, and latency resource accounting all passed.

The analyzer found genuine updates and repeated behavioural evaluations. The serial latency maximum was 105.71 ms, above the registered 100 ms maximum. P95 latency was 17.88 ms, below the 50 ms limit. The latency integrity gate therefore failed, which also prevents promotion.

## Results

The primary Classic results below are means across the three final replicas. We evaluated them against the same registered tournament line-up.

| Artifact group            | Score/game | Kills/game | Survival | Self-kills/game |
| ------------------------- | ---------: | ---------: | -------: | --------------: |
| Frozen fallback           |      3.200 |      0.200 |    40.0% |           0.450 |
| Soft-line-up continuation |      3.033 |      0.108 |    39.2% |           0.575 |
| Hard-line-up continuation |      3.208 |      0.117 |    43.3% |           0.442 |

The hard line-up's score difference from the fallback was +0.008 with a registered paired hierarchical bootstrap interval [-1.067, +1.175]. Its elimination difference was -0.083 [-0.242, +0.092]. It therefore did not retain the fallback's hunting performance despite the additional combat exposure.

The hard arm exceeded the soft control by +0.175 score [-0.917, +1.258] and +0.008 eliminations [-0.150, +0.158]. These intervals are too wide to distinguish the two line-ups. The apparent survival advantage of the hard arm is not enough to support a promotion claim.

Retention also failed. Both arms passed the Classic peaceful checks, but the hard arm failed coin-heaven collection and both loot-crate checks. The control arm also failed the self-kill checks in coin-heaven and loot-crate. Neither arm had two non-worse final replicas, which was another registered requirement.

## Decision

The registered screen did not pass. Neither arm is promotable, and we select no checkpoint from this experiment. The fixed episode-8,000 fallback remains our submission candidate.

This experiment does not show that stronger opponents are always harmful. It shows that this specific 12,000-episode continuation, with empty replay and fresh optimizer state, did not improve the relevant kill endpoint or satisfy the retention requirements. The earlier shared continuation failure in issue #211 and the separate stability experiment in issue #213 are relevant when interpreting this result.

## Evidence and reproduction

We store the analysis as `analysis.json` in the run root. We produced it with `scripts/analyze_task4_competition_v2.py`. The run root currently contains `task4-competition-evidence.tar.gz`, its manifest, configuration, per-stage observations, bindings, and final checkpoints.

The evidence currently remains machine-local. Before we treat this experiment as review-ready evidence, we need to publish the exact archive at a durable retrievable location together with its size, SHA-256, retrieval instruction, and verification command.
