# DagobertDuckDQNAntiLoop

## Model and hypothesis

This agent is a self-contained CPU DQN with a short causal visit history. In issue #225, we test whether guidance from a fixed copy of the fallback preserves useful behaviour while the student learns to avoid persistent loops. Reducing loops is the primary objective. Native score and opponent eliminations must not regress under the registered selection rules.

The network is `56 -> 64 -> 64 -> 6`, with ReLU hidden layers and actions `UP RIGHT DOWN LEFT WAIT BOMB`. Inputs consist of the original 39 navigation, crate, bomb-escape, and opponent features, twelve inactive geometry inputs, and five visit-count features describing the previous sixteen observations.

We reset history between rounds. Cached observations prevent double counting. Framework legality masking stays active. During evaluation, the learned network selects actions without a hand-written anti-loop override.

Algorithm references: Mnih et al., [Human-level control through deep reinforcement learning](https://doi.org/10.1038/nature14236); Schmitt et al., [Kickstarting Deep Reinforcement Learning](https://arxiv.org/abs/1803.03835). The teacher loss below is our project-specific DQN adaptation, not a reproduction of the complete kickstarting method.

## Earlier experiment #221

Issue #221 introduced the history-aware anti-loop pilot. Its control used inactive memory and no added loop penalty. Its treatment enabled both.

We keep its protocol and evidence separate from #225. We do not select an intermediate #221 checkpoint as a new parent or reinterpret the pilot as successful.

## Current comparison #225

Both arms now use active memory and the same conditional movement penalty. They differ only in the training-only teacher-loss weight: zero for `control` and one for `memory`, which remains the directory label for the teacher-guided treatment.

The loss combines the ordinary DQN Huber TD loss with the weighted KL divergence from teacher to student legal-action distributions, using softmax temperature one.

Teacher supervision applies only to fresh, observed non-loop replay entries. The teacher stays fixed to the initial widened fallback. Historical replay has no reconstructible visit history, so we exclude it from teacher supervision.

We persist causal loop labels and pre-action legal masks alongside replay and sample them using the same indices. We differentiate the combined loss before gradient clipping.

Evaluation does not require the teacher or any training-only module.

## Rewards and exploration

The inherited reward mapping stays in `config.py`. Both arms add -0.1 only for a successful directional return into a persistent loop when all of the following conditions hold:

* The preceding 24 consecutive observations cover at most three positions.
* Those observations contain no bombs or explosions.
* Field, visible coins, score, and opponent count stay unchanged.
* The resulting state is also calm and unchanged.
* The transition contains no collection, crate destruction, kill, or invalid-action event.

BOMB, WAIT, blocked movement, and active bomb escape receive no additional loop penalty. The existing WAIT reward stays unchanged.

Shaping applies with or without crates. The registered primary loop metric measures eligible crate-free windows. Terminal handling retains a previously observed penalty exactly once.

One seeded episode per five uses uniformly random legal actions. The other episodes are greedy. Evaluation is greedy with fixed agent seeds.

## Parent and training protocol

Both arms start from the fixed Task 4 `control-r1` milestone `008000` fallback:

* Checkpoint SHA-256: `a5dadff5cc9f04a614caa828d3a0f014cad56f00c3644bb791275da85b35a8a6`
* Size: 3,436,279 bytes
* Parent state: 8,000 completed episodes and 1,485,852 optimizer updates
* Trained runtime commit: `c4ddfa4efadf0b3ec6d4380a4239b9cb3a097113`
* New implementation provenance: we record the immutable #225 source commit from the campaign binding after preflight

Migration preserves online and target weights, Adam moments, replay, and target phase. New input columns and their Adam moments start at zero. We zero-pad missing historical feature values rather than reconstruct them.

We train two paired replicas on this PC in Classic against three seeded `rule_based_agent` opponents, with rotating slots and matched seeds.

Both arms collect 5,000 fresh transitions before updates. The learning rate is 0.0001, batch size 64, replay capacity 10,000, discount 0.9, one update per four transitions, target synchronization every 500 actual updates, and gradient norm limit 10.

Fixed checkpoints occur at 100, 500, and 10,000 new updates, with a ceiling of 1,000 episodes per arm. The full seed sets, inherited settings, and thresholds are in [the registered configuration](../../experiments/2026-09-20-teacher-loop/config.json).

The shared ceiling is 16 CPU-hours, including preflight, pilot, training, evaluation, confirmation, and recovery, with at most two single-thread game workers.

Training stops by Monday 21 September 2026 at 10:00 Berlin. The full pipeline stops by 14:00. We record actual hardware, package versions, source hashes, and resource consumption in the campaign binding and resource ledger. These limits do not represent an elapsed training-duration estimate.

## Evaluation and selection

The teacher arm must pass early retention checks and receive generated and sampled loop experience. Control regressions remain part of the comparison.

Final suites cover Classic, mixed and peaceful opponents, Coin Heaven, Loot Crate, and serial latency testing.

Selection requires:

* both replicas complete
* at least 25% relative reduction in eligible loop frequency versus control and fallback
* nonnegative native score difference
* nonnegative kill difference
* all registered survival, self-kill, collection, invalid-action, and latency gates

Multiplayer invalid-action increases are capped at 0.25 per game pooled and 0.5 per replica versus both baselines. Solo evaluations require zero invalid actions. Latency-only games are excluded from the invalid-action comparison.

Decision latency must stay below 50 ms p95 and 100 ms maximum.

Only the fixed-final `memory-r1` checkpoint is eligible for independent confirmation on fresh registered worlds. We do not select attractive intermediate checkpoints.

We report paired uncertainty, per-replica effects, loop denominators, attack exposure, bomb-attributed kills, replay origins, and teacher and loop sampling counts.

No result is available yet. Passing the screen does not automatically promote the checkpoint for submission.

## Runtime and limitations

Evaluation requires NumPy and CPU PyTorch, with dependencies declared in `requirements.txt`.

We install the selected evaluation checkpoint as module-relative `checkpoint.pt`. Temporary campaign checkpoints stay outside this directory.

Evaluation uses one inference thread, no multiprocessing, and no imports from training tools or other agents.

Teacher guidance might preserve undesirable habits or constrain useful changes. Sparse loop and attack exposure, together with the small replica count, limit inference.

A reduction in eligible looping might be misleading if survival or opportunity counts change. We therefore report those denominators together with the native metrics.

Any selected candidate still requires clean-framework, Docker, latency, and packaging checks before submission. We add final results, limitations, and artifact provenance after the run.
