# Continuation stability

## Question

We tested whether preserving the mature replay buffer and Adam state prevents the rapid loss of established capabilities observed in #211. The tested factor is the combined two-state bundle, so this experiment does not separately attribute an effect to replay or optimizer moments.

## Baseline and controls

We use the checksum-pinned control-r1 milestone-008000 fallback and its exact 39-input runtime. Both arms retain online and target weights, episode and update counters, target synchronization phase, and rewards.

Both use a learning rate of 0.0001 and the same per-replica action and replay random seeds. `preserve` retains replay transitions and Adam moments. `reset` starts with an empty replay buffer and fresh Adam state. Both retain the same slightly lagged parent target network. We keep the original artifact read-only.

Both arms train against RUEHL_BASED_AGENT, `rule_based_agent`, and `peaceful_agent`. Training uses one fully random episode per block of five, with the other four greedy. Evaluation is greedy.

We bind source and input hashes during preparation. Evaluation runs against an archived framework without changing any shipped agent directory.

The experiment differs from #211 in learning rate and in its use of the original 39-input runtime. It tests the state-preservation comparison at these settings. It does not identify which change explains differences from #211.

## Budget and evaluation

We use three paired replicas with 100 new episodes per arm, giving 600 episodes in total.

We evaluate fixed final checkpoints only. Earlier immutable generations remain recovery and diagnostic artifacts and are not candidates for favorable selection. Exact training, evaluation, smoke, and bootstrap seeds are in `config.json`. The collision audit is in `seed-audit.json`.

Seven artifacts, six trained checkpoints plus the unchanged reference, receive the same 80 Classic worlds, 40 peaceful worlds, 40 coin-heaven worlds, 40 solo-crate worlds, and ten serial latency worlds. Starting slots rotate by world index. These are development worlds, not the final holdout.

The laptop-1 cap is two CPU-hours and four elapsed hours for the scientific pipeline, with one worker at a time, workload at most 4 GiB, and at least 2 GiB available system memory. This is a ceiling, not a reliable duration estimate. Preparation and bounded smoke checks are separate.

If the cap is reached, we stop the run and retain incomplete evidence rather than extending the budget.

## Decision rule

The `preserve` arm is eligible for further testing only if all registered config gates pass:

* Classic score within 0.30 of the reference
* self-kills at most +0.05/game
* survival loss at most five percentage points
* coin collection-fraction loss at most five percentage points
* solo-crate coin loss at most 1/game
* solo-crate self-kill increase at most 0.05/game
* Classic score improvement over `reset` of at least 0.30/game
* at least two of three paired replicas improve
* serial latency p95 below 50 ms
* serial latency maximum below 100 ms

We report paired bootstrap intervals over the three replicas and shared worlds, together with every gate.

These exploratory point-estimate gates do not certify non-inferiority or tournament superiority. Equal episode counts do not ensure equal optimizer updates, so we record both. Negative or inconclusive outcomes remain valid results. The experiment does not automatically promote a checkpoint or start longer training.

## Results and interpretation

All six training jobs completed 100 episodes. The analysis marks the run ineligible for further stability testing and submission promotion.

Against the unchanged reference, `preserve` averaged +0.075 Classic score/game with a 95% paired bootstrap interval of [−0.533, +0.692]. Kills increased by +0.008/game [−0.088, +0.108].

`preserve` failed the reference safety gates. Self-kills increased by 0.071/game and survival fell by 0.133. It also lost 6.31 coins/game in the solo-crate suite [−11.525, −0.775], with a 12.6 percentage-point reduction in crate collection fraction.

`reset` averaged +0.242 Classic score/game [−0.363, +0.913] and +0.033 kills/game [−0.050, +0.125]. It did not clear its score, safety, or crate-retention gates.

Neither arm established a reliable hunting improvement.

The serial latency check passed. P95 latency was 6.79 ms and the maximum was 49.21 ms.

Preserving replay and Adam as a bundle therefore did not stabilize continuation at these settings. The experiment does not identify replay versus optimizer state as the cause. The changed learning rate and runtime also prevent a causal comparison with #211.

We select no resulting checkpoint as a submission candidate.

## Evidence and follow-up

The complete evidence archive is published in the [Issue 213 stability release](https://github.com/1BlauNitrox/mle-final-project/releases/tag/issue213-stability-evidence-v1):

`issue213-results.zip`

Size: 13,845,506 bytes

SHA-256:
`ef4c96d1c61c3b67c4bee3e4af24395aaebb67f91226914d22f57f3e629fe872`

The archive contains the registered configuration, training observations and checkpoints, per-scenario evaluation observations, `analysis.json`, and a per-file checksum manifest.

The configuration SHA-256 is:

`e9338b9cc99c36edda589edb732a3b9fedb7e1f50a8ba80a17a7952ce25acef4`

After downloading and unpacking the archive, we inspect the recorded decision with:

```text
python -m json.tool analysis.json
```

The frozen fallback remains our safety baseline. Any future hunting intervention needs its own controlled comparison and must retain early-task performance before we consider a longer continuation.

Related motivation: Ball et al., [Efficient Online Reinforcement Learning with Offline Data](https://proceedings.mlr.press/v202/ball23a.html), studies the use of prior experience during online learning. This diagnostic does not implement RLPD and does not assume its results transfer to Bomberman DQN.
