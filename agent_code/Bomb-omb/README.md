# Bomb-omb

## Purpose

Bomb-omb is the learned agent we selected for the final tournament submission. It uses a Deep Q-Network to combine coin collection, crate destruction, bomb escape, and play against opponents.

## Model

We use 39 public game-state features and a neural network with two hidden layers of 64 units each. The agent chooses between `UP`, `RIGHT`, `DOWN`, `LEFT`, `WAIT`, and `BOMB`.

Evaluation is deterministic, uses one CPU thread, and does not require multiprocessing or files outside this directory.

The state includes board layout, coins, bombs, safe escape information, and public opponent positions. The model and feature implementation are contained in this directory. `requirements.txt` lists the only additional runtime dependency.

## Training and selected checkpoint

This checkpoint comes from our warm-lineup training run. It completed 8,000 episodes and uses legal-action masking and escape-continuation features.

We record the final checkpoint and its exact provenance in `artifact.json`.

## Why we selected this candidate

We compared Bomb-omb with the conservative fallback and the latest pursuit candidate using fresh, matched game seeds.

Bomb-omb matched the fallback on the coin and crate suites. It performed better in the mixed-opponent suite, with a higher average score and more opponent eliminations.

Based on these results, we selected it as our tournament candidate.

## Known limitations

Bomb-omb does not perform better in every setting. It had weaker survival in the three-rule-based-opponent suite, and visual tests still show occasional right-left movement loops.

We therefore treat this checkpoint as our practical tournament choice rather than evidence that we solved all navigation or hunting problems.

## Verification

We tested the packaged directory locally and in the supplied Docker runtime against three `rule_based_agent` opponents.

We retain the detailed benchmark data and technical verification record with the experiment evidence.
