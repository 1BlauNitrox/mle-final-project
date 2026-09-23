# Teacher-guided anti-loop comparison (#225)

## Goal

Issue #225 tests whether a fixed teacher loss helps the history-aware agent learn to reduce persistent loops while retaining the fallback's native score, kills, and earlier capabilities.

The executable experiment definition is stored in `config.json`.

The protected fallback is control-r1 milestone 008000 with checkpoint SHA-256:

`a5dadff5cc9f04a614caa828d3a0f014cad56f00c3644bb791275da85b35a8a6`

## Comparison

Both arms use the existing history features and the same conditional -0.1 loop penalty.

The only difference between the paired arms is the weight of the legal-action distillation loss:

* control: teacher-loss weight 0
* teacher-guided arm: teacher-loss weight 1
* temperature: 1

We exclude unknown historical replay and observed loop states from teacher supervision. The teacher remains frozen throughout training.

Evaluation only requires the student's self-contained agent directory and does not depend on the teacher.

## Training protocol

Both arms inherit the online and target networks, Adam state, replay buffer, and target synchronization phase from the fallback.

Before updates begin, both arms collect 5,000 fresh transitions.

Training uses Classic against three `rule_based_agent` opponents with matched seeds and rotating player slots.

We use:

* learning rate 0.0001
* one update per four transitions
* fixed checks after 100, 500, and 10,000 new updates
* a maximum of 1,000 episodes per arm
* two paired replicas on the PC

The teacher-guided arm must retain the fallback's earlier capabilities and receive enough generated and sampled loop experience before the run progresses beyond the pilot stage. We keep regressions in the control arm as part of the comparison instead of removing them from the analysis.

## Selection rule

The final comparison requires both paired replicas to complete.

The teacher-guided arm must pass all registered gates, including:

* at least 25% relative reduction in eligible loop frequency compared with both baselines
* nonnegative score difference
* nonnegative kill difference
* all registered retention and safety requirements

Only the fixed-final `memory-r1` checkpoint is eligible for independent confirmation.

We do not select intermediate checkpoints based on favourable results, and a passing result does not automatically replace the protected fallback.

## Resource limits

The complete campaign has a shared limit of 16 CPU-hours. This includes mechanics checks, training, evaluation, confirmation, and recovery.

We include a conservative 15 CPU-minute preflight debit in this budget.

Training stops no later than Monday 21 September at 10:00 Berlin time. The complete pipeline stops at 14:00.

The protected fallback remains available throughout the experiment.

## Evidence

We record the source commits, measured timing, execution commands, retained results, and final interpretation together with the experiment evidence.

The final decision is based on the registered comparison and retained observations rather than intermediate behaviour or visually promising checkpoints.
