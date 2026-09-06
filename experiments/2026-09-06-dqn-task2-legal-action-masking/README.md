# Issue #86 legal-action masking

## Registered comparison

Compare `none` with `framework_legal` action masking. Both arms start from the
Issue #85 corrected migration artifact
`checkpoint-issue85-zero-suffix.pt` (`3edb2e7196030fcb52af6c7dc9ee69d9fc1259898ea674002fe06fbe93468015`),
use five paired replicas, the identical 2,000/2,000/6,000 episode curriculum,
and the same 10 development seed pairs for each opponent-free scenario. The
only intended difference is the persisted action-masking configuration.

The mask excludes only framework-rejected movement and unavailable BOMB; WAIT
is always legal. It has no blast-safety or strategic rule.

## Decision rule

Structural success requires zero framework-rejected selections in all primary
development evaluations. The masked arm is adopted only if its paired 95%
bootstrap lower bound is at least -0.05 for both collection fraction and
survival rate in every scenario, and it meets the existing p95 <50 ms and max
<100 ms decision-time limits. Otherwise retain the unmasked control and report
the negative result. Deterministic repeats must match their action digests.

## Server execution

This registers 102,400 episodes: 100,000 training and 2,400 paired evaluation
episodes. Run both plans in one tmux session after the PR is approved:

```bash
tmux new -s issue86
python -m training.run_plan training/run_plans/issue86-dqn-task2-unmasked.yaml 2>&1 | tee logs/issue86-unmasked.log
python -m training.run_plan training/run_plans/issue86-dqn-task2-masked.yaml 2>&1 | tee logs/issue86-masked.log
```

The plans use at most two training workers each, 8 GiB RAM, and a 48 CPU-hour /
30 wall-hour ceiling across both arms. Detach with `Ctrl-b d`; resume only the
interrupted plan with `--resume`. Do not alter a plan, source tree, or artifact
between a failed run and its resume.
