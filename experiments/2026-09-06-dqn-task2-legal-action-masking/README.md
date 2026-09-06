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

This registered 100,600 episodes: 100,000 training and 600 paired evaluation
episodes.

## Result and decision

Both plans completed with five replicas, 50 primary episodes per arm/scenario,
and matching deterministic-repeat action digests. The mask achieved its
structural objective: its framework invalid-action rate was exactly zero in all
three primary suites. The unmasked rates were 14.99% (classic), 40.54%
(coin-heaven), and 37.61% (loot-crate).

The treatment is rejected for this configuration. Its paired collection
fraction intervals were negative or crossed the -0.05 non-regression margin:
classic -0.0267 [-0.0511, -0.0089], coin-heaven -0.0144 [-0.0584, 0.0224],
and loot-crate -0.0052 [-0.0156, 0.0040]. Survival also failed the all-scenario
non-regression rule, notably coin-heaven -0.10 [-0.26, 0.02]. This does not
establish that masking is universally harmful; it rejects this exact treatment,
curriculum, starting artifact, and development seed population.

The retained raw server archive has SHA-256
`841f01f86719a28d7a9d10d69685f6293c94e281b0dd39379d09947a4c180c1f` and is
not committed. Compact numeric evidence is in `result.json`.

After extracting that archive so its `training_outputs/run-plans` directory is
available locally, reproduce the compact evidence with:

```bash
python -m training.analyze_issue86_legal_action_masking \
  --plan-root training_outputs/run-plans \
  --output training_outputs/issue86-analysis
```

The generated `summary.csv` reproduces this record's numeric metrics; the
generated `result.json` additionally reports the registered decision gates.

```bash
tmux new -s issue86
python -m training.run_plan training/run_plans/issue86-dqn-task2-unmasked.yaml 2>&1 | tee logs/issue86-unmasked.log
python -m training.run_plan training/run_plans/issue86-dqn-task2-masked.yaml 2>&1 | tee logs/issue86-masked.log
```

The plans use at most two training workers each, 8 GiB RAM, and a 48 CPU-hour /
30 wall-hour ceiling across both arms. Detach with `Ctrl-b d`; resume only the
interrupted plan with `--resume`. Do not alter a plan, source tree, or artifact
between a failed run and its resume.
