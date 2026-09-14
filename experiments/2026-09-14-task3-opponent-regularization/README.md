# Task 3 opponent-weight regularization

## Decision

This exploratory comparison failed its registered pilot screen. Applying
L2 regularization (`lambda = 0.01`) to the 832 opponent-input weights did not
produce the required hunting improvement over the unregularized control. No
checkpoint is selected or promoted, and Task 2 remains incomplete.

## Question and protocol

The hypothesis was that penalizing only the newly trainable opponent weights
would reduce destructive adaptation and improve hunting against
`peaceful_agent`, while preserving Tasks 1 and 2. The control used
`lambda = 0`; the treatment used
`L_DQN + 0.5 * lambda * sum(W_opponent^2)` with the exact `lambda * W` gradient
added before the existing norm-10 clipping. Both arms updated every eligible
transition, froze the inherited network, and used the same fresh initialization,
native kill reward `+5`, learning rate `0.0005`, five paired replicas, 100
training episodes per replica, and fixed evaluation checkpoints.

Evaluation comprised 3,520 episodes: ten trained final checkpoints plus the
unchanged reference, four suites, forty worlds per suite, and two repeats.
The suites were `classic` with `peaceful_agent`, `coin-heaven`, `loot-crate`,
and opponent-free `classic`. Training and evaluation seeds were disjoint and
the registered crossed paired bootstrap used 10,000 resamples.

Source/configuration provenance:

- issue #179, branch `experiment/179-opponent-regularization`;
- executed source `08ecb39e32c2790b5e7e573efb91f058ea81eb0b`;
- configuration SHA-256 `992563d97e3524fdac26f240836e7e7356aed6a4d608b3674e03f411ff113cd2`;
- initial checkpoint SHA-256 `8076ea7ddcf2c934a9a9ef21d3dc87833ededf860cf8fd8bf486d1e222c2ccf2`;
- reference checkpoint SHA-256 `99144d1688f66dcc6369d3efc02b2b9d5755cc8d21e6f2c1e1ffef466d0a7113`;
- runtime source `c4ddfa4efadf0b3ec6d4380a4239b9cb3a097113`.

## Results

The primary hunting result is treatment minus control across paired replicas
and worlds:

| Metric | Difference | Crossed paired 95% interval | Gate |
| --- | ---: | ---: | --- |
| Eliminations | `+0.045` | `[-0.045, +0.140]` | **fail**: required `>= +0.100` and lower bound `> 0` |
| Score | `+0.545` | `[-0.470, +1.660]` | descriptive only |
| Strict win | `+0.020` | `[-0.095, +0.135]` | descriptive only |
| Collection fraction | `+0.0356` | `[-0.0394, +0.115]` | descriptive only |
| Self-kills | `+0.050` | `[-0.075, +0.190]` | pass: within registered margin |

The treatment mean elimination rate was `0.145`, versus `0.100` for control
and `0.250` for the reference. Only one of five treatment replicas was
non-worse than the reference under the registered hunting criterion; the
required minimum was four. Treatment was therefore also below the parent
retention gate.

All three opponent-free retention suites were unchanged between control and
treatment. Their collection and self-kill retention checks passed. Exact
opponent-free behavior, frozen inherited weights, genuine updates, resource
budgets, and latency passed. Latency was median 1.823 ms, p95 5.790 ms,
maximum 65.939 ms.

The invalid-action gate failed: the retained matrix contains 16 invalid
actions in total, including 10 for the reference, 2 each for control replicas
1 and 4, and 2 for treatment replica 5. These failures are preserved rather
than filtered. No evidence supports adopting the treatment.

Training used 1,000 episodes and completed without failed jobs. Training took
2,440.9 seconds wall time with a 329.6 MB peak; evaluation took 2,836.4
seconds wall time with a 239.4 MB peak. All 44 evaluation jobs completed.

## Reproduction and evidence

The registered analyzer reproduced the result from the transferred archive:

```powershell
$python = ".\\.venv\\Scripts\\python.exe"
& $python -m scripts.pilot_task3_stability import `
  --root training_outputs/result-verification-01 `
  --archive task3-stability-evidence.tar.gz `
  --sha256 38cf0b17e6cd141793432ef35efe0bc23a02e6dcafdeb57da602e1acb665152b
& $python -m scripts.pilot_task3_stability analyze `
  --root training_outputs/result-verification-01 `
  --output training_outputs/result-verification-01/analysis-reproduced.json
```

The archive supplied for this result is 73,183,147 bytes with SHA-256
`38cf0b17e6cd141793432ef35efe0bc23a02e6dcafdeb57da602e1acb665152b`.
The durable archive and its manifest are published in the
[`issue179-evidence-v1` release](https://github.com/1BlauNitrox/mle-final-project/releases/tag/issue179-evidence-v1).

## Limitations and next step

This is an exploratory Task 3 comparison, not evidence that Task 2 is complete.
The intervention preserved earlier tasks but did not resolve the hunting
deficit. The next experiment should target a separately justified cause of the
opponent-conditioned policy drift. It must be prospectively registered and
must keep the same retention and integrity gates.
