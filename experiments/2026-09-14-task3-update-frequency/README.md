# Task 3 opponent update-frequency comparison

## Question and protocol

The hypothesis was that less frequent updates would limit destructive
opponent adaptation and improve hunting against `peaceful_agent`, while
preserving Tasks 1 and 2. The control updated every eligible transition; the
treatment updated every eight. Both arms retained all transitions in replay,
used the same fresh initialization, native kill reward `+5`, learning rate
`0.0005`, frozen inherited weights, five paired replicas, 100 training
episodes per replica, and fixed final checkpoints.

Evaluation comprised 3,520 episodes: ten trained final checkpoints plus the
unchanged reference, four suites, forty worlds per suite, and two repeats.
The suites were `classic` with `peaceful_agent`, `coin-heaven`, `loot-crate`,
and opponent-free `classic`. Training and evaluation seeds were disjoint and
the registered crossed paired bootstrap used 10,000 resamples.

Source/configuration provenance:

- issue #178, branch `experiment/178-update-frequency`;
- executed source `a489784` (bound tool source `fc11aa15cddbc47d954c5cc0d92c6f847f8df414`);
- configuration SHA-256 `418c29171ffc804445ed5af0a3bb74813b253f69cccf6a3f3a37134f689ff0b3`;
- initial checkpoint SHA-256 `8076ea7ddcf2c934a9a9ef21d3dc87833ededf860cf8fd8bf486d1e222c2ccf2`;
- reference checkpoint SHA-256 `99144d1688f66dcc6369d3efc02b2b9d5755cc8d21e6f2c1e1ffef466d0a7113`;
- runtime source `c4ddfa4efadf0b3ec6d4380a4239b9cb3a097113`.

## Results

Treatment minus control on the paired hunting suite was:

| Metric | Difference | Crossed paired 95% interval | Gate |
| --- | ---: | ---: | --- |
| Eliminations | `+0.055` | `[-0.035, +0.145]` | **fail**: required `>= +0.100` and lower bound `> 0` |
| Score | `+1.280` | `[+0.450, +2.100]` | descriptive improvement |
| Strict win | `+0.110` | `[+0.010, +0.220]` | descriptive improvement |
| Collection fraction | `+0.1117` | `[+0.0455, +0.1778]` | descriptive improvement |
| Self-kills | `+0.040` | `[-0.070, +0.160]` | pass: within registered margin |

Treatment mean eliminations were `0.145`, compared with `0.090` for control
and `0.075` for the reference. All five treatment replicas were non-worse than
the reference under the registered parent-retention criterion, and parent
retention passed. These gains do not establish a successful Task 3 result
because the primary elimination threshold and confidence requirement failed.

All three opponent-free retention suites passed, as did exact opponent-free
invariance, frozen inherited weights, genuine updates, and resource limits.
The invalid-action gate failed with eight retained invalid actions: treatment
replica 1 had 2, treatment replica 3 had 4, and control replica 4 had 2.
Latency also failed the maximum gate: median `5.431 ms`, p95 `19.370 ms`,
maximum `144.003 ms` against the registered `100 ms` maximum.

Training used 1,000 episodes and completed without failed jobs.

## Reproduction and evidence

The registered analyzer was run on the completed PC outputs with:

```powershell
$python = ".\\.venv\\Scripts\\python.exe"
& $python -m scripts.pilot_task3_stability analyze `
  --root training_outputs/campaign-v1 `
  --output training_outputs/campaign-v1/analysis.json
& $python -m scripts.pilot_task3_stability results `
  --root training_outputs/campaign-v1 `
  --output training_outputs/campaign-v1/task3-stability-evidence.tar.gz
```

The evidence archive is 74,494,938 bytes with SHA-256
`56c029257352854696314cd5ecc1dfbec3e885d117dcbc7bc4626afa722bf3f0`.
Its manifest SHA-256 is
`d3e3fe9c4b795c08e90f037872700c59e5fa6abe40878a00dcb0b1839d8437e6`.
Both files are published in the
[`issue178-evidence-v1` release](https://github.com/1BlauNitrox/mle-final-project/releases/tag/issue178-evidence-v1).

## Limitations and next step

The lower update frequency is promising for score, strict wins, and collection,
but it did not meet the preregistered hunting elimination gate and produced a
maximum latency violation. It must not be adopted as a Task 3 checkpoint. Any
follow-up should address the invalid-action and latency issues prospectively,
while preserving the same retention and integrity gates. This exploratory result
does not establish Task 2 completion.


## Decision

This exploratory comparison **failed its registered pilot screen**. Updating
the opponent-input weights every eight eligible transitions improved several
descriptive hunting measures, but the registered elimination gate was not met.
No checkpoint is selected or promoted, and Task 2 remains incomplete.
