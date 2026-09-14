# Task 3 learning-rate retention pilot — DRAFT

## Question and setup

Does reducing Adam's learning rate from **0.0005 to 0.00005** improve retention
under episode-based exploration without reducing hunting? Both arms used the
same 80% greedy / 20% random episode schedule and fresh #168 initialization.
Only learning rate changed, in configuration and optimizer state; initial online
and target weights matched. Three paired replicas trained for 50 episodes per
arm on classic against one peaceful agent. All six fixed-final checkpoints and
the unchanged reference were greedily evaluated on four suites, five fresh
worlds per suite, with exact repeats: **300 training and 280 evaluation episodes**.

[config.json](config.json) preserves the prospective seeds, controls, screens and
budgets. Runtime source was `c4ddfa4efadf0b3ec6d4380a4239b9cb3a097113`;
runner/analyzer source was `5bee5a723c7f4857c5e4f13959ef8401a14ade6c`.
Input hashes, payload checks and final artifacts are in
[execution-verification.json](execution-verification.json) and [checkpoints.csv](checkpoints.csv).

## Results

The retention pilot failed. No checkpoint was selected or promoted.
The table uses primary episodes only; repeats add no independent observations.

| Metric / suite | Control | Lower rate | Reference |
| --- | ---: | ---: | ---: |
| Task 1 collection, coin heaven | 38.8% | 45.6% | 20.4% |
| Collection, loot crate | 21.47% | 16.27% | 30.40% |
| Collection, classic empty | 34.81% | 37.78% | 46.67% |
| Hunting, eliminations/game | 0.267 | 0.133 | 0.000 |
| Self-kills, classic empty | 13.33% | 20.00% | 0.00% |

Lower-rate minus control Task 1 collection was +6.8 percentage points
missing the required +10 points; descriptive 95% interval [-37.47, +57.20]
points. Hunting difference was -0.133 eliminations/game, interval
[-0.533, +0.267], so it was worse then the Control.

Coin-heaven retention passed. Loot-crate collection was 14.13 points below the
reference, failing its -5-point margin. Classic empty collection was 8.89 points
below reference and self-kills 20 points above it, both failed. Hunting against
the reference and all three treatment replicas' nonworse-reference checks
passed, but the reference eliminated no opponents on these five worlds.

All six models genuinely updated (12,380–13,362 optimizer updates); all 140
behavioral repeat pairs matched. One invalid action occurred in control r2,
classic-peaceful world 5931405, and recurred in its exact repeat. Treatment had
zero invalid actions; the registered whole-matrix zero-invalid gate still failed.
Its cause was not established and no diagnostic games were added. Decision
latency passed: median 5.91 ms, p95 17.35 ms, maximum 32.53 ms.

All effects and paired intervals are in [contrasts.csv](contrasts.csv), with
[per-replica results](per-replica.csv), [all gates](gates.csv),
[training diagnostics](training-summary.csv) and [paired-effects figure](figures/paired-effects.png).
Intervals use the registered 10,000-draw crossed replica/world bootstrap and
are descriptive, with only three replicas and five shared worlds.

## Interpretation and next step

This pilot does not support lower learning rate as a sufficient retention remedy.
The Task 1 point estimate improved, but missed the screen; hunting and other
retention results prevent adoption. Wide intervals do not establish general harm
or benefit. The identical reference scored 20.4% Task 1 collection here versus
88% in #168 on different worlds: cross-study means cannot rank checkpoints or
show that the unchanged reference deteriorated. No favorable replica is selected.

Retain the negative result and review it before proposing another controlled
experiment. Do not launch coin-collector continuation, extend this budget or
change defaults. Task 2 remains cumulatively incomplete, and the original
Task 3 requirements remain unmet.

## Verification and evidence


The verified archive is 93,712,377 Bytes, SHA-256
`768e040c416730b820b42cacce3a5d113a98f7b2929a71f92f58ce1963ca6708`.
Its 966 members are listed in [archive-members.json](archive-members.json).
[observations.json.gz](observations.json.gz) retains compact native observations
and timings; [analysis.json](analysis.json) is the registered result.
[evidence-manifest.json](evidence-manifest.json) contains exact import/analysis
commands and source requirements..
