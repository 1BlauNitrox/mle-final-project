# Task 4 trainable scope against strong opponents

## Decision

The registered screen did not pass. No arm is promotable, no checkpoint was selected,
and the installed Task 3 incumbent remains the submission candidate. The
comparison is still informative: it is the first measurement of any of our agents
against the tournament opponent line-up, and it answers the question it was
registered to answer.

## Question and protocol

Every Task 3 comparison froze the inherited network and trained only the 832
opponent-input columns. The registered question was whether that freeze is what
prevented improvement once the opponent inputs finally carry information the agent
needs — that is, against three `rule_based_agent`s rather than a passive opponent.

- Changed factor: `trainable_scope`, `opponent_columns` (control) against
  `all_weights` (treatment). Everything else is shared: the installed incumbent as
  the starting point, native rewards, no approach shaping, one optimizer update per
  eligible transition, the #168 episode-mixture exploration schedule, learning rate
  0.0005.
- 6 replicas per arm, 400 episodes per replica-arm, 4,800 training episodes.
- Primary endpoint: mean native score per game on `classic-rule-based` against the
  unchanged incumbent, carried through every arm as a no-training reference.
- Retention suites `classic-peaceful`, `coin-heaven` and `loot-crate` were gated in
  advance, with the registered risk stated before execution: an `all_weights` scope
  may lose Task 1 and Task 2 behaviour that the frozen scope preserved exactly, and
  a loss there is a failure rather than a trade-off to be argued away afterwards.
- Uncertainty: hierarchical paired bootstrap over replicas and shared worlds,
  10,000 resamples, 95% intervals, Bonferroni over the single promotable arm.

## Results

Arm means on the tournament suite (`classic`, three `rule_based_agent`s):

| artifact | score | strict win | eliminations | collection | survived | self-kills |
|---|---:|---:|---:|---:|---:|---:|
| reference (unchanged incumbent) | 2.350 | 0.100 | 0.050 | 0.233 | 0.300 | 0.625 |
| control (`opponent_columns`) | 1.221 | 0.037 | 0.029 | 0.119 | 0.125 | 0.825 |
| treatment (`all_weights`) | 2.650 | 0.196 | 0.096 | 0.241 | 0.458 | 0.462 |

The registered contrast, treatment minus control, with every interval excluding
zero:

| endpoint | difference | 95% interval |
|---|---:|---|
| score | +1.4292 | [+0.8875, +1.9958] |
| strict win | +0.1583 | [+0.0750, +0.2500] |
| collection fraction | +0.1218 | [+0.0815, +0.1634] |
| survived | +0.3333 | [+0.2167, +0.4458] |
| self-kills | −0.3625 | [−0.4833, −0.2333] |
| eliminations | +0.0667 | [0.0000, +0.1417] |

Eliminations is the one endpoint whose interval touches zero.

Two further contrasts were measured. Treatment against the unchanged incumbent is
positive on every endpoint but significant on none: score +0.30, interval
[−0.546, +1.113]. Control against the unchanged incumbent is significantly
negative, score −1.1292, collection −0.1139, survived −0.1750, self-kills
+0.2000, all intervals excluding zero. Training under the frozen scope produced an
agent measurably worse than not training at all.

Non-worse replicas: control 2 of 6, treatment 5 of 6.

### Retention

The registered risk materialised.

| suite | control coins | treatment coins | reference coins | verdict |
|---|---:|---:|---:|---|
| `classic-peaceful` | 0.071 | 0.242 | 0.331 | FAIL for both arms |
| `loot-crate` | pass | 0.166 | 0.228 | FAIL for treatment |
| `coin-heaven` | pass | pass | — | pass for both arms |

`earlier_task_retention` is therefore among the treatment arm's failed efficacy
gates, and is the reason the arm that won the primary endpoint is not promotable.

### Gates

Integrity: `resources`, `genuine_updates`, `behavioral_repeats`,
`opponent_free_invariance` and `latency` pass. `frozen_online_weights` fails, which
is the expected and correct behaviour for an arm whose registered purpose is to move
those weights; the gate is scope-conditional and is reported here rather than
silently skipped. `invalid_actions` fails under the known registered defect — the
gate counts the unchanged reference's own invalid actions, so no arm can pass it.

Efficacy, treatment arm: `exposure_increase`, `self_kills_versus_control`,
`self_kills_versus_reference` and `nonworse_replicas` pass; `hunting_improvement`,
`hunting_versus_reference`, `score_versus_reference_ci` and
`earlier_task_retention` fail.

Latency passed on the serial stage: median 5.86 ms, p95 15.17 ms, max 23.77 ms,
against registered limits of 50 ms and 100 ms.


## Limitations and next step

The comparison establishes the direction but not a usable agent. Full fine-tuning
buys tournament strength and pays for it in earlier-task capability, and the size of
that trade-off was not itself varied — scope was binary here, so the experiment
cannot say whether a smaller step or a narrower unfreeze keeps the gain while
holding retention. That is the registered follow-up.

The tournament baseline this run produced is worth recording on its own: the
installed incumbent scores 2.350 against three `rule_based_agent`s, wins 10% of
games outright and survives 30%.

Refs
