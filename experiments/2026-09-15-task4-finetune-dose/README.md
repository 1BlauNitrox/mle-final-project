# Task 4 fine-tuning dose-response

## Decision

The registered screen did not pass and no checkpoint was selected. The installed
Task 3 incumbent remains the submission candidate.

That verdict understates what the comparison found. The `reduced` arm is the first
arm in the Task 4 programme to beat the unchanged incumbent on the registered
primary endpoint with its interval excluding zero, while passing every retention
gate. What blocks it is an integrity gate that failed for a reason unrelated to the
agent, described below.

## Question and protocol

The trainable-scope comparison established that lifting the Task 3 freeze converts
strong-opponent experience into a better tournament agent, and that the same arm
loses earlier-task capability. Scope was binary there, so it could not separate
forgetting inherent to fine-tuning from forgetting that scales with how far each
update moves the inherited weights.

- Changed factor: `learning_rate` alone, over a registered ladder — 0.0005
  (`control`, the already-measured anchor), 0.0002 (`reduced`), 0.00005
  (`minimal`). Every arm shares the `all_weights` scope, three `rule_based_agent`s
  as the training line-up, native rewards and one update per eligible transition.
- 6 replicas per arm, 400 episodes per replica-arm, 7,200 training episodes.
- Primary endpoint: mean native score per game on `classic-rule-based` against the
  unchanged incumbent carried through as a no-training reference.
- Registered in advance: the control arm reproduces the dose that already failed
  retention, so at least one arm was expected to fail those gates.

## Results

Arm means on the tournament suite (`classic`, three `rule_based_agent`s):

| artifact | score | strict win | eliminations | collection | survived | self-kills |
|---|---:|---:|---:|---:|---:|---:|
| reference (unchanged incumbent) | 1.375 | 0.075 | 0.000 | 0.153 | 0.200 | 0.750 |
| control — lr 0.0005 | 2.121 | 0.146 | 0.079 | 0.192 | 0.308 | 0.583 |
| **reduced — lr 0.0002** | **2.083** | 0.113 | 0.062 | 0.197 | 0.304 | 0.621 |
| minimal — lr 0.00005 | 1.883 | 0.121 | 0.104 | 0.151 | 0.167 | 0.737 |

Score against the unchanged incumbent:

| arm | difference | 95% interval | |
|---|---:|---|---|
| control | +0.7458 | [+0.1000, +1.4625] | excludes zero |
| **reduced** | **+0.7083** | **[+0.1750, +1.2791]** | **excludes zero** |
| minimal | +0.5083 | [−0.0125, +1.0583] | includes zero |

Control also improves eliminations against the reference, +0.0792
[+0.0208, +0.1542]. No contrast *between* arms excludes zero: `reduced` − `control`
on score is −0.0375 [−0.7251, +0.5583], and `minimal` − `control` is −0.2375
[−1.0126, +0.5458]. All three arms have 6 of 6 non-worse replicas.

### The dose-response is in retention, not in score

Collection fraction on the opponent-free and passive suites, against a registered
margin of 0.05:

| suite | reference | control (5e-4) | reduced (2e-4) | minimal (5e-5) |
|---|---:|---:|---:|---:|
| `classic-peaceful` | 0.331 | 0.180 **FAIL** | 0.294 pass | 0.203 **FAIL** |
| `coin-heaven` | 0.487 | 0.614 pass | 0.658 pass | 0.405 **FAIL** |
| `loot-crate` | 0.177 | 0.097 **FAIL** | 0.137 pass | 0.113 **FAIL** |

Score is flat between 0.0005 and 0.0002 and falls at 0.00005, while retention is
lost at the largest dose and recovered at the middle one. The middle dose therefore
dominates: it buys the same tournament gain as the largest and is the only arm in
the ladder that keeps the earlier-task behaviour.

The `minimal` arm failing retention is worth stating plainly, because it is not
what a pure step-size story predicts. The smallest dose lost capability on all
three suites while gaining least on the tournament suite, which points at the
smallest dose training *badly* rather than training *little*.

### Gates

Efficacy, `reduced`: `score_versus_reference_ci`, `earlier_task_retention`,
`nonworse_replicas`, `self_kills_versus_control`, `self_kills_versus_reference` and
`hunting_versus_reference_ci` all pass. The exposure and elimination-threshold
gates fail.

Integrity: `resources`, `genuine_updates`, `opponent_free_invariance` and `latency`
pass. `frozen_online_weights` fails as expected for the `all_weights` scope every
arm registers. Two others require comment.

`invalid_actions` fails, and unlike in the earlier comparisons the failure is
informative: 15 of the 18 trained artifacts exceed the unchanged reference's own
count of 40, the worst at 126. Fine-tuning is making the agent attempt more illegal
moves. This is a finding about the agent, not only about the known defect in the
gate's global-zero formulation, and it deserves its own investigation.

`behavioral_repeats` fails, and this one is a measurement artifact rather than a
property of any agent. Four of 3,040 repeated evaluations disagree — 0.13% — and
three of the four are on the *unchanged reference*, which cannot have been altered
by training. Each disagreement shows an agent losing or gaining a single action in
a 400-step episode (`attempted_actions` 400 against 399), the signature of a step
exceeding its decision-time budget under load, after which the episode diverges.
The evaluation stage runs five concurrent workers, and this machine was doing
unrelated heavy work during that stage. The runner's own documentation claims
evaluation "is deterministic however many run at once"; these four rows show that
claim is false in the presence of a decision-time limit.

Reported metrics are taken from repeat 0 throughout, so the arm means and intervals
above are unaffected. The gate is doing its job and is not relaxed here.

## Reproduction and evidence

18 training jobs, 76 evaluation, 19 latency, zero failed. Latency passed on
the serial stage: median 5.95 ms, p95 16.16 ms, max 32.06 ms. Provenance, checksums
and the reproduce command are in `results/verification.json` and
`results/evidence.json`.

## Limitations and next step

The unchanged incumbent scored 1.375 here, 2.200 in the opponent-mixture comparison
and 2.350 in the trainable-scope comparison, on three independent forty-world sets.
That is a spread of about one point of score for an agent that did not change,
which is far larger than this campaign previously assumed and means no absolute
score may be compared across comparisons. Every contrast reported here is paired
within this run and is unaffected, but the earlier claim that world-set noise is
around 0.15 is wrong and should be read as roughly 1.0.

The registered follow-ups are: re-run this evaluation stage on an idle machine to
obtain a determinism-clean measurement of the same checkpoints, and investigate the
rise in invalid actions under fine-tuning. The learning rate for the long training
run is 0.0002 on this evidence.

Refs
