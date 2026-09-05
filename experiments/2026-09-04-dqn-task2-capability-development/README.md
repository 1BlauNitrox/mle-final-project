# DQN Task 2 capability development record

> Status: exploratory development work, not a registered experiment. No
> hypothesis, seed set, or success criterion was fixed before any of the
> four runs below; results from the first three directly shaped the next
> attempted change. This record exists to document what was tried, per
> `docs/0004`'s guidance that failed and exploratory work should still be
> recorded. None of the outcome-selected reward variants is shipped by this
> PR. It deliberately does not prescribe what to test next -- that is left
> open for whoever designs the next prospective experiment.

## Metadata

- Issue: #44
- Commits: `0c9f82e` (capability), `4489f9a` (escape-bug fix, first reward
  revision), `e0b9893` (second reward revision), `2704aa1` (round-4 source)
- Agent: `DagobertDuckDQNTask2`
- Date: 2026-09-03 to 2026-09-04
- Owner: 1BlauNitrox, with Claude (Anthropic) driving the runs and analysis
  under direction; see `docs/0006-ai-usage.md`.

## Why this exists

Issue #44 implemented the Task 2 capability (bomb/crate/danger features, a
six-action network migrated from the frozen Task 1 baseline, new rewards)
but authorized no scientific training. Before registering a real experiment
on top of it, four development runs checked whether the
implementation actually trains sensibly. The runs exposed specific problems,
but did not provide controlled evidence for the attempted reward changes.

## Round 1: the escape feature was measuring the wrong thing

**Setup:** `loot-crate`, world seed `44100`, agent seed `44`, stopped for
inspection at 129,091 of an (arbitrary, unbounded) 500,000-episode budget.

**Result:** a 20-episode greedy snapshot showed 18/20 deaths, mean coins
3.65/50, and -- more importantly -- a Task 1 regression check on
`coin-heaven` (no bombs, no crates) showed the invalid-action rate had risen
from ~0% at migration to 59%, with `BOMB` now selected 224 times in 10
rounds versus 0 at migration. Training on Task 2 was actively eroding Task 1
behavior.

**Root causes found:**

1. `escape_after_bomb` never added a hypothetical bomb to the danger map
   before checking for an escape route -- it measured "can I escape right
   now" instead of "would this bomb trap me." A real bug, not a tuning
   issue.
2. `MOVED_TOWARDS_COIN`/`MOVED_AWAY_FROM_COIN` had been inherited from the
   frozen parent without being re-examined for Task 2, despite issue #58's
   own result for that shaping being inconclusive.

**Correctness bug fixed in `4489f9a`.** The reward changes made in the same
commit were selected after observing this run and are not shipped by the
reviewed baseline. Full detail is in
`agent_code/DagobertDuckDQNTask2/README.md` ("Development history") and the
linked PR #74 comment.

## Round 2: a passive local optimum

**Setup:** `loot-crate`, world seed `44200`, agent seed `44`, `--rounds
500000` (the same oversized, arbitrary budget as round 1 -- itself a mistake,
see below). Checked at episodes ~5k, ~12k, and ~24k.

**Result:** Task 1 retention fully recovered (coin-heaven invalid-action
rate back to ~0%, `BOMB` still never selected) -- the round 1 fix held under
real training. But Task 2 itself stopped improving: `loot-crate` death rate
rose 40% -> 65% -> 70%, mean coins plateaued around 2.2-2.25, and the action
mix became lopsided (~43% `WAIT`, next to no `LEFT`/`RIGHT`, consistent
across different evaluation seeds).

**Diagnosis** (stated as a diagnosis, not a proven cause):

1. `--rounds 500000` was run against an epsilon schedule computed for a
   **10,000-episode** budget (see `config.py`'s docstring). Epsilon hit its
   floor around episode 8,000 -- under 2% of that run's target -- leaving
   little further exploration to escape a bad optimum for the remaining 98%.
2. `SURVIVED_ROUND` (`+5.0`) was large relative to
   `CRATE_DESTROYED`/`COIN_FOUND` (`+1.0`/`+2.0`), making passive survival
   plausibly competitive with actually engaging the round.

**Post-hoc variant tried in `e0b9893`**: `SURVIVED_ROUND` reduced to `+2.0`,
`WAITED` strengthened to `-0.3`. Those values were selected after observing
this run and are therefore recorded, but not shipped as the baseline.

## Round 3: rebalanced rewards, a properly-scaled run

**Setup:** `loot-crate`, world seed `44300`, agent seed `44`, `--rounds
15000` -- matched to the epsilon schedule this time. `summary.csv` contains
the retained 1,500-episode aggregates and `figures/` visualizes them. The raw
per-episode source used to create those files was not committed and is no
longer available, so this record is useful diagnosis but not independently
reproducible evidence.

**Result, binned across training** (epsilon-greedy, not the greedy
evaluation snapshots below):

| Episodes | Mean epsilon | Mean coins | Invalid rate | Mean steps | Mean abs. TD error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0-1,500 | 0.805 | 0.01 | 0.453 | 12.7 | 2.13 |
| 3,000-4,500 | 0.327 | 0.56 | 0.311 | 21.9 | 1.62 |
| 6,000-7,500 | 0.133 | 1.39 | 0.278 | 36.6 | 1.49 |
| 9,000-10,500 | 0.100 | 1.68 | 0.272 | 40.1 | 1.56 |
| 13,500-15,000 | 0.100 | 1.90 | 0.266 | 34.2 | 1.82 |

See `figures/learning-curve.png` (raw per-episode coins/score plus a
200-episode rolling mean), `figures/task-metrics.png` (episode length and
invalid-action rate over the whole run), and `figures/behavior-diagnostics.png`
(action distribution, epsilon decay, TD error).

**Greedy evaluation snapshots** (20 episodes on `loot-crate` at each point,
epsilon forced to 0, checkpoint otherwise untouched):

| Episode | Deaths | Mean steps | Invalid rate | Mean coins | Coin-heaven `BOMB` | Coin-heaven invalid rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ~4,500 | 4/20 | 324 | 43% | 1.1 | 0 | 19.4% |
| ~11,000 | 7/20 | 267 | 36% | 2.3 | 0 | 0.025% |
| 15,000 (final) | 12/20 | 179 | 27% | 3.3 | 0 | 0.0% |

**Interpretation.** The run is consistent with the stronger `WAITED` penalty
changing the action distribution, arguably too strongly at first (52%
`RIGHT` at episode ~4,500) before the distribution rebalanced
on its own by episode ~11,000 without further intervention. Task 1 retention
holds throughout. Invalid-action rate improved (43% -> 27%) but stayed
persistently high, and death rate did not converge monotonically (20% ->
35% -> 60% across the three snapshots) -- this is not a clean success.

**The most useful single observation**, visible only because per-episode
training data was available during analysis: **training-time survival is ~0%
throughout the entire run** (`summary.csv`'s `survived_round_rate` column
never exceeds 0.002), while greedy evaluation of the same checkpoints
survives 40-65% of episodes. This is the same shape of training/evaluation
gap issue #58 documented for Task 1, but the opposite direction: issue #58
found epsilon-forced randomness let the agent recover from states its greedy
policy would get stuck in, so evaluation (no randomness) did *worse*. Here,
a single random action at the wrong moment near a live bomb is enough to
cause instant, irreversible death, so epsilon-greedy exploration makes
*training* look far worse than the policy actually is. This is stated as an
observed pattern with a plausible mechanism, not a fully diagnosed cause.

## Round 4: repeat with a fresh world seed

**Setup:** `loot-crate`, world seed `44400`, agent seed `44`, 15,000 episodes,
source commit `2704aa12c47a1506c1c5890a07120efaabe5bed6`. The run completed in
4,007.36 seconds. Its raw `episodes.csv` remains intentionally ignored under
`training_outputs/`; SHA-256:
`c149851f23b1fbf9f60bb6b33261e23cf8981eb7cdbef4f919d9fd0ef2453942`.
The trained checkpoint is preserved in the local Git stash named
`user's local DagobertDuckDQNTask2 training run (15000 episodes, world_seed
44400) - do not discard`; it is not the committed migration checkpoint.

**Result:** mean coins `1.0848`, zero-coin rate `51.83%`, invalid-action rate
`22.83%`, survival `27/15,000` (`0.18%`), mean episode length `34.84`, and
mean shaped reward `-1.6362`. These values come from the runner's retained
`summary.json`; they are descriptive only. This was another unregistered run
of the post-hoc reward variant and cannot validate the variant or select a
new configuration.

## Decision

No reward configuration is adopted from these runs. The framework-correctness
fixes remain; all outcome-selected reward changes are reverted to the initial
Task 2 implementation defaults in the reviewed PR. The four runs are retained
as a transparent negative/process record and as possible motivation for a
future preregistered, controlled experiment.

The persistently high invalid-action rate (~23-43% across the measured
snapshots, despite the attempted reward revisions) is recorded here as a
known, unresolved limitation. It is not a new mystery in kind: the frozen
`DagobertDuckDQN` baseline's own README already flagged "legal-action
masking is deferred to Task 2, where a blocked move next to a live bomb is
fatal rather than merely wasteful," and a crate-dense board has illegal-move
opportunities everywhere that nothing in the current design prevents
selecting. What to do about it -- whether that is worth a registered
experiment, what the hypothesis and protocol should be, and what else might
be worth testing (hyperparameters, reward magnitudes, feature changes) -- is
intentionally left open here rather than prescribed.
