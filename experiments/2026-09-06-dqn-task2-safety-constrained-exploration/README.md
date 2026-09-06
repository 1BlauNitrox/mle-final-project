# Issue #98 safety-constrained exploration for Task 2 DQN

> Status: **Backlog — not specified enough to implement.** This is a
> registration of the owner's one-line proposal, not an executable protocol.
> See "What's missing" before doing anything else with this directory.

## Hypothesis and single factor

Restricting only the epsilon-random exploration draw to actions that are both
framework-legal and immediately safe (would not enter a lethal blast at
arrival) may reduce self-kills and improve usable training data, without
changing the learned policy's own action-value estimates. Greedy action
selection, evaluation, and the Bellman target remain governed by legality
alone (Issue #86's mask, or none) — unchanged by this treatment.

This isolates the exploration-time safety restriction as a factor separate
from #86: #86 changes what the greedy policy and Bellman target may select;
this issue changes only what epsilon-random exploration may sample during
training.

## Proposed comparison

- **A (control):** the winning arm from #86 / #97 — masking mode and
  curriculum schedule as already decided by those experiments.
- **B (treatment):** identical to A, except epsilon-random exploration draws
  only from actions that are legal *and* immediately safe. Greedy/evaluation
  behavior is byte-identical to A.

## What's missing

This proposal is a single sentence, not a full protocol like #86 or #97. Before
this can move out of Backlog, the owner needs to specify:

- the exact definition of "immediately safe" — e.g. reuse
  `safe_direction`/`build_danger_map` from
  `agent_code/DagobertDuckDQNTask2/features/bombs_and_crates.py` as-is, or a
  narrower/different rule specific to this treatment;
- what happens when the legal-and-safe set is empty during exploration
  (fall back to legal-only? to `WAIT` only? something else?);
- replicas, seeds, and curriculum — presumably reusing #86/#97's, but not yet
  confirmed as appropriate here;
- the primary effect and non-regression thresholds, following the #46/#86/#97
  pattern.

No code, run plan, or config is prepared here yet — writing the masking logic
before the above is settled would mean guessing at the owner's intent for
exactly the kind of design decision they've asked to make themselves.

## Dependencies

Depends on #86 and #97 both having adopted outcomes, to fix arm A's exact
configuration.
