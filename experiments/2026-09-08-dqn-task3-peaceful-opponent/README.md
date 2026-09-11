# Issue #109: prospective peaceful-opponent Task 3 experiment

Status: preparation only; no scientific execution or result. The user approved
these gates and 40 development pairs per suite on 2026-09-11. Owner:
1BlauNitrox; review requested from LiliWestermann, not yet approved. PR #120
integrates #125 / PR #130's 39-input migration. See [SERVER.md](SERVER.md).

## Hypothesis and controls

Opponent-aware training improves elimination and paired match outcome against
exactly one unchanged peaceful_agent while retaining Task 1 navigation and
Task 2 collection, crate destruction and survival. Five Task 3 replicas start
from one checksum-bound migration and train exactly 10,000 classic episodes
each. The frozen Task 2 parent is the single untrained reference. Learner is
command slot 0, opponent slot 1; world seeds determine starting corners.

The main variable is the existing Task 3 extension as a whole (opponent
features, native elimination reward and opponent training). This study does
not isolate their individual effects. Freeze the 39-input feature schema,
six actions, rewards, hidden sizes and inherited applicable hyperparameters,
masking and escape modes. Reset optimizer/replay/exploration/RNG/counters;
preserve online and target networks independently and zero the opponent suffix.
Evaluation is greedy, immutable, self-contained and single-threaded.
The exact model contract is [0008](../../docs/0008-task-3-opponent-awareness-contract.md).

## Parent and scientific scope

Task 2 completion is not required for this explicitly exploratory study.
Verified historical #107 A/r2 is available now; its failed Task 2 gates remain
explicit. The [migration record](../2026-09-10-task3-escape-migration/README.md)
provides its durable source, SHA-256, size and producing commit. A later
mechanically selected Task 2 checkpoint is configurable through the same
binding command. Preserve its source/selection evidence and use a new binding
and campaign directory; never silently replace a parent after Task 3 results.
An exploratory pass or retention of a weak predecessor does not complete
Task 2, #46 or #51. Frozen parent bytes are never modified.

## Seeds and budget

Training pairs: world 109001..109005, agent 209001..209005.
Each suite has 40 development pairs, one episode per pair, and an exact repeat:

| Suite | World seeds | Agent seeds | Opponents |
| --- | --- | --- | --- |
| classic-peaceful | 1091101..1091140 | 2091101..2091140 | peaceful_agent |
| classic-retention | 1091201..1091240 | 2091201..2091240 | none |
| coin-heaven-retention | 1091301..1091340 | 2091301..2091340 | none |
| loot-crate-retention | 1091401..1091440 | 2091401..2091440 | none |

The next ten values after each range are reserved confirmation seeds and never
executed here. Final held-out populations remain unopened. Preflight checks
committed run-plan YAML and experiment configurations for collisions and records
the inspected hashes. The owner must also confirm no unregistered remote use;
a repository audit cannot prove that. These replace the old ten-pair templates.

Budget: 50,000 training episodes; 1,600 candidate plus 320 reference evaluation
episodes including repeats. Candidate: 1,605 jobs; reference: 320 jobs. Select
only the final checkpoint after exactly 10,000 episodes, never a best checkpoint.

Registered ceiling: 24 CPU-hours, 15 wall-hours, 8 GiB aggregate framework-process
RAM, two training workers and serial evaluation. These are ceilings, not a
runtime prediction or compute authorization. One monitor covers both plans,
run sequentially, and persists usage on resume; wall time includes interruptions.
A breach stops only this campaign's registered process trees and preserves
failures. Do not reuse Task 2 resources or run on the RAM-constrained PC.

## Metrics and conjunctive gates

Elimination is the fraction of episodes with at least one native attributable
kill, not the mean event count. First place means strictly higher score than
the opponent; ties are separate. Score margin is own minus opponent score.
Filter by metadata's observed_agent; never pool opponent rows into observations.

| Gate | Requirement |
| --- | --- |
| Elimination | Rate >=0.60; candidate-minus-parent >=0.20 and 95% lower bound >0 |
| First place | Rate >=0.60 and candidate-minus-parent 95% lower bound >0 |
| Score margin | Mean >0 and candidate-minus-parent 95% lower bound >0 |
| Peaceful self-kill | Fraction of episodes with own self-kill <=0.10 |
| Invalid actions | Ratio of totals <0.01 for every candidate replica and aggregate in every suite |
| Task 1 collection | Coin-heaven collection-fraction difference 95% lower bound >-0.02 |
| Task 2 collection | Classic and loot-crate collection difference 95% lower bounds >-0.03 |
| Retention survival | Each retention-suite difference 95% lower bound >=-0.05 |
| Retention self-kill | Each retention-suite difference 95% upper bound <=0.02 |
| Task 2 crates | Each Task 2 suite: 95% lower bound of candidate - 0.9 * parent >=0 |
| Runtime | Every primary/repeat episode p95 <50 ms, max <100 ms; resource ceilings, immutable artifacts and exact deterministic repeats pass |

The crate difference avoids division by zero for a zero-crate parent. Report
absolute earlier-task results too; matching a poor parent is not capability.

For every contrast use 10,000 crossed bootstrap resamples, seed 109: independently
resample five replica indices and 40 common world/agent-pair indices, reuse each
reference observation across sampled candidates, and take the 2.5/97.5 percentiles
of the mean contrast. Do not treat 200 candidate episodes or repeated copies of
the reference as independent replicas. Intervals condition on the fixed parent.

Report per-model/aggregate score, elimination, coins and collection fraction,
crates, bombs, survival/steps, self-kills, invalid actions, action counts and
global decision-time median/p95/max from per-call times. Retain training episode
learning diagnostics, every zero-kill/death episode and all failed attempts.

## Decision and evidence

All gates must pass. Only then select the median replica by primary peaceful
elimination rate, ties ordered by replica ID. Otherwise record mixed/negative
exploratory results and stop; no automatic fallback or coin-collector launch.
Issue #137 tracks exploratory continuation after the peaceful decision; #51
retains its validated-predecessor requirements and remains open.

The analyzer rejects incomplete matrices, changed metadata/model bytes, CSV/raw
mismatches, missing kills/latencies, differing repeated actions/outcomes and
unbound resources. It writes observations.json.gz (evaluation observations,
per-call times and training diagnostics), source-manifest.json (input hashes and
sizes), and result.json (gates, intervals, summaries, resources and provenance).
Failed raw attempts stay in the campaign directory. Publish required evidence
durably with hashes, sizes and retrieval commands before making a result claim;
a local path alone is not evidence. Synthetic tests and isolated smokes validate
infrastructure only. No scientific result is in scope.
