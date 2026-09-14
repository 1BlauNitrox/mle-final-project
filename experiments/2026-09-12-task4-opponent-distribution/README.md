# Issue 126: proposed fixed opponent-distribution comparison

Preparation only, not Ready for scientific execution. Owner Julius / 1BlauNitrox;
request LiliWestermann review. Refs #126 and PR #145's scheduling decision.
No result, trained parent, execution authorization or tournament readiness claim.

The single factor is training opponent distribution: three fixed rule_based_agent
opponents versus a fixed three-opponent lineup of rule_based_agent,
coin_collector_agent and frozen DagobertDuckDQN. The learned opponent must retain
its committed frozen Task 1 artifact and source fingerprints. This is a fixed
lineup, not online self-play, a changing pool or a new model family. Opponent
command order is fixed as listed; world placement randomizes starting corners.

Five paired training roots per arm, 5,000 classic episodes each (50,000 total),
identical starting online/target weights, fresh optimizer/replay/epsilon/RNG;
inherit parent rewards/features/hyperparameters/masking unchanged. Initialize
only from the mechanically selected passing reviewed #137 final training parent.
That parent is unavailable. A different exploratory parent requires a distinct
prospective owner decision; no #109 best-seed or #91 substitution is implied.

Nine matched evaluation suites: one and three rule_based_agent opponents,
the fixed mixed lineup, one frozen learned opponent, one peaceful_agent, one
coin_collector_agent, and opponent-free classic/coin-heaven/loot-crate retention.
Each uses forty common world/agent pairs and exact repeats, across ten trained
models and one untrained frozen parent: 7,920 episodes including repeats.
Explicit disjoint populations and unused confirmation ranges are in the YAMLs.
Learner is command slot 0; supplied opponent RNG isolation must remain active.

Proposed primary endpoint: mixture-minus-strong-only native score against three
rule_based_agents >=1 point with positive paired crossed-bootstrap 95% lower
bound (10,000 draws, seed 126). Replicas and common seed pairs are resampled
jointly. Additional conjunctive treatment gates: strict first-place >=35%,
positive score margin and positive score-improvement lower bound over frozen
parent on that suite; aggregate self-kills <=10% and invalid actions <1% per
model and aggregate; prior collection (-2 pp coin heaven, -3 pp classic/loot), crate (90%),
survival (-5 pp) and self-kill (+2 pp) retention bounds; peaceful/collector
elimination-retention lower bounds >=-5 pp; p95 <50 ms and max <100 ms on every
primary/repeat episode; complete provenance, immutable models and exact repeats.
Report every arm/gate, not only the primary score. Only a passing treatment may
provisionally select its median final replica by primary score, ties replica ID.
Otherwise no new selection; no favorable control or seed fallback. These are
proposed numeric decisions, requiring owner ratification before training.

`python -m training.task4_protocol --dry-run` expands the full matrix and audits
seeds. The tested statistical decision function is for already verified primary
rows, explicitly marked statistics-only; it does not claim raw integrity. The
shared Task 3 raw loader provides the reusable evidence path. Remaining readiness
work is the selected-parent binding, inherited mode/weight validation, frozen
opponent artifact inventory, and bound execution/raw-evidence adapter. No raw
run-plan launch is permitted while those are missing; templates contain no parent.

Proposed cap: 24 CPU-hours, 10 wall-hours, 8 GiB, two training workers and serial
evaluation on one server environment for both arms. This is **not yet a credible
overnight ETA**: nine suites and four-player trajectories differ materially from
#109. Time a small isolated integration workload after binding and verify the
complete training/evaluation/export allocation before approval. Reduce a new
prospective episode design or allocate another window if it cannot fit; never
extend a running budget or silently drop suites. PC can prepare these tools while
#147 runs; no second speculative training run and no host-treatment confound.

The September 17 compatibility, September 19 freeze, September 20 audit and
September 21 21:00 deadline remain protected. If this preparation slips, drop
optional tuning before sacrificing evidence or packaging. AI assistance: Codex
prepared the proposed matrix, statistics and documentation; none is run evidence.
