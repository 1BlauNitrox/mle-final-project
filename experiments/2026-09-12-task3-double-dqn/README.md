# Issue 150: standard versus Double DQN targets

**Prospective exploratory protocol; no scientific result or selected model.**
Owner Julius / 1BlauNitrox; reviewer requested LiliWestermann. Refs #150,
#146, #147 and the unmerged #145 roadmap. Task 2 remains incomplete.

The #146 development diagnostic reproduced learned stationary and legal
two-tile stalls even with masking. Representative cycles have no visible coins
and zero movement reward. This does not justify coin-distance shaping for those
states or establish an implementation defect. Online/target action disagreements
motivate testing target learning, not a claim of proven overestimation relative
to optimal return. [Double DQN](https://arxiv.org/abs/1509.06461) separates online
action selection from target-network evaluation to address maximization bias;
its published success in other environments is not Bomberman evidence.

## Hypothesis and exact intervention

Double DQN improves peaceful-opponent elimination over otherwise identical
masked standard DQN, while meeting every unchanged hunting/retention gate.
Control bootstraps with max_a Q_target(s',a). Treatment bootstraps with
Q_target(s',argmax_a Q_online(s',a)); both restrict the maximization to the same
framework-legal actions. Terminal targets contain no bootstrap. Online ties
use the first legal action in UP/RIGHT/DOWN/LEFT/WAIT/BOMB order.

Both arms freshly migrate original provisional #91 A/r3, source
`cbd52be8392f5003a91c5600fda4efd544b48ec5`, 2,382,903 bytes, SHA-256
`c0b6081acf5a733258005d46b9430f3ef50e403470a3bb5428661729f9aeedf7`.
No trained #147 replica is selected. Preserve online/target networks separately,
zero opponent columns and reset optimizer/replay/epsilon/RNG/episode counts.
Enable masking in both fresh initializations; their complete payloads differ
only in the persisted `double_dqn` flag. Exact binding validation is mandatory.
The frozen reference is the original unmodified Task 2 parent; matching the
weak parent does not establish absolute cumulative capability.

Unchanged: 39 inputs, two 64-unit ReLU layers, six actions, active escape features,
rewards, gamma .9, Adam .0005, batch 64, replay 10,000 / warmup 500, target sync 500,
gradient clip 10, epsilon 1 -> .1 at decay .9997. No anti-loop rule, reward change,
new feature, held-out tuning or automatic promotion of masking as a default.

## Matrix, numeric gates and selection

Five paired training replicas per arm, exactly 10,000 classic episodes each
against one seeded peaceful_agent, learner slot 0 / opponent 1. World/agent roots
150001..150005 / 250001..250005. Four 40-pair development suites use world seeds
1501101..1501140 (peaceful), 1501201..1501240 (solo classic), 1501301..1501340
(coin heaven), 1501401..1501440 (loot crate), with agent seed=world+1,000,000.
Every evaluation has an exact repeat. Next ten values per range are reserved
and unused. Training, development and reserved populations are separate;
the launcher audits collisions against all repository plans/registrations.

Totals: **100,000 training / 3,520 evaluation episodes**, including 1,760 primary
and 1,760 repeat observations. The reference uses 320; each trained arm 1,600.
Only final checkpoints after exactly 10,000 episodes are evaluated/selected.

All #109/#147 values are unchanged in [config.yaml](config.yaml): elimination
>=60%, improvement over frozen parent>=20 percentage points and positive 95%
lower bound; strict first place>=60% with positive improvement lower bound;
score margin>0 with positive improvement lower bound; peaceful self-kill<=10%;
invalid actions<1% per replica and aggregate on every suite. Collection-difference
lower bounds must exceed -2 points coin heaven/-3 points classic and loot crate;
survival lower>=-5 points, self-kill-increase upper<=2 points, and classic/loot
crate contrast candidate-.9*reference lower>=0. Every primary/repeat episode
must have p95<50 ms and max<100 ms, with verified raw provenance, immutable models,
exact repeats and resource integrity. Report absolute values and every gate.

Additional treatment criterion: Double-minus-control elimination>=10 percentage
points with 95% lower bound>0. Use 10,000 crossed replica/world-pair bootstrap
draws, seed 150, preserving matched replicas and common evaluation pairs.
Reference observations are shared, not replicated as independent models.
All gates are conjunctive. Only a passing treatment selects its median final
replica by primary peaceful elimination, ties by replica ID. Otherwise select
nothing; no control fallback or best-replica selection. Even a pass requires a
reviewed continuation binding; it does not bypass #137's existing helper.

## Machines and budgets

The **server** runs reference, control and Double plans serially, at most two
training workers within an arm, all evaluation serially in one recorded
environment. Caps: 24 CPU-hours / 10 wall-hours / 8 GiB process-tree memory; require
8 GiB free disk and 8 GiB available RAM. Confirm an exclusive 12-hour allocation,
including transfer/export/recovery margin. Keep campaign source/parent/seeds/
budget immutable; interruption uses the same retained authorization and resume.

#147 measured 6h 36m on the server. The extra online training forward pass adds
work; provisional estimate 7-9 hours plus up to 1 hour for export/transfer.
Short implementation timing cannot certify a full trajectory-duration estimate.
A cap breach is incomplete evidence, not permission to extend the run.

The **PC** can independently reproduce the bounded #146 diagnostic, including
preserved negative attempts, or work on #126's four-player adapter and #127
packaging. The **16-GB laptop** can independently verify the published #147/#146
evidence and run compatibility checks. Do not start three speculative training
campaigns simply to occupy three devices. This next comparison keeps both arms
and efficacy/latency evaluation on the server; the existing launcher does not
support moving its resume state across hosts. Task 4 training still needs a
registered available parent and its separate owner decision; PR149 is preparation.

See [DEVICES.md](DEVICES.md) for exact independent PC/laptop commands.

## Execution and evidence

[SERVER.md](SERVER.md) provides preparation, preflight, detached launch, progress,
resume, analysis and export commands. `training.task3_double_campaign` reuses
the existing resource monitor and raw analyzer; it also verifies the exact
Double DQN flag on every final checkpoint before accepting results.
The compact export preserves every failed attempt, all final/reference artifacts,
raw statistics/timings, episode CSVs and metadata/bindings/authorization/resources.
Duplicate staged source and verbose logs are omitted. Incomplete campaigns have a separately labeled `export-incomplete` path.
Retain originals until
review and publish the archive/manifest with any resulting scientific claim.

The owner authorized setup and short diagnostic tests. No long run was launched
by this preparation. Launch requires current-head CI, peer review or a separately
recorded owner execution exception, and explicit authorization of this concrete
budget. Neither this README nor an old campaign's authorization is permission
to start a different unreviewed run.

AI assistance: Codex prepared the tested target, protocol, tooling and commands.
Tests/smokes establish mechanics only. Human review is required.

## Preparation validation

The bounded 26-episode smoke verifies actual fresh-parent training in both modes,
all four evaluation suites, exact repeats, raw statistics/CSV consistency, hashes,
resource accounting and warning-only metadata. Its separate mechanics seeds are
not the scientific evaluation seeds. The production analyzer rejects the reduced
matrix; no scientific selection is reported. Compact verification/export is
tested separately with complete synthetic observations, including tampering,
failed attempts and refusal to overwrite an archive.

Executed smoke source: `1a959a2604b43d0962a4d4cb2272fd949f0c3228`.
Measured 152.36 s wall, 130.14 CPU s, peak process-tree RSS 296,054,784 bytes.
The first 26-game attempt completed raw checks but its harness incorrectly
requested a scientific decision from one replica. That rejected postprocessing
and all original outputs are preserved; the corrected harness expects rejection.
One additional frozen evaluation with normal logging matched every non-latency
episode field of the warning-only run. These checks establish mechanics only.

A fixed synthetic batch benchmark used three repeats of 500 updates per mode,
25 warmup updates, one CPU thread and alternating order. [timing.json](timing.json)
retains observations/environment: median standard 2.012 ms/update, Double 2.132 ms,
ratio 1.060. It excludes game simulation, startup, replay sampling and changing
policy trajectories, so it does not certify a full-run duration. The 7?9-hour
server estimate remains provisional; reserve the complete 12-hour allocation.
Reproduce with `python scripts/benchmark_issue150.py` in an isolated checkout;
no checkpoint is written. Run the integration check using
`python scripts/smoke_issue150.py --binding-dir <prepared-binding> --output-root <new-smoke-root>`.
The smoke cap is 15 minutes / 0.25 CPUh / 4 GiB, serial.
