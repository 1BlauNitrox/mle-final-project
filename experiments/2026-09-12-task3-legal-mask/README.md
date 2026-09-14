# Issue 147: consistent legal masking in exploratory Task 3

**Completed exploratory negative result; no checkpoint selected.**
See [RESULTS.md](RESULTS.md) for verified observations, every gate, durable
evidence and reproduction commands. Masking achieved 21% peaceful elimination
versus 18% control; the paired difference was +3 percentage points (95% CI
[-14.5, 20.5]), failing the registered benefit rule. Collection and crate
retention also failed. No automatic #137 continuation is permitted.
The owner explicitly authorized execution before peer review; that exception
does not imply reviewer approval. The prospective config remains unchanged
as the executed registration, including its historical authorization wording.
Owner Julius / 1BlauNitrox; reviewer requested: LiliWestermann. Refs #147,
#109 and roadmap #106/PR #145 at 32a2450 (not assumed merged).

The completed #109 experiment failed its peaceful and retention gates. Its
6.79% peaceful invalid actions and 10.15%/26.84%/15.92% retention invalid actions
motivate testing the already implemented framework-legal mask. This is a
testable mechanism, not an established cause of poor hunting or #146 loops.
#124 masking was not adopted; no Task 2 result is reclassified as successful.
This is Task 3 training against peaceful_agent, not standalone Task 2 tuning.

## Hypothesis and intervention

Consistent legal masking improves elimination compared with otherwise identical
unmasked Task 3 training, while meeting the existing hunting and retention
contracts. Control and treatment both freshly migrate #91 A/r3, SHA-256
`c0b6081acf5a733258005d46b9430f3ef50e403470a3bb5428661729f9aeedf7`,
2,382,903 bytes, source `cbd52be8392f5003a91c5600fda4efd544b48ec5`.
Use the original `binding-parent91/task2-parent.pt`, not any #109 trained replica.
Both online/target networks are independently preserved and opponent columns
zeroed. Replay, optimizer, epsilon and random streams start fresh. The treatment
changes only `action_masking=True` on a verified fresh migration; network and
other state equality is checked before execution. All six actions, 39 inputs,
gamma 0.9, active escape features, rewards and other hyperparameters stay fixed.

The existing mask filters currently occupied/blocked movement and unavailable
bombs during exploration, greedy acting and Bellman maximization. It neither
avoids dangerous legal moves nor adds an anti-loop policy. In moving-opponent
games it cannot promise zero invalid actions after simultaneous action ordering.

## Executed registration

- Five paired replicas per trained arm, 10,000 classic episodes each against
  exactly one unchanged peaceful_agent. Learner slot 0, opponent slot 1; seeded
  world placement. Identical world/agent training roots 147001..147005 and
  247001..247005; independent roots across replicas, shared across arms.
- Four development suites, each forty identical pairs per model: peaceful,
  opponent-free classic, coin heaven and loot crate. World seeds
  1471101..1471140, 1471201..1471240, 1471301..1471340, 1471401..1471440;
  agent seeds are world seeds +1,000,000. Each has an exact repeat.
  Next ten values per range are reserved and unused. Repository collision audit
  is mandatory; owner must confirm no unregistered remote seed use.
- Evaluate the frozen original #91 Task 2 parent once per pair/repeat under the
  same conditions. Do not reuse historical #109 evaluation observations.
- Total: **100,000 training, 3,520 evaluation episodes** (1,760 primary plus
  1,760 repeats). Final checkpoints only, no favorable early checkpoint.
- All original #109 numerical hunting, collection, survival, self-kill, crate,
  invalid-action and per-episode latency gates apply independently to each arm
  against that frozen parent. The exact unchanged values are in config.yaml.
- Treatment selection additionally requires masked-minus-control elimination
  >=0.10 and positive 95% lower bound. Use 10,000 crossed bootstrap draws, seed
  147, resampling paired replica indices and common world/agent indices. All
  benefit/guardrail tests are conjunctive; no multiple-endpoint fishing.
- Only if all treatment conditions pass, select its median final replica by
  peaceful elimination, ties by replica ID. Otherwise select nothing. A control
  pass alone does not trigger fallback selection. Zero invalid actions alone
  never qualifies. No automatic #137 binding: review a new result/parent handoff.

## Allocation and estimate

Run reference, control and masked plans serially on the **same Linux server**;
at most two parallel training workers within an arm. All evaluations are serial
on that same recorded environment, without competing experiments. PC performs
independent implementation/diagnosis; laptop can verify transferred compact
evidence. Do not split control/treatment by host or duplicate evaluation merely
to occupy the laptop.

Registered shared cap: **24 CPU-hours, 10 elapsed wall-hours, 8 GiB process-tree
memory**, with at least one additional hour for analysis/export/transfer inside
a confirmed 12-hour allocation. #109 measured 3h19m for one arm plus reference;
twice that is about 6h38m. Plan **7-9 hours for games**, with uncertain masked
trajectory lengths and host load, then up to one hour for analysis/transfer.
This is an estimate, not a completion guarantee. The hard cap produces an
incomplete result if exceeded; never extend it or alter a running campaign.
Require 20 GiB free disk, 8 GiB available RAM and an exclusive allocation.

## Commands and evidence

See [SERVER.md](SERVER.md). `training.task3_mask_campaign` provides prepare,
dry-run, guarded run/resume, raw verification/analysis, compact verification and
export. The shared tested launcher preserves authorization, source/dependency
fingerprints, jobs, failed attempts and cumulative resources. Raw analysis must
use the original source/environment or the historical verifier documented in
RESULTS.md, which checks executed Git bytes and dependencies separately from
the analysis environment. Compact verification also requires no games.

Export retains raw statistics, per-episode CSVs, metadata/status, bindings,
checkpoints, full analysis and SHA-256/size manifests, while omitting duplicated
staged source directories and verbose logs. Keep original outputs until review.
Publish the archive and manifest durably with the result PR. Analysis integrity
failure is distinct from scientific gate failure; neither permits selection.

No new agent defaults, model family, reward objective or held-out seeds change.
AI assistance: Codex prepared plans, guarded tooling, tests, result verification
and documentation. Human review of the result remains required before merge.
