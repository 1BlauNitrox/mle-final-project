# Issue #137: exploratory coin-collector continuation

Prospective preparation only, stacked on #109 / PR #120. No scientific run or
result exists. Owner: 1BlauNitrox. Non-author review requested from
LiliWestermann; review, resources and execution authorization remain pending.
#51 is unchanged: this study cannot satisfy its validated-Task-2 prerequisites.

## Hypothesis and controls

Continuing the mechanically selected peaceful-stage Task 3 DQN against exactly
one unchanged coin_collector_agent improves attributable elimination and paired
match outcome while retaining Tasks 1/2. Compare five fresh training replicas
against that one frozen peaceful predecessor. Both arms use the same 40
world/agent pairs, fixed command slots (learner 0, opponent 1), classic scenario,
and primary/repeat measurements. The parent already has opponent awareness;
the main variable is additional training against the harder opponent. Keep the
39-input schema, six actions, rewards, hidden sizes, inherited hyperparameters,
mask/escape modes and source/dependencies fixed. Reset optimizer, replay,
exploration, RNG and counters; preserve online and target networks independently.

## Prerequisite and parent selection

Preparation requires a compact verified #109 result with every peaceful gate
passing and its mechanically selected final 10,000-episode training checkpoint.
The helper verifies the selected hash and completed-episode count and copies
the prerequisite evidence into the binding. It refuses another replica,
evaluation-only artifacts, non-passing results and existing output directories.
The launcher revalidates the prerequisite and lineage before execution.
Peer review of that result is also required before compute authorization.

If peaceful gates fail, stop; do not run this plan merely because it parses.
No Task 2 success is inferred from either stage. A new predecessor or changed
objective requires a new prospective registration and separate output binding.

## Seeds, rounds and resources

Training world seeds: 109011..109015; agent seeds: 209011..209015. These were
already reserved by the existing coin-collector template. Train each replica
for exactly 10,000 classic episodes, then use its final checkpoint only.

| Suite | World seeds | Agent seeds | Opponents |
| --- | --- | --- | --- |
| classic-coincollector | 1092101..1092140 | 2092101..2092140 | coin_collector_agent |
| classic-retention | 1092201..1092240 | 2092201..2092240 | none |
| coin-heaven-retention | 1092301..1092340 | 2092301..2092340 | none |
| loot-crate-retention | 1092401..1092440 | 2092401..2092440 | none |

One episode per pair and an exact repeat. The next ten values after each range
are reserved confirmation populations; all final seeds remain unopened. The
repository seed audit runs before execution; the owner must also confirm no
unregistered use on other machines.

Budget: 50,000 training and 1,920 evaluation episodes (1,600 candidate, 320
reference, including repeats). Registered ceiling: 24 CPU-hours, 15 wall-hours,
8 GiB aggregate framework-process RAM, two training workers and serial evaluation.
Reference and candidate plans share one monitor and execute sequentially.
This is a separate ceiling and authorization, not a reuse of #109 or Task 2's
budget. Dedicated checkout, bindings, outputs and resource records are required.

## Metrics and decision

Use the common [Task 3 metric and uncertainty definitions](../2026-09-08-dqn-task3-peaceful-opponent/README.md):
native attributable kills, strict first place with separate ties, paired score
margin, coins/collection fraction, crates, bombs, survival/steps, self-kills,
invalid actions, action counts, full decision latency and learning diagnostics.
Every candidate/reference observation and deterministic repeat is retained.

The explicit exploratory capability targets use #51's published levels:
elimination >=0.40, paired gain >=0.20 with 95% lower bound >0, first-place
>=0.55, positive mean opponent score margin, self-kills <=0.10 and invalid actions
<0.01 overall and per replica. Additionally require positive paired score-margin
and first-place improvement over the frozen peaceful predecessor (95% lower
bounds >0); do not claim improvement merely from an absolute rate.

Retention keeps the registered #109 guards: collection lower bounds >-0.02
for Task 1 and >-0.03 for each Task 2 suite; survival lower bound >=-0.05;
self-kill increase upper bound <=0.02; Task 2 crate contrast
`candidate - 0.9 * parent` lower bound >=0. Runtime requires every episode
p95 <50 ms, max <100 ms, one-thread evaluation, memory <8 GiB, byte-immutable
artifacts and exact primary/repeat outcomes/actions. Report absolute earlier-task
performance as well as retention. All gates are conjunctive.

Use 10,000 crossed replica/seed-pair bootstrap samples, analysis seed 137,
with the single reference shared across sampled candidates. On a complete pass,
select the median candidate by primary coin-collector elimination rate, ties by
replica ID, for review of the next stage. Otherwise preserve the mixed/negative
result and stop. No automatic Task 4 training, freeze or Task 2 completion.

## Commands

Before any scientific work, use a peer-reviewed clean checkout and separately
authorize compute. The default protocol remains peaceful; select this protocol
explicitly. A template dry-run needs no peaceful result and starts no game:

```bash
python -m training.run_task3_campaign --protocol coincollector --dry-run
```

After a passing, reviewed peaceful result, provide its selected **training**
checkpoint (the helper verifies the selection; do not choose a favorable replica):

```bash
python scripts/prepare_task3_continuation.py \
  --parent PATH_TO_SELECTED_FINAL_TRAINING_CHECKPOINT \
  --peaceful-evidence /srv/task3-analysis/issue109-exploratory \
  --output-dir /srv/task3-bindings/issue137-exploratory

python -m training.run_task3_campaign --protocol coincollector --dry-run \
  --binding-dir /srv/task3-bindings/issue137-exploratory

python -m training.run_task3_campaign --protocol coincollector \
  --binding-dir /srv/task3-bindings/issue137-exploratory \
  --output-root /srv/task3-runs/issue137-exploratory \
  --reviewed-commit FULL_PEER_REVIEWED_COMMIT_SHA \
  --authorized-by HUMAN_COMPUTE_AUTHORIZER \
  --hardware-description ACTUAL_SERVER_CPU_AND_RAM \
  --available-memory-gib 8 --authorize-compute
```

Resume with the identical command plus `--resume`. Never share a checkout with
another active campaign or reuse Task 2 records. CPU and wall usage are retained.

```bash
python -m training.analyze_task3_campaign --protocol coincollector \
  --campaign-root /srv/task3-runs/issue137-exploratory \
  --binding-dir /srv/task3-bindings/issue137-exploratory \
  --output /srv/task3-analysis/issue137-exploratory

python -m training.analyze_task3_campaign --protocol coincollector \
  --verify-evidence /srv/task3-analysis/issue137-exploratory
```

Preserve failures and publish claim-checkable evidence durably using the
repository's hash/size/retrieval contract before reporting a scientific result.
Synthetic tests verify preparation and rejection paths; they are not results.
