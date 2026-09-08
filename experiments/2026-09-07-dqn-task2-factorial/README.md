# DQN Task 2 escape and protected-replay factorial

> Status: prospective protocol only. No scientific training, evaluation result,
> performance conclusion, or compute authorization is part of this change.

## Metadata

- Issue: #107
- Roadmap: #106
- Agent: `DagobertDuckDQNTask2`
- Protocol base: `f4eddf9ea411eef0975e811bde5a353011f5144e`
- Owner: Waffelmanufaktur
- Required non-author reviewer: 1BlauNitrox
- Execution revision: the full SHA of the clean, reviewed campaign branch
- Framework revision: the repository framework fingerprint recorded by each plan

The owner and reviewer names identify the required roles. They do not record
review approval or teammate availability. The launcher requires the reviewed
commit and an explicit compute-authorization flag at execution time.

## Research questions and hypotheses

The Issue #46 DQN result showed a high classic self-kill rate and severe Task 1
forgetting. The four-cell experiment tests two prospective explanations:

1. Multi-step escape-continuation inputs reduce classic self-kills.
2. Protected Task 1 replay improves coin-heaven retention.

Their combination may let retained navigation convert better survival into
collection. The registered cells are:

| Cell | Escape continuations | Replay |
| --- | --- | --- |
| A | off | uniform |
| B | on | uniform |
| C | off | protected Task 1 |
| D | on | protected Task 1 |

The planned contrasts are B-A and D-C for escape, C-A and D-B for replay, and
D-B-C+A for the interaction. Replicas and world/agent seeds are paired across
all cells. Repeat episodes verify determinism and are never treated as
independent observations.

## Starting artifacts and controls

All cells start from the committed 26-feature checkpoint, SHA-256
`4ad409472e7ca008dfcc82aa26017017c90259b65f9e593020d1b63921430f60`.
This is the Issue #87 schema migration of the corrected Issue #85 artifact,
SHA-256
`3edb2e7196030fcb52af6c7dc9ee69d9fc1259898ea674002fe06fbe93468015`:
the corrected 21 columns are copied and the five continuation columns are zero.
This two-step lineage resolves the older Issue #107 wording, which names the
21-feature artifact directly even though it cannot be resumed by the merged
26-feature trainer.

The frozen Task 1 reference is SHA-256
`eb08e3f67b620ac2a253a2af4db3d5b4c6ea9e667a2aaf1d91e3fccf4ba8b05e`.
Action masking remains `none`. Rewards, architecture, optimizer, gamma,
epsilon schedule, replay capacity, update rule, warmup, target synchronization,
and the 2,000/2,000/6,000 curriculum are fixed in `config.yaml`.

## Training protocol

Each cell uses five replicas with the same paired seed table:

| Replica | World seed | Agent seed |
| ---: | ---: | ---: |
| r1 | 107001 | 207001 |
| r2 | 107002 | 207002 |
| r3 | 107003 | 207003 |
| r4 | 107004 | 207004 |
| r5 | 107005 | 207005 |

Each replica trains for exactly 10,000 episodes: 2,000 `coin-heaven`, 2,000
`loot-crate`, then 6,000 opponent-free `classic`. The episode-10,000 checkpoint
is selected mechanically. The campaign totals 20 replicas and 200,000 training
episodes. Failed and interrupted attempts remain in the run-plan directory;
resume continues the immutable plan rather than replacing evidence.

## Evaluation protocol

Each final checkpoint receives one primary and one deterministic-repeat episode
on 40 paired world/agent seeds for each of `classic`, `coin-heaven`, and
`loot-crate`. The untrained migration is evaluated on the same three suites;
the frozen Task 1 agent is evaluated on the coin-heaven suite. This totals 2,560
primary and 2,560 repeat episodes.

The exact development lists are `307001-307040`, `308001-308040`, and
`309001-309040`, with corresponding agent lists `407001-407040`,
`408001-408040`, and `409001-409040`. The ten following values in each range are
reserved for confirmation and remain unopened. The committed preflight scans
existing run plans and experiment configurations for collisions.

## Metrics, contrasts, and multiplicity

The analyzer reports the complete Issue #46 collection, crate, bomb, survival,
self-kill, invalid-action, action-distribution, efficiency, determinism, and
latency metrics for every cell and replica. Training job durations, episode
counts, runtime versions, and producing commits are retained separately.

All planned contrasts receive hierarchical paired 95% bootstrap intervals with
10,000 resamples. The four efficacy decisions use Bonferroni-adjusted 98.75%
intervals as a family; the 95% intervals remain the primary reported uncertainty.

- Escape efficacy requires a classic self-kill reduction of at least `0.15`
  and an adjusted lower confidence bound above zero.
- Replay efficacy requires a coin-heaven collection improvement of at least
  `0.10` and an adjusted lower confidence bound above zero.
- Every treatment contrast must keep the lower 95% bound above `-0.05` for
  collection and survival in all three suites.
- Every cell is separately checked against all Task 2, Task 1 retention, and
  compatibility gates inherited from Issue #46. Relative eligibility and
  absolute Task 2 completion are reported separately.

## Prospective selection rule

B is eligible only through B-A escape efficacy and guards. C is eligible only
through C-A replay efficacy and guards. D is eligible only if D-C passes escape,
D-B passes replay, and both contrast guard sets pass. If no treatment is
eligible, select control A.

If several treatments are eligible, choose lexicographically by: overall
absolute pass, Task 2 pass, Task 1 pass, higher classic collection, lower
classic self-kill rate, higher coin-heaven collection, then alphabetical cell
ID. Within the selected cell, sort the five replicas by primary classic mean
collection and replica ID and select the median (third) artifact. The best
individual seed is never selected.

If the selected cell fails the absolute gates, it may be handed to Task 3 only
as an explicitly exploratory predecessor. It is not evidence that Task 2 is
complete or that the agent is submission-ready.

## Server preflight and execution

From the clean reviewed commit, validate the complete campaign without writing
outputs:

```bash
python -m training.run_issue107_campaign --dry-run
python -m training.analyze_issue107_task2_factorial --validate-protocol
```

Scientific execution requires explicit human authorization and an actual
hardware record. The launcher refuses a dirty checkout, a non-reviewed commit,
missing authorization, fewer than 8 GiB available RAM, or a changed plan:

```bash
python -m training.run_issue107_campaign \
  --authorize-compute \
  --reviewed-commit <full-reviewed-commit-sha> \
  --authorized-by '<human authorizer>' \
  --hardware-description '<CPU, host/server, OS>' \
  --available-memory-gib <available-GiB>
```

Resume the exact retained campaign with the same arguments plus `--resume`.
Do not delete or rename the existing `training_outputs/run-plans/issue107-*`
directories. After all six plans complete:

```bash
python -m training.analyze_issue107_task2_factorial
```

The fixed ceiling is 48 CPU-hours, 24 wall-clock hours, four training workers,
8 GiB aggregate RAM, and one serial evaluation worker. No paid resource is
authorized. The launcher samples every active Bomberman process tree, persists
campaign-wide CPU, wall-clock, and peak-memory use beside the authorization
record, and includes completed jobs when resuming. Crossing any ceiling
terminates all active campaign process trees and retains their failed-attempt
snapshots plus the resource record. Do not extend the budget, delete the
partial record, or remove weak replicas. This training-only guard requires
`psutil`, declared in `requirements-dev.txt`.

The analyzer accepts evidence only when every recorded resolved plan exactly
matches the plan loaded from this reviewed execution revision, including jobs,
seeds, source/dependency/agent fingerprints, and parent-artifact hashes.

## Evidence and follow-up

Raw plan outputs, checkpoints, episode rows, decision times, and authorization
records remain below ignored `training_outputs/`. A result PR must publish the
smallest claim-checkable evidence package and give any required large archive a
durable locator, SHA-256, byte size, schema description, retrieval steps, and
verification command. This protocol PR uses `Refs #107`; it does not close the
experiment.
