# Issue #109 peaceful-opponent Task 3 comparison

See [PREPARATION.md](PREPARATION.md) for the current parent-migration path,
matched retention coverage and the explicit scientific launch gates.

> Status: **Registered protocol, not ready for scientific execution.** Run
> plans dry-run cleanly and are ready to launch mechanically, but issue
> #109's own acceptance criteria require an assigned owner/reviewer, fixed
> numeric decision criteria, and explicit compute authorization before any
> run counts as more than infrastructure validation. Both starting
> artifacts are provisional pending Issue #107's development-predecessor
> selection (#108's own stated dependency).

## Hypothesis

Does opponent-aware training (`DagobertDuckDQNTask3`, issue #108) improve
`classic` peaceful-opponent elimination and paired score relative to its
opponent-blind Task 2 predecessor, under identical seeds and opponent
slots, while retaining Task 1 navigation and Task 2 collection/survival
behavior?

## Comparison

Two arms, paired by evaluation seed (not by replica -- see "Why one frozen
reference, not five" below):

- **Task 3** (`issue109-task3-vs-peaceful.yaml`): 5 replicas, each trained
  10,000 episodes of `classic` against exactly one `peaceful_agent`, from
  the same provisional Task 3 migration checkpoint (0 completed episodes).
- **Task 2 predecessor** (`issue109-task2-predecessor-vs-peaceful.yaml`):
  the frozen provisional Task 2 artifact
  (`checkpoint-issue85-zero-suffix.pt`, the same artifact Task 3 was
  migrated from), evaluated as-is -- no training stage -- against the same
  `classic` + `peaceful_agent` seeds. This agent has no opponent features
  at all, so it is a genuine opponent-blind baseline, not a weaker version
  of Task 3.

Both arms are evaluated on the identical 10 development seed pairs
(`world_seed` 109101-109110, `agent_seed` 209101-209110) for `classic` +
`peaceful_agent` (primary + repeat, for the determinism check), so Task 3's
5 replicas can each be paired against the same single Task 2 reference
point.

Both plans additionally evaluate the standard opponent-free
retention battery (`classic`, `coin-heaven`, `loot-crate`, reusing the same
development seed pairs established in #86/#97/#103 for cross-experiment
comparability), per issue #109's explicit requirement to report absolute
Task 1/2 performance alongside retention, not just the peaceful-opponent
result.

### Why one frozen reference, not five paired replicas

Unlike #86/#97/#103 (which pair freshly-trained replicas against each
other under matching seeds), the Task 2 side here is a single frozen,
already-evaluated artifact -- there is nothing to retrain five independent
copies of. This mirrors Issue #107's own `frozen_task1`/`untrained` plans
(one replica each, compared against multiple trained replicas), not its
paired treatment-vs-control cells.

## Transition to `coin_collector_agent`

`issue109-task3-vs-coincollector.yaml` is prepared with the same structure
(fresh seeds, same retention battery) but **must not be executed** until a
peaceful-stage decision is made, per issue #109's own explicit gating. It
is registered now so the team is not blocked waiting on it once that
decision is reached.

## What this registration does NOT do (per issue #109's own requirements)

- **Does not fix numeric decision criteria.** Issue #109 explicitly
  requires the owner to set elimination, paired-score/first-place,
  self-kill, and retention thresholds before scientific execution, and
  explicitly warns against silently reusing #51's coin-collector
  thresholds as peaceful-agent criteria. None are proposed here.
- **Does not assign an owner or non-author reviewer.**
- **Does not authorize compute or fix a resource ceiling.** A *proposed*
  ceiling (scaled from #97's comparable single-arm estimate) is recorded
  in `config.yaml` for the owner to accept, adjust, or replace.
- **Does not bind a final parent artifact.** Both starting checkpoints are
  the same provisional pair used throughout #108; #107's eventual
  development-predecessor selection must be bound here (new checksums,
  re-run `--dry-run` on both plans) before this stops being a provisional
  fixture.

## Execution (once the above is resolved)

```bash
python -m training.run_plan training/run_plans/issue109-task3-vs-peaceful.yaml --dry-run
python -m training.run_plan training/run_plans/issue109-task2-predecessor-vs-peaceful.yaml --dry-run
```

Then, once owner/reviewer/criteria/ceiling/authorization are all in place:

```bash
tmux new -s issue109-task3-peaceful
python -m training.run_plan training/run_plans/issue109-task3-vs-peaceful.yaml 2>&1 | tee logs/issue109-task3-peaceful.log
```

```bash
tmux new -s issue109-task2-predecessor
python -m training.run_plan training/run_plans/issue109-task2-predecessor-vs-peaceful.yaml 2>&1 | tee logs/issue109-task2-predecessor.log
```

(`--evaluation-only` is a different mechanism -- reusing an artifact
already produced by a separate plan's training stage -- and is not needed
here: this plan already has no training stages, so `run_plan.py` runs only
its evaluation jobs by default, exactly like #107's `frozen_task1` plan.)

The two can run concurrently (independent plan directories, no shared
state); `issue109-task3-vs-coincollector.yaml` must wait for the peaceful
stage's decision regardless of available compute.

5 replicas × (1 training stage + 8 evaluation suites × 10 seed pairs) = 405
jobs, 50,000 training episodes, for the Task 3 arm. 1 replica × 8
evaluation suites × 10 seed pairs = 80 jobs for the frozen Task 2
predecessor. Proposed ceiling: 24 CPU-hours / 15 wall-hours / 8 GiB / two
training workers (owner must accept or replace, per issue #109).
