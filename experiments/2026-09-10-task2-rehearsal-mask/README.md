# Escape-enabled Task 2 rehearsal and masking preparation

Refs #124; prerequisite #123. Prospective executable protocol, awaiting
non-author review. The user explicitly overrode the pre-training review gate
on September 10 ("I want you to run it anyways!"). This execution-only
exception does not approve or merge either PR. The user confirmed four arms, four workers, 8 GiB RAM and
ten wall hours for local Windows execution. Execution started and stopped incomplete on September 10; see [SECOND-INTERRUPTION.md](SECOND-INTERRUPTION.md) for the current state and [INTERRUPTION.md](INTERRUPTION.md) for the first stop.

| Arm | Task 1 practice | Legal-action masking |
| --- | --- | --- |
| A | blocked | none |
| B | interleaved | none |
| C | blocked | framework_legal |
| D | interleaved | framework_legal |

All arms use escape-continuation inputs and uniform replay. Rewards, network,
optimizer, epsilon schedule and source weights are fixed. This does not adopt
masking: #86 rejected it; the combination with active escape inputs is a new
hypothesis. #107's negative protected-replay result remains unchanged.

## Registered matrix

Each arm has five paired replicas and twenty 500-episode blocks: 2,000
coin-heaven, 2,000 loot-crate and 6,000 classic episodes per replica. Blocked
arms run four coin blocks, four loot blocks and twelve classic blocks.
Interleaved arms use coin blocks at positions 1, 6, 11 and 16, loot blocks at
positions 2-5, and classic for the other twelve. Every arm has identical process
boundary counts. World-seed offsets pair the same scenario occurrence across
arms; they do not replay the same initial seed at every later practice block.
The intervention changes experience order, including the exploration level at
which a scenario is encountered; it does not claim identical trajectories.

Training root pairs are 124001-124005 / 224001-224005. The fresh development
world/agent ranges are 324001-324040 / 424001-424040 (classic),
325001-325040 / 425001-425040 (coin-heaven), and
326001-326040 / 426001-426040 (loot-crate). Repeats reuse these pairs and are
not independent observations. Frozen Task 1 and untrained migration references
receive matching suites. The untrained reference retains its pinned disabled
escape setting; its five added input weights are zero. All four trained arms
enable escape inputs. Final and confirmation seeds stay unopened.

Total budget: 200,000 training and 5,120 evaluation episodes, 400 training
stage jobs. Select only the final-budget checkpoint. The new control is a
fresh matched run, not a claim that the reblocked curriculum equals historical
#107 execution. The source is the committed fresh 26-input migration,
SHA-256 `4ad409472e7ca008dfcc82aa26017017c90259b65f9e593020d1b63921430f60`.

## Prospective decision

The four primary contrasts are rehearsal B-A and D-C on coin-heaven collection,
and masking C-A and D-B on classic collection. Each requires at least +0.10
fractional improvement and a strictly positive 98.75% bootstrap lower bound.
The 10-point threshold keeps the previous practical collection-effect scale;
it is a prospective choice, not evidence that this effect will occur. Use the
same hierarchical paired bootstrap as #107, 10,000 resamples, and a four-test
Bonferroni efficacy family. Report 95% intervals descriptively as well.

B/C additionally require all collection and survival non-regression intervals
against A to have 95% lower bounds strictly above -0.05. D requires both
conditional efficacies and those guards against A, B and C. Zero invalid
actions alone never makes masking successful. Report D-B-C+A interactions
descriptively; they do not override the decision rule.

Every absolute Task 1, Task 2 and latency/determinism gate from #107 remains
unchanged. Rank eligible treatments by overall pass, Task 2 pass, Task 1 pass,
classic collection, lower classic self-kills, coin-heaven collection and cell
ID. If none is eligible, choose A. Choose the third of five replicas ordered
by classic collection and replica ID. Failing absolute gates always means an
exploratory Task 3 predecessor. Incomplete evidence has no scientific selection.

The new analyzer reuses #107's statistical primitives and absolute gates, not
its factor-specific interpretation. It verifies registered plan bytes,
metadata, episode counts, artifacts, repeats and resource bounds before
returning a result. It retains per-file evidence hashes and failed attempts.

## Local execution

The original campaign authorization, including evaluations/analysis, had a ten-hour wall limit,
a derived 40 CPU-hour ceiling and 8 GiB aggregate process RAM. The documented runtime amendment extended time to 24 wall hours / 96 CPU-hours; all memory safeguards remained active. One worker per
arm trains concurrently (four total); all training completes before serial
evaluation. The monitor includes supervisor memory and CPU. It samples every
second and stops this campaign's workers on breach. At least 4 GiB free RAM is
required at startup; dropping below 1 GiB system availability also stops work.
Resource accounting and the original wall-clock deadline survive resume.
This is not a guarantee that 200,000 episodes finish within ten hours.

Use the exact approved PR #131 head in a clean, dedicated worktree. The
launcher checks GitHub non-author approval and the exact SHA, so neither these
commands nor the ordinary compute authorization bypass review. For the explicit
September 10 owner exception only, pass `-OwnerAuthorizedReviewException` to
the PowerShell launcher (Python: `--owner-authorized-review-exception`). The
campaign records the exception in authorization and every job's metadata;
`reviewed_commit` remains the legacy execution-SHA field and is not a claim
of peer approval when the exception is present. All resource and evidence
checks remain active. Install the
documented requirements in a separate environment. Dry-run:

```powershell
python -m training.run_issue124_campaign --dry-run
```

Detached launch from that worktree (replace the two absolute paths):

```powershell
.\scripts\start_issue124.ps1 -Python C:\path\to\.venv\Scripts\python.exe -OutputRoot C:\path\to\issue124-output
Get-Content C:\path\to\issue124-output\campaign-status.json
Get-Content C:\path\to\issue124-output\resources.json
```

The hidden supervisor remains independent of the invoking terminal. Timestamped
stdout/stderr and launcher PID records live in the output directory. Keep the
machine powered and awake; no startup command changes Windows power settings.
For a technically interrupted run, use the same command with `-Resume` after
checking that its previous workers have stopped. A breached ceiling cannot
be reset through resume. Do not edit the running worktree or its environment.

The supervisor writes `completed` only after analysis; failures/limits write
`stopped_incomplete`. A complete run can still have `analysis_valid: false` or
fail every Task 2 gate. Reproduce analysis without playing games:

```powershell
python -m training.analyze_issue124_campaign --campaign-root C:\path\to\issue124-output
```

Keep all outputs until the result is reviewed and required evidence is
published with retrievable bytes. Do not use the generic per-plan launcher to
bypass aggregate limits. No agent default or submitted checkpoint is replaced.

## Deadline priorities

Finish this bounded Task 2 attempt and select mechanically; do not chase an
unlimited sequence of sweeps. Prepare #125's escape-preserving Task 3 migration
concurrently, then complete existing #109/#120 and progress to #51. Target
Task 3 work on September 10-12, Task 4 (#126) on September 13-16, optional
compatibility on September 17, candidate freeze by September 19, and final
verification/submission (#127) before September 21 at 21:00 Europe/Berlin.
These are targets, not a promise of completed learning milestones.
