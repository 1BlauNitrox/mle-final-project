# Task 1 DagobertDuckDQN development baseline

## Status

This experiment is preregistered but has not started. No registered training or
development-evaluation seed has been consumed.

Training is blocked until the non-author reviewer is assigned and the complete
Task 1 tabular baseline from issue #36 / PR #37 is accepted and available. As
of 2026-09-01, PR #37 is open, requires review, and its five reported model
checksums are not backed by five locally available artifacts in this worktree.

## Metadata

- Issue: [#41](https://github.com/1BlauNitrox/mle-final-project/issues/41)
- DQN implementation: PR #38, merge commit
  `0f315f323f4363ead44ab6893ea34dce47d72f25`
- Observation and launcher implementation:
  `cc3dc32979a1d25a0d33ad5e65c5a64783f009cd`
- Tabular comparison: issue #36 / PR #37, branch commit
  `a4bbca53c709ea93d71c6a69e0de1c9186532202`
- Branch: `experiment/41-dqn-task1-development-baseline`
- Agent: `DagobertDuckDQN`
- Owner: Waffelmanufaktur
- Reviewer: not yet assigned
- Date: 2026-09-01

## Hypothesis

With the fixed Task 1 implementation from PR #38 and the same 10,000-episode
interaction budget used for the tabular baseline, `DagobertDuckDQN` will learn
reproducible visible-coin navigation while remaining non-inferior to
`DerKleineVermoegensumverteiler`.

Success requires all thresholds registered in issue #41 and `config.yaml`. A
negative or mixed result remains valid and will not trigger exclusions or
post-hoc changes in this experiment.

## Setup

The machine-readable configuration is in `config.yaml`. The principal fixed
conditions are:

- five serial, independent training runs of exactly 10,000 episodes;
- `coin-heaven`, no opponents, final-checkpoint selection;
- training world seeds `12001` through `12005`;
- agent seeds `22001` through `22005`;
- development world seeds `31001` through `31040`;
- reserved seeds `31041` through `31050` remain unused;
- no final held-out Task 1 seed is inspected;
- one PyTorch CPU thread and no evaluation multiprocessing.

The agent-source fingerprint after adding observation-only diagnostics is:

```text
56938f004403b056aef6df07079e0f6d94f0e0c7093ae693a3cad5c131e2da4e
```

The diagnostics add no reward, feature, action, update, replay, optimizer, or
checkpoint-selection change. They expose loss, replay size, optimizer-update
count, and target-network synchronization counts already produced by the fixed
DQN training process.

## Seed-collision check

On 2026-09-01, the registered training and development seeds were checked
against tracked repository text and every local
`training_outputs/**/metadata.json`. No prior scientific or smoke run used a
registered issue #41 training world seed or agent seed, and none used a
development seed from `31001` through `31040`.

The launcher repeats the training-seed metadata check immediately before the
first run and refuses to start on a collision, dirty worktree, source-hash
mismatch, or pre-existing DQN checkpoint.

## Overnight training launcher

After every blocker above is resolved and the preregistration commit is clean,
run from the repository root:

```powershell
.venv\Scripts\python.exe -m training.run_dqn_task1_baseline
```

The launcher runs serially to avoid five CPU-heavy DQN jobs contending with one
another. Each run starts without `checkpoint.pt`. Its mechanically selected
final checkpoint is moved into the ignored series artifact directory, hashed,
and recorded in `series.json` before the next run begins. Failed runs and any
partial checkpoint are retained; later registered runs continue without
silently retrying or replacing the failure.

Raw outputs are written below:

```text
training_outputs/issue-41-dqn-task1-baseline/<series timestamp>/
```

They remain outside Git. The compact results, figures, checksums, producing
commit, interpretation, and next decision will be added here after training and
the registered evaluation.

## Results

Not available. Training has not started.

## Decision and follow-up

Do not launch the scientific runs until the listed blockers are resolved. Once
they are resolved, use the committed launcher without changing the registered
features, rewards, hyperparameters, seeds, metrics, thresholds, or checkpoint
selection rule.
