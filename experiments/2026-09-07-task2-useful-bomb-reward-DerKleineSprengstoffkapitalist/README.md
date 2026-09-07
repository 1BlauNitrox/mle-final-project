# Tabular Task 2 immediate useful-bomb reward

> Status: completed_negative

## Metadata

- Issue: #114
- Agent: `DerKleineSprengstoffkapitalist`
- Owner: LiliWestermann
- Date: 2026-09-07
- Registration base: `933a8fe`
- Experiment commit: `eb215172b16146772dd543ae90813f852ad041e4`
- Framework revision: `0f55c1d`
- Execution authorization: user-directed; no separate pre-training peer approval

The requester explicitly authorized immediate execution without waiting for the
usual pre-training peer approval. This is a documented process exception, not a
claim that independent review occurred.

## Research question

Does an immediate reward for placing a bomb that can destroy at least one crate
improve Task 2 credit assignment and hidden-coin collection?

## Hypothesis

Adding `USEFUL_BOMB_PLACED: +1.0` will increase useful bomb placement, crate
destruction, and `classic` coin collection without materially increasing
self-kills.

## Baseline and treatment

- Control: unchanged unmasked Issue #102 configuration with a useful-bomb
  reward of `0.0`.
- Candidate: the same configuration with a useful-bomb reward of `+1.0`.

Both arms use the same untrained Task 2 artifact, SHA-256
`8f2e618bfb38d690b565be1d3034f153d120887a36d90a61f8adcc1a765c1bbb`.
No action masking is used.

## Controlled variables

Features, all other rewards, hyperparameters, parent artifact, curriculum,
training budget, seeds, scenarios, opponents, checkpoint selection, and
evaluation suites are identical between arms.

## Training protocol

Train five replicas per arm using the Issue #102 seeds. Each replica receives
2,000 `coin-heaven`, 2,000 `loot-crate`, and 6,000 `classic` episodes. Only the
final checkpoint after 10,000 episodes is evaluated.

## Evaluation protocol

Evaluate each replica on 40 paired development seeds in `classic`,
`coin-heaven`, and `loot-crate`, followed by an identical deterministic repeat.
Evaluate the frozen Task 1 artifact on the paired `coin-heaven` seeds.
Confirmation seeds remain unused.

## Metrics and decision rule

Primary metric: candidate-minus-control hidden-coin collection fraction on
`classic`.

The candidate passes only if:

- the primary mean difference is above zero;
- the paired 95% CI lower bound is above zero;
- at least four of five replicas improve;
- mean useful-bomb placements per training episode increase;
- mean crates destroyed per `classic` evaluation episode increase;
- aggregate `classic` self-kill rate is no more than control plus `0.02`;
- Task 1 retention CI is above `-0.05`;
- deterministic repeats and the registered runtime limits pass.

## Execution

```bash
python -m training.run_plan training/run_plans/issue114-tabular-task1-frozen.yaml --dry-run
python -m training.run_plan training/run_plans/issue114-tabular-task2-control.yaml --dry-run
python -m training.run_plan training/run_plans/issue114-tabular-task2-useful-bomb-reward.yaml --dry-run
```

No result was inspected before this protocol and its decision rule were written.

## Results

All registered original plans completed: 80/80 Task 1 jobs and 1,215/1,215
jobs in each Task 2 arm. Primary/repeat action hashes matched for every
evaluation episode. The candidate-minus-control `classic` collection effect was
`+0.00333` (paired 95% bootstrap CI `[-0.00167, +0.00944]`), and only three of
five replicas improved. The primary effect is therefore not distinguishable
from zero under the registered rule.

The treatment changed bomb behavior strongly: mean useful bombs per training
episode increased from `1.34866` to `1.72198`, and mean crates destroyed per
`classic` evaluation episode increased from `1.175` to `7.070` (paired
difference `+5.895`, 95% CI `[+5.110, +6.695]`). This did not translate into a
reliable collection gain. Instead, the aggregate `classic` self-kill rate rose
from `0.145` to `0.595`, far beyond the allowed `+0.02` margin.

Task 1 retention passed with a candidate-minus-frozen effect of `0.0` and CI
`[0.0, 0.0]`. Runtime limits passed, and confirmation seeds remained unused.

## Diagnostic recorder correction

The agent returned `useful_bombs` during the original training, but the generic
episode CSV schema initially discarded that field. After discovering this only
during final analysis, the recorder was corrected and training-only diagnostic
plans replayed the exact registered seeds and budgets. All ten final replay
model SHA-256 hashes matched their corresponding original models exactly, so
the replay restored the missing observation without changing the learned
policies or replacing the original evaluation evidence.

## Decision and interpretation

Decision: **reject `USEFUL_BOMB_PLACED: +1.0` for this configuration**. The
reward successfully encourages crate-directed bombs, but over-incentivizes
bomb placement without accounting for escape safety. It fails the registered
CI, replica-consistency, and self-kill gates. The default reward therefore
remains `0.0`; the opt-in capability is retained for controlled follow-up work.
A sensible next experiment would combine a smaller useful-bomb reward with an
explicit safe-escape condition, registered as a new experiment rather than
tuned on these development results.

## Figures and reproduction

- `figures/collection-effects.png`: paired collection effects and 95% CIs.
- `figures/bomb-behavior.png`: useful-bomb and crate-destruction changes.
- `figures/classic-self-kills.png`: the safety regression and allowed margin.

```bash
python -m training.analyze_issue114_useful_bomb_reward
python -m training.plot_issue114_results
```

`evidence.csv`, `training-summary.csv`, `summary.csv`, and `result.json` are
compact committed derivatives. The immutable run evidence is linked below
after packaging.
