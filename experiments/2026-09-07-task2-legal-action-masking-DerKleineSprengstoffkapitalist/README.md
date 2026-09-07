# Tabular Task 2 legal-action masking

> Status: completed_negative

## Metadata

- Issue: #110
- Agent: `DerKleineSprengstoffkapitalist`
- Owner: LiliWestermann
- Date: 2026-09-07
- Experiment commit: `6ceebe3fe6c75a378512cb7c7ebd022ce302df1c`
- Framework revision: `0f55c1d`
- Reviewer: `Waffelmanufraktur`
- Approval: Personally approved on 2026-09-07 before training.

## Research question

Does consistent framework-legal action masking improve the Task 2 tabular
agent compared with an otherwise identical unmasked control?

## Hypothesis

Applying the same framework-legal mask during exploration, greedy action
selection and Bellman bootstrapping reduces invalid actions and improves
hidden-coin collection on `classic`.

## Experimental variable

The only changed factor is action masking:

- control: `action_masking: none`;
- candidate: `action_masking: framework_legal`.

The mask excludes only actions rejected by the framework. It does not exclude
dangerous but executable actions.

## Controlled variables

Both treatments use identical features, rewards, hyperparameters, parent
artifact, training curriculum, training budget, seeds, scenarios, opponents,
checkpoint selection and evaluation suites.

## Success criteria

The candidate passes if:

- mean classic collection fraction exceeds the control;
- the paired 95% confidence-interval lower bound is above zero;
- at least four of five replicas improve;
- invalid-action rate is below `0.01`;
- Task 1 retention CI remains above `-0.05`;
- no bombs are selected on `coin-heaven`;
- self-kill rate increases by no more than `0.02`;
- p95 decision time is below `50 ms`;
- maximum decision time is below `100 ms`;
- deterministic repeats match.

Confirmation seeds remain unused.

## Execution

Training begins only after the experiment definition has been reviewed.

## Evidence

The compact evaluation evidence is available from the
[Issue #110 evidence release](https://github.com/1BlauNitrox/mle-final-project/releases/tag/issue110-evidence-v1).

- File: `issue110-evidence-v1.tar.gz`
- SHA-256: `0648c1d864e1fc75040c951fe31626db5c18185775c0836457703dc6208f9ce5`
- Size: `1964311` bytes
- Contents: primary and deterministic-repeat episode rows, resolved plans,
  run statuses, evaluated model artifacts and a per-file manifest.

Retrieve and verify:

```bash
gh release download issue110-evidence-v1 \
  --repo 1BlauNitrox/mle-final-project \
  --pattern issue110-evidence-v1.tar.gz

echo "0648c1d864e1fc75040c951fe31626db5c18185775c0836457703dc6208f9ce5  issue110-evidence-v1.tar.gz" \
  | shasum -a 256 --check
```

Reproduce:

```bash
mkdir issue110-evidence
tar -xzf issue110-evidence-v1.tar.gz -C issue110-evidence

python -m training.analyze_issue110_legal_action_masking \
  --plan-root issue110-evidence/issue110-evidence-v1/training_outputs/run-plans \
  --output /tmp/issue110-reproduced
```

## Results

The registered experiment did not pass.

- Classic masked-minus-unmasked mean difference: `-0.00111`
- Paired 95% confidence interval: `[-0.00389, 0.00111]`
- Replicas improving: `0/5`
- Masked invalid-action rate: `0.0`
- Coin-heaven retention difference versus Task 1: `0.0`
- Deterministic repeats: passed
- Runtime limits: passed

Framework-legal masking successfully eliminated invalid actions but did not
improve Task 2 hidden-coin collection.

## Interpretation

Most masked replicas placed very few bombs and collected no hidden coins.
Replica r1 was an exception, but also had a high self-kill rate. Overall,
masking reduced invalid actions and aggregate self-kills without improving
collection performance.

## Decision

The registered collection-performance hypothesis was rejected. Framework-legal
masking eliminated invalid actions and reduced aggregate self-kills without a
statistically demonstrated collection regression. The masking capability is
therefore retained as an optional configuration, but it does not replace the
unmasked baseline.
