# Five-step DQN credit assignment

## Question and design

We tested whether a fixed five-step DQN return improves delayed bomb-to-elimination credit assignment compared with the existing one-step update. Both arms started from the same frozen fallback checkpoint. We kept the network, features, rewards, learning rate, replay capacity, exploration schedule, opponents, seeds, and training budget fixed. The only planned difference was `n_step=1` versus `n_step=5`.

We used three paired replicas for each arm and trained each replica for 1,000 episodes. Training opponents were RUEHL_BASED_AGENT, `rule_based_agent`, and `peaceful_agent`. For evaluation, we used 60 paired classic worlds against three `rule_based_agent` opponents plus three 40-world retention suites. The prospective protocol, gates, and seed ranges are in `config.json`.

## Result

All six training jobs and all registered evaluation and latency jobs completed. We re-ran the analyzer from the committed per-episode observations and reproduced the committed `analysis.json` exactly.

The primary paired estimates are means per game with registered 95% hierarchical bootstrap intervals:

| Contrast                 | Metric     | Estimate |      95% interval |
| ------------------------ | ---------- | -------: | ----------------: |
| five-step minus one-step | kills      |  -0.0167 | [-0.1278, 0.0889] |
| five-step minus one-step | score      |  +0.0833 | [-0.7111, 0.8333] |
| five-step minus one-step | survival   |  +0.1167 | [-0.1167, 0.3278] |
| five-step minus one-step | self-kills |  -0.1111 | [-0.2389, 0.0222] |
| five-step minus fallback | kills      |  -0.0611 | [-0.2000, 0.0667] |
| five-step minus fallback | score      |  -0.1944 | [-1.0333, 0.5833] |
| five-step minus fallback | survival   |  -0.1444 | [-0.3222, 0.0278] |
| five-step minus fallback | self-kills |  +0.0611 | [-0.0944, 0.2056] |

The five-step arm also failed the registered retention gates. Against the fallback, classic-peaceful survival changed by -0.2000 [-0.3667, -0.0583], classic-peaceful self-kills by +0.2000 [0.0583, 0.3667], and loot-crate self-kills by +0.3083 [0.0500, 0.6000]. All latency measurements passed comfortably. Five-step p95 decision latency was 5.61 to 6.27 ms, and the maximum was 10.32 ms.

Late-game looping became worse compared with one-step learning. The pooled late loop rate was 0.5703 for five-step and 0.4361 for one-step. This gives a difference of +0.1342 against the registered maximum of +0.02. The late-state time fraction increased by +0.0298, also beyond its +0.02 gate. The frozen fallback's late loop rate was 0.6506. Neither this isolated detector nor the primary metrics support replacing the fallback.

The efficacy claim failed, the promotion gates failed, and the registered decision rule selected `reference`. We should not ship a five-step checkpoint.

## Interpretation

The tested uncorrected five-step target did not improve hunting or killing. Its point estimate for kills was below both the one-step control and the frozen fallback, while the uncertainty still included modest gains and losses. The arm had a lower self-kill point estimate than the simultaneously trained one-step control, but this did not translate into safety relative to the fallback and did not pass the retention suites.

This result rules out the tested five-step configuration as a deadline-time promotion candidate. It does not show that every multi-step method is harmful. We did not test Retrace or another off-policy correction, other horizons, prioritized replay, or a larger training budget. Given the negative primary, retention, and looping results, extending this exact arm would have a poor expected return before submission.

## Evidence and reproduction

We store the evidence under `evidence/run-82744b0/`. It contains the complete compressed per-episode observations needed for the statistics, the prospective config, environment binding, source manifest, compact summaries, and file manifest. We intentionally exclude checkpoints, replay snapshots, and framework logs.

From the repository root, we reproduce the analysis with:

```powershell
python scripts/analyze_five_step_experiment.py `
  --root experiments/2026-09-19-five-step-dqn/evidence/run-82744b0 `
  --config experiments/2026-09-19-five-step-dqn/config.json `
  --output issue207-analysis-reproduced.json
```

The key machine-readable decision fields are `claim_success: false`, `promotable: false`, and `selected_artifact: "reference"` in the committed `analysis.json`.
