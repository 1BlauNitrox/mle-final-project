# DRAFT — Issue #210: one-step vs five-step Q-learning

> status: completed failed
## Question

Does a five-step Q-learning target propagate the delayed reward for eliminating
a peaceful opponent better than the current one-step update, without materially
reducing the retained Task 2 abilities?

The treatment follows the n-step bootstrapping formulation in Sutton and Barto,
*Reinforcement Learning: An Introduction* (2nd edition), Chapter 7. For a horizon
of five, the update target contains up to five discounted rewards followed by a
discounted bootstrap value. A terminal state truncates the return.

## Preregistered design

- Control: `tabular_update_horizon: 1`.
- Candidate: `tabular_update_horizon: 5`.
- Both arms use `compact_opponent`, the frozen Task 2 prior, and otherwise
  identical hyperparameters.
- Five independent replicas per arm train for 10,000 classic rounds against one
  `peaceful_agent`.
- Each replica is evaluated on 20 paired seeds in peaceful, coin-collector,
  classic, coin-heaven, and loot-crate suites, followed by exact repeats.
- Planned budget: 100,000 training and 2,000 evaluation episodes (2,010 jobs).

The complete gates and seeds are fixed in `config.yaml` and the two run plans.

## Results

All 1,005 jobs per arm completed without a failed job. The five-step candidate
did not improve peaceful-opponent hunting:

| Metric | One-step | Five-step | Difference |
| --- | ---: | ---: | ---: |
| Peaceful opponents eliminated / episode | 0.080 | 0.060 | -0.020 |
| Peaceful collection fraction | 0.244 | 0.147 | -0.098 |
| Peaceful self-kill rate | 0.050 | 0.070 | +0.020 |
| Classic collection fraction | 0.183 | 0.014 | -0.169 |
| Coin Heaven collection fraction | 0.924 | 0.874 | -0.0504 |
| Loot Crate collection fraction | 0.232 | 0.012 | -0.220 |

## Interpretation

The five-step treatment is rejected and the current one-step update is retained.
Longer returns did not make the sparse elimination reward easier to exploit in
this setting. Instead, they changed the learned policy enough to reduce hunting,
coin collection, and overall safety. The candidate visited fewer distinct table
states but accumulated many more visits per state, so its failure is not
explained by insufficient state reuse.

The experiment doesn't show that n-step leasrning is worse, but that n=5 doesn't
improve the agent.
