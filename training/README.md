# Training Orchestration

This directory is for tooling shared across experiments and agents:

- curriculum launchers;
- multi-seed runs;
- hyperparameter sweeps;
- result aggregation and plotting;
- optional training-only parallelization.

Framework callbacks and evaluation-time code belong in each agent's own
`agent_code/<agent_name>/` directory. Agents must never import this directory
when `self.train` is false because official evaluation copies only the selected
agent directory.

Every launcher should record the agent commit, configuration, scenarios,
opponents, seeds, rounds, and output location. Follow
[`docs/0004-experimentation-protocol.md`](../docs/0004-experimentation-protocol.md).
