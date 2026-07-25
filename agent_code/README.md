# Agent Code

The framework loads agents by directory name from this folder.

## Supplied baselines

The imported framework provides:

- `random_agent`
- `peaceful_agent`
- `coin_collector_agent`
- `rule_based_agent`
- `fail_agent`
- `user_agent`
- `tpl_agent`

These are reference or integration agents. They are not team-developed learned
models and do not satisfy the final-project requirement.

## Team agents

Create a separate self-contained folder for each distinct learned model by
copying `_team_agent_template`. Give it a stable descriptive name because the
directory name identifies the agent in the framework and tournament.

Every team agent must include:

- `callbacks.py`;
- `train.py`;
- `README.md` as its agent card;
- all evaluation-time Python modules;
- all trained parameters;
- agent-specific `requirements.txt` if it needs extras beyond the repository
  baseline.

Evaluation-time code must not import from outside the agent directory. See
[`docs/0002-repository-architecture.md`](../docs/0002-repository-architecture.md).
