# Task 2 discount-horizon experiment (Issue #91)

## Hypothesis and evidence boundary

Increasing gamma from 0.90 to 0.97 improves delayed coin collection after
bomb placement, escape and crate destruction. This is a mechanistic hypothesis,
not a claim that higher gamma will improve the agent. Under discounted return,
a reward after twenty additional transitions receives a factor of 0.9^20 =
0.1216 versus 0.97^20 = 0.5438. The effective 1/(1-gamma) horizon rises from
10 to about 33 transitions. This also changes value scale and can increase
instability or preference for delayed survival rewards; larger Q-values alone
are never success. See Sutton and Barto, *Reinforcement Learning*, second
edition, section 3.3 (https://www.incompleteideas.net/book/bookdraft2018mar21.pdf).

The accepted #107 result motivates retaining escape features: its escape-only
arm improved classic survival and collection but failed cumulative Task 1/2
gates. It does not establish that gamma caused the remaining failures. Reward
experiments #103 and protected replay #107 did not establish their proposed
fixes. #124 separately tests rehearsal and masking, so this study holds both
fixed rather than repeat that factorial design. Its incomplete evaluation is
not used to select a treatment, starting replica, thresholds or seeds here.

This work reactivates existing #91 at the user's request. Owner: 1BlauNitrox;
requested reviewer: Waffelmanufaktur. This PR is prospective tooling/protocol
only. No scientific run, improvement or submission-ready model is in scope.
A non-author review of this exact protocol must precede scientific execution.

## Controlled design

Two arms, five paired training replicas each:

| Arm | Gamma | All other settings |
| --- | --- | --- |
| A | 0.90 | Identical |
| B | 0.97 | Identical |

Both use active 26-input escape features, no action mask, uniform replay,
control event rewards, the existing 64x64 DQN, Adam/epsilon/target defaults and
2,000 coin-heaven -> 2,000 loot-crate -> 6,000 classic episodes without opponents.
There are exactly 100,000 training episodes, no post-hoc budget extension and
no best-checkpoint search. Fresh weights come from the committed #87 migration
SHA-256 `4ad409472e7ca008dfcc82aa26017017c90259b65f9e593020d1b63921430f60`.
No #124 result or server download is required. The tool creates separate fresh
checkpoints with identical online/target weights and empty replay; both enable
escape and only gamma differs between their configs. It never overwrites the
agent's committed checkpoint. Changing gamma does not preserve the meaning of
old Q-values as accurate returns; both arms receive the same initialization and
must learn under their registered objectives.

Training world roots: 9100001-9100005; agent roots: 19100001-19100005.
Development world seeds: 29100001-29100040 (classic), 29100101-29100140
(coin-heaven), 29100201-29100240 (loot-crate). Corresponding agent seeds use
39100001-39100040, 39100101-39100140, 39100201-39100240. Repeats reuse the
same pairs and are not independent observations. All final/confirmation seeds
remain unopened. The launcher audits these populations against committed YAML
experiment/plan seed inventories and separates training and evaluation.
The same root is used across the three distinct training scenarios, equally in
both arms; this is the original #107 three-stage schedule, not #124 rehearsal.

Evaluate only each final 10,000-episode checkpoint: forty primary episodes and
forty deterministic repeats in each scenario. Two additional references are
the untrained active-escape migration and the frozen Task 1 DQN. Total: 2,720
evaluation episodes, 1,360 independent primary rows. The frozen reference is
only evaluated on coin-heaven; the untrained reference covers all scenarios.

## Prospective decision

Primary: B-A classic total-board collection fraction must improve by at least
0.10 with a strictly positive paired hierarchical 95% bootstrap lower bound
(10,000 resamples; one efficacy test). All-scenario collection and survival
lower bounds must be strictly above -0.05; self-kill *improvement* lower bounds
must also exceed -0.05. This means a five-point increase in self-kills is not
allowed by the uncertainty guard. No added favorable endpoint can rescue a
failed primary test. Repeats must match deterministic columns and artifacts
must remain unchanged. Evaluation p95 <50 ms and maximum <100 ms are required.

If efficacy and every guard pass, select B; otherwise retain A. In either arm,
choose the median of five replicas ordered by classic collection and replica
ID. Passing this comparison is separate from completing Task 2: retain all
original #107 Task 1, Task 2, determinism and latency gates. A selected model
with failed cumulative gates is only an exploratory Task 3 predecessor.
Missing/invalid observations produce no scientific selection. Never delete
failed seeds or increase gamma again because this treatment failed.

Report collection, zero-coin/full-clear rate, survival, self-kills, invalid
actions, action counts, crate/coin reveal counts, efficiency and latency using
the existing tested aggregation. Report per-stage mean loss and absolute TD
error plus Q-values on three fixed synthetic public-board probes. Those probes
are scale diagnostics, not performance episodes, and do not enter replay.
`collected_per_revealed_coin_proxy` is total collected / total COIN_FOUND events
per model/scenario, null when no coins were revealed. It is not an identified
per-coin conversion rate: initially visible coins can also be collected, so it
may exceed one. Interpret it alongside raw counts and total-board collection.

## Compute and evidence

Proposed server target: the earlier #107 Linux server (four logical CPUs,
about 8 GiB RAM); actual available resources must be verified at startup.
Two concurrent training workers (one per arm), then serial evaluation and
automatic analysis. Ceiling: 24 wall hours, 48 CPU-hours and 4 GiB aggregate
RAM, including supervisor; at least 5 GiB available RAM and two logical CPUs
required to start, stop below 1 GiB system availability. These are ceilings,
not duration forecasts. Parallel server work must leave these resources free;
do not compete with the interrupted #124 evaluation for the same allocation.

The user requested implementation and server instructions, not an automatic
server launch. The execution command explicitly records the human authorizer,
actual hardware and reviewed commit. No training has been started by this PR.
Resume preserves start time, completed stages and all failed attempts; it does
not clear a resource breach. A stale lock after an external kill needs inspection.

Generated `protocol.json` pins configuration, source/dependency fingerprints,
initial artifacts and all plan bytes on the execution host. The server runs a
clean immutable commit and retains per-job metadata, episodes, final artifacts,
failures and cumulative resources. Automatic analysis writes `analysis/result.json`,
`summary.csv`, primary `episodes.csv` and `evidence-files.json`. Retain underlying
repeat observations and metadata: primary rows alone do not prove determinism.
For review, publish the necessary bytes at a durable location with hashes,
sizes and retrieval instructions; a server path alone is not evidence.

## Server commands

Use a new checkout; do not switch or modify an existing running checkout.
Python 3.13 is the documented target. Installing or changing Python is a server
administration task; if `python3.13` is unavailable, use the project's existing
3.13 environment or report the available versions rather than silently change
compatibility targets. The previous #107 server used 3.14, which must not be
mistaken for this recommended target.

```bash
cd /home/julius
git clone --branch experiment/91-task2-discount-horizon --single-branch https://github.com/1BlauNitrox/mle-final-project.git issue91-task2
cd issue91-task2
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
nproc
free -h
.venv/bin/python -m training.run_issue91 --prepare --output-root "$PWD/training_outputs/issue91"
.venv/bin/python -m training.run_issue91 --dry-run --output-root "$PWD/training_outputs/issue91"
```

After non-author review of the exact checked-out commit, start the authorized
campaign inside tmux (use an honest hardware description):

```bash
tmux new -s issue91
cd /home/julius/issue91-task2
.venv/bin/python -m training.run_issue91 \
  --output-root "$PWD/training_outputs/issue91" \
  --reviewed-commit "$(git rev-parse HEAD)" \
  --authorized-by Julius \
  --hardware-description 'Linux server; record actual CPU and RAM here' \
  --authorize-compute 2>&1 | tee /home/julius/issue91-task2/issue91-console.log
```

Detach with Ctrl+B, then D. Reconnect with `tmux attach -t issue91`.
Check `cat training_outputs/issue91/status.json` and `resources.json`.
The output progresses through training, evaluation and completed (after analysis).
For a compatible technical resume, append `--resume` to the same command;
inspect processes/locks first. Keep the repo and environment unchanged.
Recompute without playing games:

```bash
.venv/bin/python -m training.analyze_issue91 --campaign-root "$PWD/training_outputs/issue91"
```

AI assistance: Codex proposed and implemented this controlled protocol and
verification tooling. Tests and primary/code evidence support correctness;
only the future experiment can support an improvement claim.
