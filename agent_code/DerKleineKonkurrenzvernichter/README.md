# DerKleineKonkurrenzvernichter

> task3 baseline agent

`DerKleineKonkurrenzvernichter` is the separately named tabular Task 3
successor of the frozen interim Task 2 agent `DerKleineSprengstoffkapitalist`.

## State and actions

The state contains the five frozen `compact_decision` values followed by three
public-opponent values: shortest-path direction to the nearest opponent,
distance bin, and whether a currently available bomb has a clear blast line to
an opponent. The fixed actions are `UP RIGHT DOWN LEFT WAIT BOMB`.

## Migration

`parent-model.npz` is the Issue #183 frozen Task 2 checkpoint with SHA-256
`93470f92082597b1848f2fa65265c7288dbb1b50ac90370e5747e085e775ea3e`.
For an unseen Task 3 state, the initial six are the parent values for the
dirst five value teask2 prefix. Task 3 starts form the beginning with zero episodes
snd an empty table.

The parent files are now read only.

## Learning

The agent uses tabular Q-learning with learning rate `0.05`, discount factor
`0.9`, seeded epsilon-greedy exploration, and the inherited Task 2 rewards.
The attributable native `KILLED_OPPONENT` event adds `+5`. No deterministic
hunting action is encoded by the features.

## Experimental status

Issue #186 completed the first peaceful-opponent baseline with five independent
10,000-episode replicas. The mean peaceful-opponent elimination rate was `0.08`
per episode versus the registered `0.20` target, so the baseline is rejected as
a Task 3 candidate. Earlier-task collection gates passed (`0.207` Classic,
`1.000` Coin Heaven, `0.319` Loot Crate), and aggregate self-kill rate was
`0.09`. Opponent-suite repeats were not fully deterministic and one maximum
decision-time observation exceeded the registered limit; both remain explicit
limitations. The Task 2 parent itself failed its full Issue #183 confirmation,
and that limitation also remains part of this successor's provenance.

Issue #228 established the unchanged agent's first Task 4 competitive baseline.
After 10,000 Classic training episodes against three rule-based opponents, five
replicas achieved a mean strict first-place rate of 0.04 (95% cluster-bootstrap
interval [0.02, 0.05]) versus the registered 0.10 target. Competitive self-kill
rate was 0.41 and peaceful-opponent elimination retention was 0.08. Collection
and latency gates passed, but the primary, peaceful-retention, and repeatability
gates failed. No Task 4 checkpoint is promoted; the current agent remains a
Task 3 successor rather than a validated competitive policy.
