Issue 150: standard versus Double DQN

## Registered question and controls

Issue #150 / PR #154 tested whether Double DQN targets improve peaceful-opponent
elimination over matched standard DQN while passing every hunting and Task 1/2
retention gate. Both arms migrated provisional #91 A/r3, with masking
and escape features enabled, identical features/rewards/hyperparameters, five
paired training roots, 10,000 episodes per replica, and common evaluation seeds.
Only target computation differed. The frozen original Task 2 checkpoint was a
shared reference. The exact registration is retained in `protocol.yaml`.

Executed source: `6f014485a3026cc3707fa2cc3a379880dd0b74bd`.
Parent SHA-256: `c0b6081acf5a733258005d46b9430f3ef50e403470a3bb5428661729f9aeedf7`.
Current PR #154 preparation code contains later storage fixes and is distinct from the executed source.

All 100,000 training and 3,520 registered evaluation episodes completed. Primary
evaluation uses 40 common world/agent pairs per suite; each has an exact repeat.

| Primary peaceful metric | Frozen parent | Standard DQN | Double DQN |
| --- | ---: | ---: | ---: |
| Opponent elimination | 17.5% | 23.5% | 23.5% |
| Strict first place | 82.5% | 89.0% | 86.5% |
| Mean score margin | 4.05 | 4.76 | 4.83 |
| Self-kill rate | 20.0% | 5.0% | 4.5% |

The registered Double-minus-standard elimination contrast is **0 percentage
points, 95% CI [-18.5, +16.0]**, using 10,000 crossed replica/seed bootstrap draws.
It fails the required at-least-10-point benefit with positive lower bound.
This does not prove algorithmic equivalence; it fails to establish the registered
benefit. Both arms also fail the absolute 60% elimination gate and the required
improvement over the parent. Absolute first-place, score-margin and self-kill
gates pass; first-place and score-margin improvement confidence bounds fail.

| Primary collection fraction | Frozen parent | Standard DQN | Double DQN |
| --- | ---: | ---: | ---: |
| Coin heaven | 38.35% | 59.81% | 47.92% |
| Solo classic | 37.22% | 32.94% | 30.56% |
| Loot crate | 23.85% | 35.05% | 21.76% |

Standard DQN passes coin-heaven and loot-crate retention, but fails classic
collection and crate retention. Double DQN fails collection retention on all
three suites and crate retention on classic and loot-crate. A higher sample
mean alone does not establish retention: Double DQN's coin-heaven collection
contrast has a lower confidence bound of -17.24 points against the allowed
-2-point margin. Survival, self-kill and invalid-action retention checks pass.
Runtime gates pass for both arms. `gates.csv` lists every registered decision;
`contrasts.csv` provides the confidence bounds and `per-replica.csv` shows variation.

## Integrity and scope

The verifier checks the archive hash and all 10,642 member hashes/sizes, the
historical Git source and executed dependency fingerprints, initialization,
all final checkpoint flags/counts, raw statistics against normalized observations,
registered seeds, exact repeats, and every analysis result. Recomputed observations,
decisions and source manifests match the server analysis exactly.

`evidence.json` records complete hashes and the verification command.
The verified archives are public at [issue150-evidence-v1](https://github.com/1BlauNitrox/mle-final-project/releases/tag/issue150-evidence-v1).
Committed technical tables alone do not reproduce pooled per-decision latency;
full independent verification requires the external raw archive.

## Decision

The mechanical outcome is `exploratory_mixed_or_negative`, with no selected arm
or replica and no automatic coin-collector continuation. Task 2 remains incomplete.
This supplies a complete, reproducible exploratory Task 3 development comparison,
including all five standard-DQN replicas as a matched benchmark. It does not
establish a passing cumulative Task 3 checkpoint or authorize selecting the best
standard-DQN replica as an unregistered fallback.

Recommended next step to discuss: bounded diagnosis of weak hunting and collection
retention before another long treatment. The current comparison does not support
adopting Double DQN. A new feature/reward intervention, or explicitly exploratory
later-task parent, needs a separate prospective protocol and owner decision.
No new scientific experiment is registered or authorized by this result record.
