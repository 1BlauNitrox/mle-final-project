# Hunting curriculum

## Metadata

* Issue: #217
* Branch: `experiment/217-hunting-curriculum`
* Owner: Julius
* Non-author reviewer: pending
* Exact prospective protocol: `config.json`
* Execution commands: `../../training/issue217-commands.md`
* Runtime: `c4ddfa4efadf0b3ec6d4380a4239b9cb3a097113`
* Parent: control-r1 milestone-008000
* Parent SHA-256: `a5dadff5cc9f04a614caa828d3a0f014cad56f00c3644bb791275da85b35a8a6`

## Hypothesis and controls

We tested whether a fixed mixture of ordinary Classic training and validated hunting starts improves Classic eliminations compared with an ordinary continuation while retaining the capabilities of the parent agent.

Both arms preserve the parent networks, replay buffer, Adam state, and target phase. Both use a learning rate of 0.0001 and one update per four collected transitions. Rewards, features, inference, and exploration remain identical.

The experiment therefore measures the combined effect of the changed training distribution, including both changed starting states and opponents. It does not separate the effects of these two components.

## Protocol

We assigned three paired replicas across two machines, with r1 and r2 running on the PC and r3 on the laptop.

Each arm targets 120,000 new gradient updates with a ceiling of 6,000 episodes. Reaching the episode ceiling or a resource stop before the update endpoint makes the run incomplete. We do not treat such a run as a shorter selectable training run.

Curriculum episodes alternate between Classic and hunting scenarios. Hunting opponents change over training:

* episodes 1 to 100: three `peaceful_agent`s
* episodes 101 to 500: three `coin_collector_agent`s
* later episodes: three `rule_based_agent`s

Normal Classic episodes always use three `rule_based_agent`s.

The hunting starts contain no coins and use the official checkerboard walls. They alternate between open and crate corridor layouts, use opponent path distances of 2, 3, 5, and 7, and require validated escape geometry.

These scenarios change the training distribution. They do not change the official action rules, physics, or observations available to the agent.

The configuration registers the exact seed populations, checkpoint triggers, paired 80-game early retention suites, final suites, gates, and resource limits.

Before learning starts, we verify that the zero-update policy behaves identically to the parent.

Early snapshots occur at:

* 100 updates or 10 episodes
* episode 10
* 500 updates or 25 episodes
* episode 25
* episode 50

Later fixed safety checks occur at 10,000, 40,000, and 80,000 updates.

The same gates apply to both arms. We use the early checks as operational screens with point-estimate thresholds. We do not interpret them as repeated hypothesis tests or evidence of statistical non-inferiority.

The CPU ceilings include training, evaluation, and export:

* PC: 24 CPU-hours
* laptop: 12 CPU-hours
* maximum elapsed time per device: 18 hours
* absolute stop: Sunday at 20:00 Berlin time

No policy is installed automatically.

The pooled screen requires three complete pairs. Candidate replica 1 was fixed before we observed the results. Any selected candidate would still require fresh confirmation and compatibility testing before submission.

## Evidence

We retain each device's manifest-verified export ZIP together with the outer `export.json`, final pipeline ledger, `resources.json`, completed or stopped decisions, exact source commit, and bundle checksums.

For retrievable evidence, we record the file size, checksum, schema, retrieval instructions, and recomputation commands rather than relying on machine-local paths.

The retained data also covers:

* generated, replay-resident, and sampled origin proportions
* attack opportunity, reached, and placement exposure
* standard performance metrics and uncertainty
* late-game loops together with eligible-window denominators
* survival

## Executed result

Both devices' lossless JSON observations are retained under `evidence/pc` and `evidence/laptop`.

This experiment stopped during the pilot phase. It did not reach the registered final-checkpoint comparison.

The published laptop r3 archive is release `issue217-laptop-r3-evidence-v1`, asset `issue217-results-1789859140926788500.zip`.

Its size is 2,461,296 bytes and its SHA-256 is:

`3174aa99d93fa53df2a1b6bbf7470c2c5e0a977b07619b8d777a73baa83386ad`

We also verified the PC archive against its complete member manifest.

The PC weights remain local. All observations required to reproduce the reported metrics are retained in Git. We do not need to publish every unselected checkpoint for this stopped-pilot result. Re-running those exact weights would require the model archives separately.

All three paired replicas stopped according to the prospective pilot rules. The registered final screen is therefore incomplete and ineligible.

We select no checkpoint and support no submission promotion from this experiment.

| Replica     | Stop checkpoint | Arm that failed        | Failed retention metric                   |
| ----------- | --------------- | ---------------------- | ----------------------------------------- |
| r1 (PC)     | u500-or-e25     | control                | Classic self-kills +0.125/game            |
| r2 (PC)     | e10             | control                | Classic self-kills +0.150/game            |
| r3 (laptop) | e10             | control and curriculum | Classic self-kills +0.125 and +0.175/game |

The registered limit allowed an increase of at most +0.100 self-kills per game relative to the frozen fallback.

The r1 and r2 curriculum arms passed the check that stopped their pairs, but paired progression required both arms to pass. The r3 curriculum arm also failed.

All three replicas passed the earlier 100-update checks.

These operational screens use fixed point estimates. We do not interpret them as significance tests.

## Training observations

The curriculum changed the training distribution as intended.

At the final available pilot snapshots, curriculum-generated hunting transitions represented:

* r1: 74.3%
* r2: 58.9%
* r3: 64.4%

All control transitions came from Classic games.

The curriculum also increased the number of episodes with an initially reachable attack from zero in every control arm to:

* r1: 6 episodes
* r2: 5 episodes
* r3: 5 episodes

Native kills were:

* curriculum: 8 / 3 / 4
* control: 0 / 2 / 0

Native tagged safe-bomb kill credits for the curriculum arms were 1 / 3 / 2.

We treat these only as descriptive early-training observations. Different replica stop times, small episode counts, and the failed retention criterion prevent a performance claim or candidate selection.

Training loop-window rates under the registered crate-free, hazard-free, progress-free cond
