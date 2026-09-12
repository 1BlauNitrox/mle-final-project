# Issue 146: bounded development diagnosis

This is read-only diagnosis, not an efficacy experiment or a model selection.
The owner authorized short tests to prepare the next Task 3 protocol. The
[prospective registration](https://github.com/1BlauNitrox/mle-final-project/issues/146#issuecomment-5644860530)
defines 22 single episodes: original #91 A/r3 on two opponent-free classic
seeds, and all ten #147 finals on classic alone and against peaceful_agent.
No favorable replica is selected and no optimizer updates occur.

World/agent pairs are (1460001, 2460001) and (1460002, 2460002); the first
is solo, the second peaceful except for the Task 2 parent. These fresh
development diagnostic seeds are not efficacy/confirmation/final seeds.
Limits: serial CPU, 30 minutes wall, 0.5 CPU-hours, 4 GiB process RAM,
120 seconds per episode. At most three exact repeats may validate detections.

A conservative stall is 24 consecutive steps with unchanged field and visible
coins, no bombs/explosions anywhere, no coin/crate/kill progress and position
period 1-4. Excluding all board hazards deliberately undercounts some stalls.
It distinguishes unsafe waiting from persistent static-world behavior; absence
does not prove there are no loops. Aliased feature vectors across positions
are descriptive, not by themselves defects. No policy action is overridden.

`training.diagnose_dqn_stalls` records full public states, features, Q-values,
executed actions, legality, native/custom reward components and immutable
artifact/source identity. Instrumented time is not latency evidence. Example:

```bash
python -m training.diagnose_dqn_stalls --agent DagobertDuckDQNTask3 --checkpoint /path/to/registered/final.pt --world-seed 1460002 --agent-seed 2460002 --opponent peaceful_agent --output training_outputs/diagnosis/example
```

Candidates and original campaigns must remain untouched. The output directory
must not exist. Findings will be retained separately with unsuccessful attempts,
and any resulting treatment will need its own prospective protocol. No cause
of the user's exact GUI event is presumed, since its state/round is unknown.

AI assistance: Codex designed the diagnostic instrumentation and tests, with
human review required. The instrumented observations, not AI output, support
any reported finding.
