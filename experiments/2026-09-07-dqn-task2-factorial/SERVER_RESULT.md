# Issue #107 server result: independent evidence verification pending

This record interprets the server-generated analysis downloaded on 2026-09-09.
The server analyzer reports `analysis_valid: true`. The complete raw evidence
transfer is still in progress, so this record is partial and must not be treated
as an independently reproduced completed experiment. Issue #107 remains open.
The original prospective protocol and its thresholds remain unchanged.

## Execution and design

Execution revision: `69e2a931e4ebfd358cc0128049516b1a12adc2a8`, approved in
PR #118 before execution. The downloaded authorization names Julius /
1BlauNitrox and records Linux, Python 3.14.4, four logical CPUs and 8 GiB
available memory. Python differs from the recommended local 3.13 environment;
this must be retained in the reproduction record.

The experiment compared two factors in the cumulative Task 2 DQN:

| Cell | Multi-step escape inputs | Protected Task 1 replay |
| --- | --- | --- |
| A | off | off |
| B | on | off |
| C | off | on |
| D | on | on |

Escape inputs describe whether each movement or WAIT action admits a safe
continuation through the search horizon. They are inputs to the learned policy.
Protected replay reserves 2,000 of 10,000 transitions from initial Task 1
training and samples 16 of each 64-transition batch from that partition when
available. Rewards, initial weights, network size, curriculum and paired seeds
were held fixed. Five replicas per cell each trained for 2,000 coin-heaven,
2,000 loot-crate and 6,000 classic episodes without opponents.

The server analysis reports 200,000 training episodes across 60 stage jobs,
2,560 primary evaluation episodes and 2,560 deterministic repeats. Comparisons
use the registered hierarchical paired bootstrap with 10,000 resamples;
efficacy decisions use 98.75% multiplicity-adjusted intervals. Repeat episodes
are not independent observations.

## Reported measurements

Values below are means over the five equal-sized replica evaluation suites.
Collection is the fraction of all board coins collected, including hidden coins.

| Cell | Classic collection | Classic self-kills | Coin-heaven collection | Coin-heaven survival | Loot-crate collection |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 1.67% | 75.0% | 5.59% | 100.0% | 2.32% |
| B | 24.22% | 2.0% | 33.78% | 95.5% | 14.70% |
| C | 0.72% | 51.0% | 4.79% | 39.0% | 1.16% |
| D | 17.28% | 6.5% | 21.31% | 63.5% | 10.61% |

The frozen Task 1 reference collects 83.75% on coin-heaven. The untrained
migration matches that fraction, but collects zero coins on classic and
loot-crate. Every trained cell fails both the complete Task 2 and Task 1 gate
sets. All cells pass the analyzer's determinism and measured latency gates;
these server measurements are not an official tournament compatibility test.

## Planned contrasts and decision

- Escape B-A reduces classic self-kills by 73 percentage points: 95% CI
  [61, 85.5], adjusted interval [58, 88]. Classic collection increases by
  22.56 points, 95% CI [18.11, 27.22]. The escape efficacy criterion passes.
- B fails the coin-heaven survival guard: difference -4.5 points, 95% CI
  [-14, 0]. The rule requires the lower bound to be strictly above -5 points.
  This is failure to establish non-regression, not proof of a survival loss.
- Escape D-C reduces classic self-kills by 44.5 points: 95% CI [17, 71.5],
  adjusted interval [9, 75.5]. Its efficacy criterion also passes, but its
  loot-crate survival guard lower bound is exactly -5 points and therefore
  fails the strict rule.
- Replay C-A changes coin-heaven collection by -0.8 points: 95% CI
  [-4.36, 3.34]. Replay D-B changes it by -12.47 points: 95% CI
  [-38.62, 4.47]. Neither meets the required +10-point improvement and
  positive adjusted lower bound. C and D also fail non-regression guards.
- Classic collection interaction D-B-C+A is -6 points, 95% CI
  [-11.67, -0.78]. This suggests less-than-additive benefit on that metric;
  it does not establish a general interaction across tasks or configurations.

No treatment is eligible under the prospective selection rule. The server
therefore selects A, then its median replica r2. This fallback is a protocol
decision, not a claim that A has the strongest observed performance.

Selected training artifact reported by the server:

- Path within the control plan: `artifacts/r2/classic-crates/checkpoint.pt`
- SHA-256: `d571b199bc00a0419d16706cc5f7844364a184384058268b2cc8de5b79bb479d`
- Size: 2,382,903 bytes
- Task 3 status: `exploratory_predecessor_task2_not_complete`

The bytes and all twenty final artifacts still require independent local
verification after transfer. No model is frozen or replaced by this record.

## Interpretation and follow-up

The reported result supports multi-step escape inputs as a promising mechanism
for survival and collection in this configuration. It does not authorize
adoption under the registered all-task safeguards. Protected replay at this
fixed quota did not solve forgetting; changing its quota would be a new
experiment, not a reinterpretation of this one.

Task 2 is incomplete. Even B's 24.22% classic collection is below the 30%
absolute requirement, its per-replica invalid-action gate fails, and its
Task 1 gate set fails. The strong escape effect and severe retention failures
should both be preserved in the report.

Next steps are to finish transfer, reproduce the full analysis, retain compact
per-episode evidence and artifact provenance, then obtain non-author review.
The accepted roadmap proceeds to #109 using the mechanically selected
predecessor, explicitly as exploratory if Task 2 remains incomplete. PR #120
still needs final parent binding, numeric decision criteria, owner/reviewer
and compute authorization before execution. No additional Task 2 sweep is
authorized under #107. Any decision to pursue B instead requires an explicit
new scientific plan; it must not silently replace the registered selection.

## Evidence status and reproduction

`server-analysis/` retains the downloaded analyzer outputs; Git normalizes CSV
line endings to LF without changing field values. These
aggregate files are not sufficient to reproduce confidence intervals without
the underlying episode evidence. The server resource record reports 54,882.1
CPU seconds (15.25 hours), 28,892.67 wall seconds (8.03 hours), peak memory
1,689,563,136 bytes (1.57 GiB), no active roots and no recorded budget breach.
These records are retained alongside the outputs; they are not independently
audited here yet.

The user ran the registered command in the unchanged server checkout:

```bash
python -m training.analyze_issue107_task2_factorial
```

It reported selection A/r2 and `Analysis valid: True`. Local execution from
the exact revision currently stops at missing control-plan `status.json`.
Local reproduction must also account explicitly for the analyzer's execution
dependency fingerprints; it must not bypass provenance validation to make a
different runtime pass. Completion requires the full evidence and a durable,
reviewer-accessible reproduction package. This partial record uses Refs #107.

AI assistance: OpenAI Codex inspected the downloaded output and drafted this
interpretation. A human reviewer must verify the result and its scope; AI
output is not scientific evidence.
