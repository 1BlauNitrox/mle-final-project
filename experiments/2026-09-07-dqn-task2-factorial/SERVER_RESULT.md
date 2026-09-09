# Issue #107 independently reproduced result

The server-generated result was independently reproduced on 2026-09-09 from
downloaded observations using the exact execution revision. All 5,180 jobs and
twenty final training artifacts were verified, with no failed attempts. The
complete recomputed result equals the server JSON and reports
`analysis_valid: true`. The prospective protocol and thresholds remain unchanged.
Issue #107 remains open for review and the documented Task 3 handoff.

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

The selected bytes and all twenty final artifacts passed independent checksum
verification. No model is frozen or replaced by this record.

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

Next steps are to review the reproduced analysis and retained per-episode
evidence, then complete the documented parent binding for Task 3.
The accepted roadmap proceeds to #109 using the mechanically selected
predecessor, explicitly as exploratory if Task 2 remains incomplete. PR #120
still needs final parent binding, numeric decision criteria, owner/reviewer
and compute authorization before execution. No additional Task 2 sweep is
authorized under #107. Any decision to pursue B instead requires an explicit
new scientific plan; it must not silently replace the registered selection.

## Evidence status and reproduction

### Replica-level diagnosis from the downloaded summaries

These are descriptive observations from `server-analysis/result.json`, not
additional prospective tests or a new selection rule. Each row summarizes 40
primary episodes per scenario.

| Escape-only B replica | Classic collection | Classic self-kills | Task 1 collection | Task 1 self-kills | Task 1 invalid actions |
| --- | ---: | ---: | ---: | ---: | ---: |
| r1 | 20.00% | 2.5% | 5.75% | 0% | 2.55% |
| r2 | 25.56% | 0% | 31.35% | 22.5% | 37.87% |
| r3 | 24.44% | 5% | 22.10% | 0% | 28.12% |
| r4 | 21.94% | 2.5% | 10.95% | 0% | 2.69% |
| r5 | 29.17% | 0% | 98.75% | 0% | 12.10% |

The classic survival effect is distributed across all five B replicas; it is
not created by a single unusually safe seed. Task 1 performance is much less
consistent. Four B replicas collect at most 31.35%, while r5 collects 98.75%.
The latter still violates the Task 1 invalid-action and zero-BOMB requirements,
and its classic invalid-action rate is 23.27%, above the per-replica limit.
Selecting r5 as a success would therefore both ignore failed gates and violate
the registered median selection rule. B/r2 accounts for all nine Task 1
self-kills among B's 200 primary episodes; retaining this replica is essential
to an honest uncertainty estimate.

Task 1 failure is not just a survival issue: all A replicas survive every
coin-heaven episode, yet collect only 3.45%-8.20%. A/r1 spends 15,402 of 16,000
actions waiting (96.26%); A/r3 has an invalid-action rate of 81.44%. B/r1 also
waits on 15,364 of 16,000 actions (96.03%). These counts support passivity and
invalid-action problems, but cannot establish their learning-time cause or
prove a particular movement loop without trajectories. Action-BOMB counts are
attempts, not necessarily successful bomb placements.

### Optimization implications and limits

The strongest supported direction is to preserve the escape mechanism while
addressing navigation retention and action quality. This is a recommendation
for a future controlled proposal, not adoption under #107's failed guards.

- Audit the retained training records and checkpoint replay partitions before
  proposing a fix: was the protected pool populated as configured, sampled
  at the fixed quota, and retained through the stage transitions? A correct
  implementation can still fail scientifically. The aggregate outcome does
  not prove that replay is broken or that a larger quota will work.
- If implementation checks pass and the team explicitly authorizes a new
  study, a focused candidate is continued Task 1 practice during later
  training, compared with the same escape-enabled baseline under an equal
  total episode budget. Fresh experience differs from repeatedly sampling a
  fixed historical pool. Its benefit is a hypothesis; exact schedule,
  thresholds, seeds and budget must be registered before execution.
- Invalid actions deserve attention, but #107 did not test action masking.
  The earlier #86 masking comparison failed its performance guards. A new
  escape-plus-mask interaction cannot be declared beneficial from these
  tables; it would need its own controlled test.
- Do not change rewards, replay quota, network size and training duration
  together. Nothing in this experiment isolates a need for a larger network,
  longer training or a particular reward adjustment.

The accepted #106/#109 time box still takes precedence: prepare Task 3 and
finish this evidence review rather than silently start another Task 2 sweep.
If the team chooses B as an exploratory Task 3 parent, record that departure
from A/r2 explicitly and preserve B's active escape inputs in the successor.
The currently prepared 34-input Task 3 migration accepts only a control parent
with disabled continuation inputs; passing B through it would discard learned
information and is intentionally rejected. A B successor needs a reviewed
representation retaining the 26 Task 2 inputs plus the 13 opponent inputs
(39 inputs if appended unchanged), with migration and regression tests.

### Independent verification

All six plans and 5,180 jobs passed local verification. The verifier checks
clean source revision, campaign authorization, plan/job fingerprints, completed
attempts, episode counts, artifact SHA-256 values and the full registered
analysis. Twenty final checkpoints were verified; no failed attempts were
recorded. The recomputed result equals the complete downloaded server JSON.

The original analyzer binds fingerprints to the current host's installed
packages and path ordering. Linux and Windows order mixed-case agent filenames
differently. `scripts/verify_issue107_download.py` reconstructs Linux directory
ordering and uses the internally hashed, consistent execution dependency
inventory for plan validation. It separately records the local analysis
environment (Python 3.13.14 / NumPy 2.5.1, versus server 3.14.4 / 2.5.3).
It changes no observations, thresholds, resampling, ranking or scientific
decision logic; the registered analyzer runs from its clean execution checkout.

`server-analysis/` retains the downloaded analyzer outputs; Git normalizes CSV
line endings to LF without changing field values. These
aggregate files are not sufficient to reproduce confidence intervals without
the underlying episode evidence. The server resource record reports 54,882.1
CPU seconds (15.25 hours), 28,892.67 wall seconds (8.03 hours), peak memory
1,689,563,136 bytes (1.57 GiB), no active roots and no recorded budget breach.
These records passed the registered campaign checks during local reproduction.

The user ran the registered command in the unchanged server checkout:

```bash
python -m training.analyze_issue107_task2_factorial
```

It reported selection A/r2 and `Analysis valid: True`; independent local
reproduction matches that result exactly. See [EVIDENCE.md](EVIDENCE.md) for
the evidence archive, checksums, and cross-platform verification command.
This result uses Refs #107 while non-author review and Task 3 handoff remain.

AI assistance: OpenAI Codex inspected the downloaded output and drafted this
interpretation. A human reviewer must verify the result and its scope; AI
output is not scientific evidence.
