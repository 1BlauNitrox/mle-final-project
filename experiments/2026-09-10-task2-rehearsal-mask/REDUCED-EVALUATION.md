# Reduced evaluation amendment (registered before evaluation)

Refs #124. Owner instruction on September 11: "Start the evaluation now".
Stop further training. Evaluate exact-budget A/B/C r1-r5 and D r1-r4, plus
the original frozen Task 1 and untrained references. D/r5 is excluded solely
because its final five training blocks were not completed before interruption.
Retain all failed/partial attempts and never substitute its partial checkpoint.

A/B/C retain their five-replica registered comparisons. D and its conditional
contrasts/interaction are exploratory; this is not completion of the original
four-arm campaign. Keep the original metric definitions, absolute gates,
paired hierarchical bootstrap and conservative four-test multiplicity correction
for A/B/C comparisons. Do not relax thresholds after evaluation. Do not promote
D through the original full-matrix selection rule with a missing replica.
For A/B/C, retain the original eligibility/ranking and representative-replica
rule restricted to eligible A/B/C. No Task 2 completion claim precedes verified
absolute gates, determinism and latency. D may inform a future experiment.

No evaluation outcome was available when this amendment was written. Retain
all original evaluation seeds, scenario conditions, episode counts and repeats
for every included replica: 4,880 evaluation episodes, zero training episodes.
`evaluate_issue124_reduced.py` verifies all final checkpoint bytes against the
completed block-20 record and pins generated plan and artifact hashes before
launch. Reference artifacts remain the original registered references. Earlier
campaign status/resources are read-only and kept separate from this execution.

## Evaluation budget and execution

Owner-authorized evaluation-only recovery: one worker, eight wall hours,
eight CPU-hours, 2 GiB aggregate campaign RAM. Startup requires 2 GiB system
RAM available for this single-worker workload; runtime still stops below 1 GiB.
This is a new bounded evaluation segment, not a reset of the original campaign.
Report its resources alongside all preceding training and failed attempts.
The earlier campaign's September 11 08:59:45 limit remains recorded unchanged;
the newly authorized evaluation segment has its own start and deadline.

Run the reviewed preparation branch's operational script against immutable
training source `1ce18c8736b6a50773b60b95ec0f11dd4c901028`:

```powershell
python scripts/evaluate_issue124_reduced.py --execution-root C:\path\to\issue124-execution --campaign-root C:\path\to\issue124 --output-root C:\path\to\issue124-reduced-evaluation --prepare-only
python scripts/evaluate_issue124_reduced.py --execution-root C:\path\to\issue124-execution --campaign-root C:\path\to\issue124 --output-root C:\path\to\issue124-reduced-evaluation
```

Preparation writes checksum-bound `evaluation-amendment.json` and six plans.
Execution uses an exclusive lock and fresh copied agent workspaces; no training
job may enter the execution plans. The registered seed sets stay unchanged.
`status.json` reports execution progress or interruption; `resources.json`
records the new segment. Each run plan retains raw and normalized observations
and per-job provenance. `evaluation_completed` is not scientific analysis or
Task 2 completion. A separate reduced-scope analysis must verify retained
observations and apply the rules above; the original full-matrix analyzer must
not be made to accept missing original data.
