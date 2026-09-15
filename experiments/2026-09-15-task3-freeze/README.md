# Task 3 incumbent freeze — draft for owner review

## Decision

Issue #188 freezes the unchanged Task 3 reference as the incumbent for the Task
4 handoff. We selected it on descriptive grounds because it is the best
available artifact across the completed Task 3 measures. This is an explicit
promotion decision even though the registered experiments declined promotion.
It must not be described as passing the Task 3 screen or as establishing Task 2
or Task 3 completion.

## Evidence and limitations

The installed `checkpoint.pt` has SHA-256
`99144d1688f66dcc6369d3efc02b2b9d5755cc8d21e6f2c1e1ffef466d0a7113` and 64,301
bytes. It comes from the verified Issue 168 evidence archive (archive SHA-256
`d9f1a1e14ebb0734ba3cfddafa58e34aabd07393f962394f3d9625a3fe5b4071`) and uses
runtime lineage `c4ddfa4efadf0b3ec6d4380a4239b9cb3a097113`.

The exploration, lower-learning-rate retention, frozen-inheritance,
elimination-reward, and later stability comparisons all preserved their
registered failures and selected no checkpoint. The current freeze does not
rewrite those conclusions, relax any gate, or close the underlying experiment
issues. Task 4 must use a separate successor and a newly registered protocol.

## Reproduction

Download the Issue 168 release archive, verify its recorded SHA-256, extract
`reference.pt`, and verify the installed checkpoint against the hash above.
The exact provenance is also recorded in `agent_code/DagobertDuckDQNTask3/freeze.json`.
