# Pull Request

## Summary

<!-- What changed, and why is this the right change? -->

## Linked issue

<!-- Use "Closes #..." only if this PR completes the entire issue. Otherwise
use "Refs #...", state the remaining work, and leave the issue open. -->

Closes #

## Evidence scope

<!-- Check every scope that applies. See docs/0005-definition-of-ready-and-done.md. -->

- [ ] Implementation or validation
- [ ] Prospective experiment protocol (no scientific result in this PR)
- [ ] Completed experiment or training result
- [ ] Frozen or released model
- [ ] Partial or incomplete result record
- [ ] Documentation/process only

## Scientific evidence

<!--
Protocol-only: record the prospective hypothesis, variables/controls, seeds,
baselines, scenarios, budgets, metrics, success criterion, artifact plan, and
validated execution/analysis commands. State explicitly that there are no
results or performance claims in this PR.

Completed experiment/training: identify the registered protocol, exact executed
revisions/configuration/artifacts, retained per-run/per-seed observations,
aggregate results and uncertainty, analysis command, conclusion, limitations,
and decision.

Frozen model: identify the selected evaluation artifact, SHA-256, byte size,
provenance, prospective selection rule, evaluation evidence, and exact export
or verification command.

Partial/incomplete result: identify missing or invalid evidence, narrow the
claims, state the follow-up, and use "Refs" rather than "Closes".

For non-scientific work, write "Not applicable" and explain why.
-->

## External evidence

<!-- Complete this only when large evidence required for a claim remains outside
Git. A machine-local path or checksum without retrievable bytes is insufficient. -->

- Durable location or release identifier:
- SHA-256:
- Byte size:
- Contents and schema:
- Retrieval instructions:
- Verification or reproduction command:
- [ ] Not applicable; all required evidence is committed or no external evidence
      is required.

## Validation

<!-- List exact commands, tests, and manual checks performed. Distinguish smoke
or integration checks from scientific performance evidence. -->

## AI assistance

<!-- Describe material AI assistance and how it was reviewed/refined, or "None". -->

## Definition of Done

- [ ] **Scope:** All acceptance criteria claimed by this PR are met; remaining
      parent-issue work is stated explicitly.
- [ ] **Linked issue:** This PR uses `Closes #...` only if it completes the
      entire issue; otherwise it uses `Refs #...` and leaves the issue open.
- [ ] **Code quality:** Team-owned code is clear and appropriately commented.
- [ ] **Testing:** New behavior is tested; local and CI checks pass.
- [ ] **Evidence scope:** Every applicable evidence scope above is selected and
      its requirements are met.
- [ ] **Claims:** Scientific and technical claims are no broader than the
      retained evidence supports.
- [ ] **Documentation:** Agent cards, dependencies, and numbered decisions are
      updated where applicable.
- [ ] **Compatibility:** Evaluation code is self-contained, uses relative paths,
      avoids multiprocessing, and respects tournament resource limits.
- [ ] **Repository hygiene:** No secrets, report PDF, raw logs, replays, or
      unnecessary large artifacts are committed.
- [ ] **AI review:** AI-assisted material is verified, refined, and disclosed.
- [ ] **Peer review:** At least one other team member approved the PR and all
      conversations are resolved.
