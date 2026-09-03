# 0006 AI Usage and Disclosure

## Course rule

AI may be used to brainstorm, assist with code, and draft report text. The final
code and report must demonstrate that the team performed the main work.
Unreviewed AI drafts are not acceptable.

## Team policy

For every material use of AI:

1. A team member owns the result.
2. The owner verifies technical claims against primary sources or experiments.
3. Generated code receives the same tests and peer review as human-written code.
4. Generated prose is critically edited and rewritten into the team's own
   precise style.
5. The PR explains material AI assistance and the verification performed.
6. No secret, personal, restricted, or unpublished competition information is
   placed into an external AI service.

AI output is a draft or suggestion, never experimental evidence.

## Disclosure log

| Date | Scope | Tool | Human verification and refinement |
| --- | --- | --- | --- |
| 2026-07-25 | Initial repository structure, workflow, requirements summary, AGENTS.md, and documentation drafts | OpenAI Codex | Team must review course-summary accuracy, adapt wording and ownership, run CI, and approve through PR review before merge. |
| 2026-08-16 | Task 1 baseline scope, metric definitions, completion thresholds, and repository contract test for issue #21 | OpenAI Codex | Team must verify that the prospective thresholds and statistical procedure match its intended compute budget, review the wording, run CI, and approve through PR review before merge. |
| 2026-08-19 | Issue #25 experiment metrics, aggregation, runner, plotting pipeline, tests, and documentation | OpenAI Codex | The author reviewed and adapted the generated design and code, traced the framework statistics flow, ran Ruff and the unit-test suite, exercised successful and failed runner paths, inspected normalized and aggregated outputs, visually reviewed regenerated plots, and verified that raw run artifacts remain ignored. The pipeline smoke run was treated only as an infrastructure check and not as experimental evidence. |
| 2026-08-21 | Review fixes for the experiment metric pipeline in issue #25: learning-metric transport, observed-agent isolation, configuration fingerprinting and dirty-state preservation, step semantics, complete action accounting, stored-data validation, coin-efficiency aggregation, tests, and documentation | OpenAI Codex | Team must inspect the callback contract and metric definitions, verify configuration recovery and multi-agent behavior, run the full validation suite, and obtain peer approval before merge. |
| 2026-08-26 | task 1 Q-learning Agent tests and documentation | OpenAI Codex | The author reviewed and adapted the generated material, ran the complete test and lint suite, executed training and evaluation smoke runs, inspected the resulting metrics and plots, and verified that evaluation did not modify the model artifact. |
| 2026-08-28 | Issue #34 DQN agent architecture, feature and reward contracts, replay buffer, online and target networks, Bellman updates, seeded policy, checkpoint persistence, framework callbacks, tests, agent card, and review fixes | OpenAI Codex | The author manually integrated and reviewed the suggestions, corrected implementation mistakes through focused tests, and retained ownership of the design. Ruff, compilation, the complete unit suite, seeded training and evaluation smoke runs, resume and determinism checks, checkpoint immutability, latency and memory measurements, packaging, clean-framework compatibility, and official Docker compatibility were completed and recorded in PR #38. Review fixes added whitespace validation and bounded DQN and Docker smoke coverage to CI; the imported Dockerfile was pinned to the published Miniconda Python 3.13 image after its moving base changed to Python 3.14 and could no longer install the supplied TensorFlow dependency. The Docker check installed the agent-local PyTorch constraint, evaluated against three random agents, verified one CPU thread and checkpoint immutability, and repeated evaluation from a clean packaged-agent framework export. Non-author approval remains required before merge; no AI output or smoke run is treated as performance evidence. |
| 2026-09-01 | automated experiment runner | OpenAI Codex | The author reviewd and adapted the generated code. |
| 2026-09-02 | task 1 Q-learning Agent experiments documentation | OpenAI Codex | The author reviewed and adapted the generated material. |
| 2026-09-01 | Issue #41 prospective DQN Task 1 development-baseline configuration, training diagnostics, serial five-run launcher, failure retention, and tests | OpenAI Codex | The owner must review the fixed seeds, thresholds, source fingerprint, and blocker assessment; inspect the launcher and metric semantics; run the complete validation suite; and obtain non-author review before scientific training. The launcher deliberately refuses dirty, resumed, source-mismatched, or seed-colliding runs. No training result or performance claim was generated by AI assistance. |
| 2026-09-01 | Issue #41 evaluation orchestration, raw decision-time retention, deterministic repeats, aggregation, figures, and result documentation | OpenAI Codex | The owner reviewed the registered seeds and retained training failure, executed all scientific jobs locally, and must review the generated tables and prose plus obtain non-author approval. Metrics were computed from retained raw framework outputs; AI output was not used as experimental evidence. The unavailable tabular pairs are reported as missing rather than reconstructed or imputed. |
| 2026-09-02 | PR #49 review remediation: clean evaluation-artifact staging, compact evidence export and verification, removal of the stale tabular comparison, tests, and documentation | OpenAI Codex | The owner must inspect the provenance design and generated evidence, verify the rerun against the immutable artifact hashes, run the full validation suite, and obtain non-author review. All scientific values remain derived from retained framework outputs; AI output is not experimental evidence. |
| 2026-09-03 | Issue #40 Task 1 baseline freeze, training-protection design, integrity tests, clean-package validation, and documentation | OpenAI Codex | The owner independently selected the frozen model from committed development evidence, reproduced its SHA-256 checksum and byte size, verified its provenance and schema values, reviewed and adapted the suggested code and documentation, and ran Ruff and the complete test suite. Clean-package CI validation and non-author approval must be completed before merge. AI output was not treated as experimental evidence. |
| 2026-09-03 | Issue #65 behavior-preserving tabular Task 2 successor, artifact-migration script, differential tests, agent documentation, architecture update, and clean-package CI | OpenAI Codex | The owner manually integrated and reviewed the suggestions, verified the parent and successor artifact checksums and byte identity, ran Ruff and the focused differential tests, inspected the packaged agent contents. No Task 2 training or performance evidence was generated by AI assistance. |

Add entries when AI materially influences agent design, implementation,
experiments, analysis, or report drafting. Minor autocomplete need not be logged
individually.
