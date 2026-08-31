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
| 2026-08-28 | Issue #34 DQN agent architecture, feature and reward contracts, replay buffer, online and target networks, Bellman updates, seeded policy, checkpoint persistence, framework callbacks, tests, agent card, and review fixes | OpenAI Codex | The author manually integrated and reviewed the suggestions, corrected implementation mistakes through focused tests, and retained ownership of the design. Ruff, compilation, the complete unit suite, seeded training and evaluation smoke runs, resume and determinism checks, checkpoint immutability, latency and memory measurements, packaging, and clean-framework compatibility were completed and recorded in PR #38. Review fixes added whitespace validation, bounded DQN framework smoke coverage, and official-Docker coverage to CI; the imported Dockerfile was pinned to the published Miniconda Python 3.13 image after its moving base changed to Python 3.14 and could no longer install the supplied TensorFlow dependency. Successful Docker validation and non-author approval remain required before merge; no AI output or smoke run is treated as performance evidence. |

Add entries when AI materially influences agent design, implementation,
experiments, analysis, or report drafting. Minor autocomplete need not be logged
individually.
