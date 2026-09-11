# 0009 DQN delivery plan: September 11-21, 2026

## Decision and scope

Refs #106, #46, #124, #91, #109, #137, #126 and #127. Julius ended the
standalone Task 2 optimization phase on September 11 and requested a bounded
Task 3/4 and tournament-delivery plan. This is an active scheduling decision,
not an executable experiment protocol or blanket compute authorization.

**Task 2 development is concluded for now; cumulative DQN Task 2 capability is
not complete.** Preserve failed gates and negative evidence. Do not mark #46,
#51, or the original full #124 matrix successful. No defaults or checkpoints
are promoted here. This decision concerns DagobertDuckDQNTask2, not the tabular
DerKleineVermoegensumverteiler / DerKleineSprengstoffkapitalist lineage.

Owner: Julius (1BlauNitrox). Every implementation, protocol, result and freeze
needs a non-author reviewer; request a teammate and confirm availability rather
than assuming it. Other agents should take one focused issue and isolated
worktree, inspect current GitHub state, and avoid the running Task 3 checkout.

## Source requirements checked

The complete 12-page supplied `final_project.pdf` was reread on September 11.
Its pages 1-4 require genuine ML, at least two learned models with one based on
lecture techniques, collaboration across models, and cumulative Tasks 1-4.
Task 2 explicitly includes retaining navigation; Task 4 includes competitive
score against strong opponents. Beating rule_based_agent is a tournament target,
not a result we currently claim. Scientific design/reporting outweigh tournament
placing. Negative results are valuable when explained honestly.

Pages 2-3 and 10-12 require:

- Agent-code zip by **September 21, 2026, 21:00 Europe/Berlin**; internal target
  **18:00**. Submit `final-project-agent-code.zip` containing only the selected
  agent directory, parameters and requirements. MaMPF names/group membership
  and the team announcement must be checked by Julius.
- Optional compatibility upload by **September 17, 21:00**; internal target
  **18:00**. This checks execution against three random agents, not strength.
- CPU-only inference, one Ryzen 5 2600 thread, at most 8 GB RAM, 0.5-second
  decisions and no evaluation multiprocessing. Retain repository safety gates
  p95 <50 ms and maximum <100 ms. Our hosts do not certify tournament-hardware
  latency; use official feedback where available.
- Original course framework and agent-relative files; no imports from training
  or other agents. Check the current official Dockerfile and course notices
  before packaging: the handout permits rule changes until seven days before
  the code deadline. Course announcements have not been inspected here.
- Report by **September 28, 21:00**, about 4,000 words per team member, named
  responsible author after headings, public repository URL, all models and
  positive/negative experiments, declared libraries, AI disclosure and human
  revision, no university logo and no report PDF in Git. Seven sections:
  Introduction, Background, Project planning, Methods, Training, Experiments and
  Results, Conclusion. Assign writers and build evidence tables now.

See [0001](0001-project-requirements.md) for the full repository checklist.
The handout does not prescribe our numeric gates; those are prospective team
contracts and cannot be relaxed retrospectively to turn a failure into success.

## Task 2 closeout and retained evidence

| Study | Verified conclusion | Remaining disposition |
| --- | --- | --- |
| #107 / merged #122 | Escape inputs improve survival; protected replay did not establish retention; no cumulative passing cell | Preserve mixed/negative result and original missing-data limitations; no rerun for cosmetic completion |
| #124 / PR #131 | Reduced 19-model evaluation: 4,880 episodes; D/r5 unfinished. Rehearsal harms coin retention. Masking C has 34.89% classic collection and 0.5% self-kills, but only 42.68% coin-heaven collection versus 74.80% frozen reference | Review reduced result; original full matrix remains incomplete. C passes Task 2-specific gates but fails Task 1 retention/no-bombs and the registered adoption rule |
| #91 / PR #136 | Complete 100,000 training / 2,720 evaluation episodes. Gamma 0.97 reduces classic collection versus 0.90 by 14.39 percentage points (95% CI -23.00 to -4.83); neither arm passes cumulative gates | Retain gamma 0.90; review negative result and disclosed execution-before-review deviation |

Immutable detailed result records and exact reproduction commands:

- [#124 results at 2252c23](https://github.com/1BlauNitrox/mle-final-project/blob/2252c23385c08e1a53d851bb8d5ff6a6e0e5b802/experiments/2026-09-10-task2-rehearsal-mask/RESULTS.md)
- [#91 results at 04e71e3](https://github.com/1BlauNitrox/mle-final-project/blob/04e71e36551c6a1e844311b5c96d851c053e3f56/experiments/2026-09-11-task2-discount-horizon/RESULTS.md)
- [#107 result PR](https://github.com/1BlauNitrox/mle-final-project/pull/122)

Required checkpoint bytes and manifests are published in
[issue124-reduced-evidence-v1](https://github.com/1BlauNitrox/mle-final-project/releases/tag/issue124-reduced-evidence-v1)
and [issue91-evidence-v1](https://github.com/1BlauNitrox/mle-final-project/releases/tag/issue91-evidence-v1).
The pinned records contain sizes, hashes, retrieval and verification commands.
Keep local originals until peer review; no cleanup is authorized by this plan.

#109 now registers #91 A/r3 as its owner-authorized exploratory predecessor:
producing source `cbd52be8392f5003a91c5600fda4efd544b48ec5`, checkpoint 2,382,903
bytes, SHA-256 `c0b6081acf5a733258005d46b9430f3ef50e403470a3bb5428661729f9aeedf7`.
This is the median fallback, not a proven best model. Do not compare #91 and
#124 as matched studies or rebind a campaign after looking at its results.

The user observed occasional loops in a GUI preview. Their frequency, cause and
impact are **unknown**. A code defect, aliased observations, reward incentives
and learned greedy cycles are hypotheses, not established diagnoses. Task 4
training is not a substitute for fixing an implementation defect.

## PR and backlog disposition

Snapshot September 11; refresh before acting:

- #131 and #136: result PRs, nine passing checks each, fresh approval required.
- #129: approved seed/replay tooling, green CI, behind main. Update/test before
  merge. #130: approved Task 3 migration, green CI, also behind main.
- #120: approved current head `0c9b0c1`, green CI, stacked on #130. Retarget main
  after #130 merges, revalidate. #140: green, awaiting review, stacked on #120.
  Merge order #130 -> #120 -> #140; merging is separate from training permission.
- #100: closed without merge as deferred scope. It only records an underspecified
  safety-constrained-exploration proposal, has changes requested, and wrongly
  says `Closes #98`. The reference is now `Refs #98`; the branch and review
  history are preserved. Keep #98 open as deferred backlog, not a failed scientific experiment.
- #89/#90/#98 and broad Task 2 sweeps are off the default critical path. #90 may
  be reconsidered only after a diagnosis supports its mechanism and a new
  reviewed, bounded protocol is ready. Do not silently reopen old campaigns.
- Leave unmet capability/evidence issues open. Administrative closeout of an
  optimization phase does not satisfy experiment acceptance criteria.

## Ten-day execution schedule

Dates are target delivery windows, not promises of convergence. Today is
September 11; the ten remaining days are September 12-21. All times are Berlin.
At 09:00 review status/resources, at 20:00 review evidence and authorize the next
ready allocation. Existing immutable campaigns keep their registered budgets.

| Date | Critical deliverable | Parallel work and exit decision |
| --- | --- | --- |
| Sep 11 tonight | Preserve running exploratory #109 campaign; finish Task 2 documentation | Review #131/#136 and #140; collect machine, output root, source, parent, status/resource records for each active run. No duplicate launch |
| Sep 12 | Verify/analyze peaceful results as soon as complete; commit evidence | Reproduce a loop on development data; prepare #126 analyzer/protocol and #127 clean-framework package checks. If peaceful gates pass, bind #137; if not, follow failure branch below |
| Sep 13 | Run/finish gated coin-collector #137 if ready | Resolve one confirmed defect, or prepare one diagnosis-backed treatment. Freeze exact Task 4 control/opponents and numeric adoption rule before its run |
| Sep 14 | Analyze coin-collector; decision on Task 3; launch first Ready #126 comparison | Check course rule/Docker updates. If Task 3 remains weak, register Task 4 as exploratory under a new explicit parent decision; do not claim #51 completion |
| Sep 15 | Continue bounded Task 4 baseline versus strong-opponent training | Free machine runs one approved targeted optimization or matched control; evaluation machine processes complete campaigns. Keep all four-player conditions represented |
| Sep 16 | Review first Task 4 result and establish a tested fallback package | Prepare optional-upload archive. Select at most one justified final optimization; cancel unstarted low-priority ideas if the schedule slips |
| Sep 17 | Optional compatibility upload by 18:00, hard deadline 21:00 | Use a verified available candidate, not an unfinished run. Continue already registered development tuning on other machines; record exact uploaded hash |
| Sep 18 | Final bounded tuning: fixed training-budget / learning-rate comparison if Ready | Analyze a maximum of one additional factor. No new feature/model-family project. Lock finalist shortlist, selection rule, confirmation/final seed sets and fallbacks before opening them |
| Sep 19 | Finish tuning and independent confirmation; freeze candidate by 18:00 | Start final held-out audit only after selection is locked. No training or performance-driven reselection using final observations; preserve backup package |
| Sep 20 | Complete final evaluation, original-framework, Docker, packaging and review | Repair only reproducible deployment/correctness defects, with tests and transparent artifact/version changes. Revalidate changed artifact; performance weakness uses the predeclared decision/fallback, not more tuning |
| Sep 21 | Submit by 18:00 and verify receipt/team membership | 3-hour upload contingency before 21:00. No new scientific run or untested model. Record exact zip hash and source; organize report handoff |

### Failure branches and stopping rules

The current #109/#137 protocols have conjunctive hunting and retention gates.
#137's helper deliberately refuses a non-passing peaceful result. A weak parent
or one failed gate does not permit us to bypass it. If #109 fails, retain the
negative result and spend at most one decision window (target Sep 12-13) choosing:
(a) one diagnosis-backed repair and new controlled campaign, or (b) a separately
registered exploratory continuation with an explicit owner decision and new
output root. Keep the failed protocol/result intact. This plan does not authorize
(b)'s scientific decisions or implement a bypass. #51's validated-parent contract
remains unmet until evidence actually satisfies it.

If a campaign slips by a day, drop the optional optimization first; keep result
analysis, final evaluation and packaging time. If no treatment is eligible, retain
the prospectively registered fallback and report limitations. Never select the
best-looking seed or stop because an interim score looks favorable. A training
budget extension is a new prospectively defined run, never a silent resume past
an exact-budget selection point. Final tuning must end before final held-out
inspection. No promise that additional training will remove all loops or make
DQN the strongest team model.

## Three-machine allocation

Julius reports PC, 16-GB Windows laptop and Linux server are available for up to
three concurrent campaigns, daytime 09:00-21:00 or overnight. This is availability,
not three approved new protocols. Server/PC capacity and the live Task 3 status
remain unverified. No local Python process was visible during this inspection.

| Resource | Default role | Isolation rule |
| --- | --- | --- |
| Server | Long critical-path training: current peaceful -> eligible coin-collector -> Task 4 | One campaign allocation at a time; dependent stages are sequential |
| PC | Independent matched control or one targeted optimization; otherwise diagnosis/tests | Respect actual free RAM; old memory pressure means availability cannot be assumed |
| Laptop | Serial evaluation and analysis; overnight independent training only when evaluation is clear | One evaluation process/thread; no competing training or Docker during latency measurement |

Keep each campaign's independent replica seeds paired across compared arms. Do
not put every control on Windows and every treatment on Linux and then attribute
host differences to the treatment. Prefer both arms on the same host, or a
prospectively balanced replica/host allocation. Evaluate compared artifacts on
one recorded environment; separately audit clean official compatibility. Moving
jobs between hosts must preserve fingerprints, histories and accounting. Do not
invent distributed resume support in the existing single-host launcher.

Register exact CPU-hours, wall-hours, aggregate RAM, workers, checkpoint cadence,
output roots, source hashes and authorizer for each new run. Existing Task 3
ceilings are 24 CPU-hours / 15 wall-hours / 8 GiB / two training workers, with
serial evaluation. A 15-hour ceiling does not fit a 12-hour availability window:
use an explicitly available longer slot or register a smaller new protocol before
launch. Do not change a running budget or kill it at 21:00 merely for scheduling.

Use tmux on Linux and the reviewed detached supervisor on Windows; mains power,
sleep/reboot settings, free disk and memory preflight, periodic atomic checkpoints,
heartbeat/status and retained stderr. These reduce interruption risk but cannot
guarantee uninterrupted execution. Inspect at least morning/evening and after a
reported failure. Transfer compact manifests/observations/checkpoints promptly;
verify hashes and reproduce analysis before releasing the machine. Reserve time
for evaluation, transfer and review, not only training. Estimate completion from
measured jobs per minute on that exact workload, with recovery margin.

## Ranked experiment queue (proposals, not launch-ready protocols)

1. **Current peaceful hunting, then gated coin-collector** (#109/#137): complete
   existing five-replica 10,000-episode campaigns with 1,920 evaluation episodes
   each. Their registered conditions and criteria stay unchanged. These are the
   immediate source of evidence, not a reason to launch another baseline.
2. **Loop diagnosis**: capture exact development seed, checkpoint/source hash,
   repeated positions, feature vectors, Q-values, selected/valid actions and
   rewards. Distinguish deliberate safe waiting from no-progress cycles. Define
   a stall statistic before comparing treatments; never count GUI impressions
   as efficacy evidence. Fix a confirmed code defect with a failing regression
   test. If movement shaping is implicated, #90's single potential-based change
   is a candidate; terminal and disappearing-coin semantics must be specified.
   Do not add a hand-coded best-action policy or bundle new features and rewards.
3. **Task 4 opponent distribution** (#126, highest-priority new experiment):
   propose fixed rule_based_agent training versus a fixed mixture including
   weaker supplied and frozen learned opponents, with equal episode budgets,
   initial weights and reset rules. Include the unchanged frozen Task 3 parent
   as a no-training reference. Use five paired replicas if measured cost fits;
   any smaller exploratory design must be registered before training. Evaluate
   one- and three-opponent classic matches, including three rule_based_agents,
   mixed opponents and a frozen team agent, with balanced slots and fixed seeds.
   Score/strict wins/eliminations plus all earlier-task guards determine adoption;
   training shaped return alone does not. Exact mixture, counts, seed lists,
   primary endpoint, effect threshold, multiplicity handling and budgets are
   unresolved Definition-of-Ready work in #126, not implied choices here.
4. **Final fine-tuning**: only after Task 4 evidence, propose one lower learning
   rate versus unchanged learning rate at the same fixed extra episode budget,
   keeping the selected opponent distribution, rewards and features fixed. This
   tests stability/retention, not an assumption that longer is better. Retain the
   frozen pre-tuning model as fallback. At most one factor/two trainable arms;
   five paired replicas where feasible and a predeclared median-final selection.
5. **Optional alternatives, not cumulative promises**: consistent legal masking
   is worth a new matched Task 3/4 comparison if invalid actions recur; #124 C is
   suggestive, not adopted. Symmetry augmentation is worthwhile only if transforms
   for all directional inputs/actions/masks are fully tested early enough. Online
   self-play, broad reward grids, architecture changes and safe-exploration #98
   are deferred by default. Do not attempt all of these in ten days.

For every proposed experiment: issue, falsifiable hypothesis, one factor, exact
parent/reset semantics, source/schema, train/development seeds and collision
audit, matched conditions, numeric criteria, uncertainty and selection/fallback,
resource ceiling, analyzer tests, reviewer and explicit compute authorization
must precede launch. Outcomes here are hypotheses, not guaranteed improvements.

## Final selection and reporting

#127 must distinguish development tuning, independent confirmation/selection,
and a final held-out audit. Register finalists and numerical selection/tie-break
rules before confirmation; freeze the selected artifact before final audit.
Include the strongest eligible tabular candidate in the team comparison: the
assignment asks for the best model, not necessarily DQN. Do not generalize this
DQN closeout to the tabular agent's current status.

Report native score, strict wins and ties, kills, survival, self-kills, coins,
crates, invalid actions, latency and per-replica uncertainty on matched suites.
Retain absolute earlier-task results alongside relative retention: retaining a
weak parent's performance is not evidence of cumulative task success. If no model
meets all capability targets, submit the best eligible learned model under the
registered rule and explicitly report unmet targets; runtime compatibility is
still mandatory.

Final package checklist: correct selected artifact/source/hash/size; no preview
or accidental smoke checkpoints; agent-only clean upstream test; declared library
installation; no absolute paths or external runtime imports; one CPU thread and
no multiprocessing; latency/memory checks; official Docker; three-random-agent
smoke; manually inspect zip; verify archive hash after transfer; human upload and
receipt before deadline. Optional September 17 feedback does not certify a later
changed checkpoint: revalidate the exact final bytes.

Maintain a compact evidence table per experiment as results arrive. September
22-28 is for report synthesis, human explanation and final proofreading, not a
way to change the already submitted tournament agent.
