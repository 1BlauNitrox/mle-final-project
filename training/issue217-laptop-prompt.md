Run the already-authorized overnight laptop part of Bomberman issue #217 / PR
#218, with the recovery watcher and an automatic local commit of generated
results. Julius explicitly requests this execution; do not ask again for the same
authorization. Preserve existing work and prior resource consumption.

Repository: https://github.com/1BlauNitrox/mle-final-project
Branch: experiment/217-hunting-curriculum
Watcher code commit: 521f1e41a71754d286e0d51448df996bddd37653
Scientific bundle source: 6cba50a6ceb98d62103f6bf2a746de5f15f12d45
Release: https://github.com/1BlauNitrox/mle-final-project/releases/tag/issue217-curriculum-v2
Bundle: issue217-laptop.zip, 44937222 bytes
SHA-256: 92940564ba7b09c2be1b104031ad236b20d3ad044c34895249fee6ef98b84670

Read applicable AGENTS.md, #217, #218, training/issue217-commands.md and the
registered config before running. Use gh for GitHub. Fetch the branch, verify
passing CI covering the watcher code, and create an isolated worktree on a new
branch such as experiment/217-laptop-results at the watcher commit above. Never
switch or overwrite a user's active checkout. This worktree is for the watcher
and generated result commits; the immutable extracted bundle runs the experiment.

Download the ZIP and its sidecar using:

    gh release download issue217-curriculum-v2 --repo 1BlauNitrox/mle-final-project --pattern issue217-laptop.zip --pattern issue217-laptop.json --dir <new-download-directory>

Verify size/SHA above, then use scripts/build_curriculum_bundle.py --verify <zip>
--extract <new-directory> to verify every member and extract to a short new path,
e.g. C:/bomberman217-v2. Use Python 3.13 with a dedicated virtual environment and
requirements-dev.txt. Keep all bundle sources and scientific settings unchanged.
The bundle includes the frozen parent and pinned runtime; no other agent download
is needed. Do not use the superseded v1 release.

Inspect active processes and existing roots first. If this laptop already runs
the same verified v2 replica3 campaign, attach the watcher to that root; do not
prepare a duplicate or rerun its mechanics check. If v1 previously ran, preserve
its stopped root and import its consumed CPU and original elapsed start with
--prior-root <stopped-v1-root>. Never claim a fresh budget after an earlier run.

For a fresh run, from the extracted bundle directory:

    <python3.13> scripts/run_hunting_curriculum.py prepare --device laptop --root C:/bomberman217-v2-run --reference inputs/reference.pt --runtime-archive inputs/runtime.tar
    <python3.13> scripts/run_hunting_curriculum.py smoke --root C:/bomberman217-v2-run

Add --prior-root to prepare if needed. Require smoke.json passed=true. Do not
rerun preparation over an existing root. Mechanics checks are disposable and
separate from scientific observations.

Launch ONLY the watcher, detached and hidden. It starts or attaches to the runner.
Substitute these four absolute paths in PowerShell:

```powershell
$pythonExe = 'C:/bomberman217-v2/.venv/Scripts/python.exe'
$watchScript = '<isolated-worktree>/scripts/watch_hunting_curriculum.py'
$runnerScript = 'C:/bomberman217-v2/scripts/run_hunting_curriculum.py'
$runDirectory = 'C:/bomberman217-v2-run'
$resultsRepo = '<isolated-worktree>'
$watchArguments = '"' + $watchScript + '" --root "' + $runDirectory + '" --runner "' + $runnerScript + '" --python "' + $pythonExe + '" --results-repo "' + $resultsRepo + '"'
Start-Process -FilePath $pythonExe -ArgumentList $watchArguments -WorkingDirectory $resultsRepo -WindowStyle Hidden -RedirectStandardOutput ($runDirectory + '/watchdog.out.log') -RedirectStandardError ($runDirectory + '/watchdog.err.log')
```

Check watchdog.json status and PID, supervisor.json, advancing resources.json,
and both error logs. This laptop runs ONLY replica3: paired control/curriculum,
one worker, serial arms. PC owns replicas1/2. Do not launch a second laptop copy.
Record the root, worktree branch and watcher/supervisor PIDs in your response.
The detached watcher stays alive after your turn; keep the laptop plugged in
with its lid open. Windows wakefulness is requested while the processes live,
but the watcher does not survive reboot. After reboot rerun the same watcher
command against the same root. Do not alter system power policy or budgets.

The registered limits remain 12 CPU-hours including training/evaluation/export,
18 elapsed hours, Sunday2026-09-20 20:00 Europe/Berlin absolute stop, <=4GiB
workload RSS and >=2GiB free. Watcher overhead is accounted separately and counted
against the same cap. Preserve all registered seeds, models, rewards, features,
update cadence, metrics and gates. Multiplayer invalid-action margins are
+0.25/game pooled and +0.5/game per replica against both baselines; solo candidate
coin/crate evaluations require zero; latency-only games are excluded.

The watcher checks every20seconds. It handles unexpected process exits, narrow
Windows sharing errors and missing supervisor heartbeat, checks immutable source
and matched checkpoint/state hashes, preserves stop history, cleans up only owned
orphan processes and resumes with prior consumed budget plus conservative debit.
It allows at most3 run recoveries or4 launch attempts. Never bypass failed pilot
gates, STOP.request, budgets, deadline, integrity errors or unknown failures. A
scientific rejection is a valid negative result, not a fault to repair. Do not
edit bound files while the campaign runs, reset ledgers, select intermediate
models, promote a candidate, or use reserved confirmation seeds.

On verified pipeline completion the watcher commits lossless generated JSON
evidence in the isolated worktree under
experiments/2026-09-19-hunting-curriculum/evidence/laptop/. Ensure Git author
identity is already configured and the index is empty before launching. The
watcher refuses main, unrelated staged changes, conflicting prior evidence or
evidence over50MB. It deliberately excludes weights/replay binaries/raw logs,
scientific prose and docs/0006-ai-usage.md. Do not stage those manually. If this
technical commit fails, preserve the output and report the precise blocker;
never invent authorship or change global Git identity. No automatic push or
release upload is required. Julius will return in the morning.

Return the evidence commit/branch, actual completed or rejected/incomplete status,
and the ZIP named in export.json, plus export.json, resources.json, complete.json,
watchdog.json and any STOP/watchdog-error/recovery records. Preserve the whole run
root. The ZIP remains outside Git and must return to the PC; a local ZIP path is
not a durable evidence-publication claim. Only all3 paired replicas together can
support the registered final screen. No result is promised by overnight execution.

Human handoff: Julius must personally read, verify and manually modify the PC's
uncommitted experiment README before it may be committed, including scientific
interpretation. Do not edit docs/0006-ai-usage.md; ask Julius to manually disclose
the laptop training, watcher/recovery and generated evidence assistance. The
older multistep experiment README draft and #213 result-label correction remain
human-owned pending work. Do not accidentally commit any such draft.
