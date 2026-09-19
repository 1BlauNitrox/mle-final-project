# Issue 217 execution commands

These commands operate on private run directories. They do not install a
submission checkpoint. The exact scientific settings are in
`experiments/2026-09-19-hunting-curriculum/config.json`.

Use revision 2 / release `issue217-curriculum-v2`. Version 1 is superseded;
preserve any v1 attempt and never launch its bundle. The amended configuration
uses fresh audited seeds and retains the interrupted PC attempt's CPU debit and
original elapsed start. If v1 ran on this laptop, stop it, preserve its root and
add `--prior-root <stopped-v1-root>` to the new preparation command. This imports
its consumption; it does not grant a fresh budget.

The final invalid-action gate allows at most +0.25/game pooled and +0.5/game for
each replica against both baselines in every multiplayer performance suite.
Solo coin/crate evaluations require zero candidate invalid actions. Latency-only
worlds are excluded from this behavioral gate. The analyzer reports per-game and
per-100-decision counts, loop denominators, attack exposure and native bomb
credits, and replay origins for each retained snapshot, including stopped pairs.
Old rows without bomb attribution report that metric as unavailable.

## Portable laptop setup

Download the checksum-pinned `issue217-laptop.zip` and its JSON sidecar from the
issue217 release. Verify the outer checksum against the supplied handoff, then
use the source-pinned verifier before extraction:

```powershell
python scripts/build_curriculum_bundle.py --verify C:/path/issue217-laptop.zip --extract C:/bomberman217
```

The extraction directory must not already exist. The bundle includes the exact
tools, registered configuration, seed audit, frozen checkpoint and archived
runtime. No Git checkout or external agent download is needed after extraction.
Do not edit bundled files after preparation.

```powershell
Set-Location C:/bomberman217
py -3.13 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
.venv/Scripts/python.exe scripts/run_hunting_curriculum.py prepare --device laptop --root C:/bomberman217-run --reference inputs/reference.pt --runtime-archive inputs/runtime.tar
.venv/Scripts/python.exe scripts/run_hunting_curriculum.py smoke --root C:/bomberman217-run
./scripts/start_hunting_curriculum.ps1 -Python .venv/Scripts/python.exe -RunRoot C:/bomberman217-run
```

Laptop allocation is replica 3, with both arms serially on the same machine.
PC allocation is replicas 1 and 2, one worker per pair. `--device pc` is the only
preparation difference. Use a fresh root for preparation. The laptop cap is
12 CPU-hours; PC cap is 24 CPU-hours. Both stop by Sunday 20:00 Berlin and have
an 18-hour elapsed ceiling. Preparation and bounded mechanics tests precede
the scientific ledger; training, evaluation and export are monitored together.

Keep the laptop plugged in. The Windows supervisor requests system wakefulness
while alive; closing the lid may still suspend the machine under its power policy.

## Status, stop and recovery

```powershell
Get-Content C:/bomberman217-run/resources.json
Get-ChildItem C:/bomberman217-run/pairs/r3 -Filter gate-*.json
Get-Content C:/bomberman217-run/pairs/r3/control/progress.json
Get-Content C:/bomberman217-run/pairs/r3/curriculum/progress.json
New-Item C:/bomberman217-run/STOP.request -ItemType File
```

`decision.json` under each pair distinguishes safety rejection from completed
training. `complete.json` means the scheduled pipeline finished; it does not
mean the policy improved. A failing pilot stops both arms. The supervisor never
promotes a model. Do not delete a gate or STOP record to force continuation.

An interrupted worker resumes from a checkpoint and matching tracker state.
Restarting `run` preserves consumed CPU and elapsed time. Before recovery,
inspect the saved stop reason and ensure no recorded worker is still alive.
Budget or scientific-gate failures do not authorize another attempt. A technical
failure needs a documented correction; preserve the failed root.

## Return results

Return the ZIP identified by `export.json`, plus `export.json`, `resources.json`,
`complete.json` and any `STOP.json`. The outer export record includes the final
pipeline CPU ledger, including archive creation; the ZIP's resource snapshot is
from before export. The ZIP has a SHA-256/size manifest for every member.
If execution stops technically, preserve the entire root; do not claim completion.

After safely extracting PC and laptop evidence into separate directories:

```powershell
python -m scripts.analyze_hunting_curriculum --roots C:/pc-evidence C:/laptop-evidence --output C:/combined-analysis.json
```

Three completed pairs are required for the registered screen. Replica 1 is the
preselected candidate for independent confirmation if the pooled screen passes;
do not choose an attractive intermediate checkpoint or another replica afterward.
Fresh confirmation seeds are reserved in the configuration. Confirmation and
submission packaging remain separate from the overnight launcher.

## Human documentation handoff

The experiment README draft is intentionally excluded from the portable bundle
and technical commits. Julius must read, verify and manually modify it before
commit. Human scientific interpretation and the manual AI disclosure entry in
`docs/0006-ai-usage.md` remain outstanding. No agent card or submission artifact
is updated by these tools.
