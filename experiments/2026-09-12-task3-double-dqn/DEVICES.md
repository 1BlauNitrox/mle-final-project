# Independent work on the other devices

Keep the complete #150 treatment/control comparison on the server. The PC and
16-GB Windows laptop can perform these bounded checks concurrently with it.
They are implementation/reproducibility checks, not additional efficacy trials
or permission to tune parameters. Do not restart completed Task 2 experiments.

## PC: reproduce two diagnosed stalls

Budget: two evaluation episodes, at most 120 seconds per episode, 4 GiB memory;
allow 15 minutes including environment setup. The cases were selected to
reproduce recorded mechanisms, not estimate their population frequency.
Use PowerShell and an installed Python 3.13. Every command is one line.

```powershell
$DiagRoot = "$HOME/task3-diagnosis-repeat"
if (Test-Path $DiagRoot) { throw 'Use a new output directory; preserve previous work.' }
git clone --branch test/146-loop-diagnosis https://github.com/1BlauNitrox/mle-final-project.git "$DiagRoot/repo"
Set-Location "$DiagRoot/repo"
git checkout --detach e960bcc21dbe8b03343b85587a3af725c415616d
py -3.13 -m venv .venv
& ./.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
$Archive = "$HOME/Downloads/issue147-evidence.tar.gz"
if ((Get-FileHash $Archive -Algorithm SHA256).Hash.ToLower() -ne '328b0cf69e99caedd1f60d33eb529f9271c270f8db7ffde571d601f6b30c828a') { throw 'Wrong evidence archive' }
New-Item -ItemType Directory "$DiagRoot/inputs"
tar -xzf $Archive -C "$DiagRoot/inputs" binding/migration/task2-parent.pt campaign/plans/issue147-masked/artifacts/r1/classic-peaceful/checkpoint.pt
& ./.venv/Scripts/python.exe -m training.diagnose_dqn_stalls --agent DagobertDuckDQNTask2 --checkpoint "$DiagRoot/inputs/binding/migration/task2-parent.pt" --world-seed 1460002 --agent-seed 2460002 --output "$DiagRoot/parent-wait"
& ./.venv/Scripts/python.exe -m training.diagnose_dqn_stalls --agent DagobertDuckDQNTask3 --checkpoint "$DiagRoot/inputs/campaign/plans/issue147-masked/artifacts/r1/classic-peaceful/checkpoint.pt" --world-seed 1460001 --agent-seed 2460001 --output "$DiagRoot/masked-cycle"
& ./.venv/Scripts/python.exe -m pip freeze > "$DiagRoot/environment.txt"
tar -czf "$DiagRoot/diagnostic-repeat.tar.gz" -C "$DiagRoot" parent-wait masked-cycle environment.txt
```

The expected examples are the parent WAIT at (3,14) from around step 100,
and masked-r1 movement between (15,4) and (15,5). Return both complete traces
even if they differ. Compare observations with the published #146 evidence;
do not compare instrumented elapsed time as tournament latency.

## Laptop: independently verify the completed #147 evidence

Budget: no games or training, one serial verifier; allow 30 minutes including
installation/download, 4 GiB available RAM and 4 GiB disk. The archive is about
126 MB; the verified raw extract is about 874 MB. This checks portability and
integrity of an existing negative result, not a new experiment.

```powershell
$VerifyRoot = "$HOME/task3-evidence-verification"
if (Test-Path $VerifyRoot) { throw 'Use a new output directory; preserve previous work.' }
git clone --branch experiment/147-task3-legal-mask https://github.com/1BlauNitrox/mle-final-project.git "$VerifyRoot/repo"
Set-Location "$VerifyRoot/repo"
git checkout --detach f0a02975a3b1ddde748d5b1c9820c2b2d21c72ea
py -3.13 -m venv .venv
& ./.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
gh release download issue147-evidence-v1 --repo 1BlauNitrox/mle-final-project --dir "$VerifyRoot/inputs"
& ./.venv/Scripts/python.exe -m training.verify_task3_mask_results --archive "$VerifyRoot/inputs/issue147-evidence.tar.gz" --manifest "$VerifyRoot/inputs/issue147-evidence.tar.gz.manifest.json" --extract-to "$VerifyRoot/import" --output "$VerifyRoot/reproduced"
& ./.venv/Scripts/python.exe -m pip freeze > "$VerifyRoot/environment.txt"
tar -czf "$VerifyRoot/verification-summary.tar.gz" -C "$VerifyRoot" reproduced environment.txt
```

Expected decision: `exploratory_mixed_or_negative`, selected arm/replica null.
Return the compact reproduced analysis and environment record. If verification
fails, retain the error and original inputs; do not reinterpret failure as a
different scientific result or delete the server evidence.

If the PC's archive has not been downloaded, use the same `gh release download`
command and set `$Archive` to its downloaded path. No server folder outside
`/home/julius` is needed. Neither check edits a running campaign.

## Task 4 remains a separate dependency

PR #149 / issue #126 contains a prospective strong-opponent comparison matrix,
not a launch-ready campaign. Its available parent, numeric owner decision,
four-player raw-evidence adapter and measured workload still need completion.
The failed peaceful gates do not authorize bypassing the #137 continuation
helper. A distinct exploratory Task 4 parent needs a prospective registration
and an explicit owner decision. Do not occupy a device with dependent training.
