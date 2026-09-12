# Issue 150 server handoff

Preparation is authorized. Launch the concrete 100,000-training/3,520-evaluation,
24 CPUh / 10 wallh / 8 GiB protocol only after its execution decision is recorded and
the source is reviewed or the owner explicitly records a pre-review exception.
The variable SOURCE identifies code, not evidence of peer approval.
All command lines below are single commands: do not insert newlines inside paths.

## Preparation (Linux server)

Use the existing Python 3.13 environment without upgrading it. Both training
arms and every comparative evaluation stay on this server. Allow 12 hours;
keep 8 GiB RAM and 8 GiB disk free. Warning-only logs are fixed for all three arms;
full episode statistics, metadata, failures and checkpoints are retained. This
reduces the previous verbose-log disk burden; no old output is deleted.

```bash
export ROOT="$HOME/task3-issue150"
export PY="$HOME/task3-issue109/repo/.venv/bin/python"
test ! -e "$ROOT/repo"
mkdir -p "$ROOT"
git clone --branch experiment/150-task3-double-dqn https://github.com/1BlauNitrox/mle-final-project.git "$ROOT/repo"
cd "$ROOT/repo"
git rev-parse HEAD
export SOURCE="$(git rev-parse HEAD)"
git checkout --detach "$SOURCE"
"$PY" --version
"$PY" -m training.task3_double_campaign prepare --parent "$HOME/task3-issue147/binding/migration/task2-parent.pt" --binding-dir "$ROOT/binding"
"$PY" -m training.task3_double_campaign dry-run --binding-dir "$ROOT/binding" > "$ROOT/preflight.json"
cat "$ROOT/preflight.json"
df -h "$ROOT"
free -h
pgrep -af 'python|run_issue|task3_.*campaign' || true
```

Confirm SOURCE equals the exact reviewed/authorized PR head before launch.
The dry-run must say 100000 training, 3520 evaluation, parent_bound true.
It intentionally says compute_authorized false: authorization is a separate
run action. The source/parent/seed matrix must not change afterward.
The parent must hash to c0b6081acf5a733258005d46b9430f3ef50e403470a3bb5428661729f9aeedf7.
If the earlier binding is missing, retrieve the original parent from the
published issue147-evidence-v1 archive; never substitute a trained 147 replica.

## Detached launch

After the exact protocol/budget execution decision, paste this as ONE line:

```bash
tmux new-session -d -s task3-150 "env TASK150_AUTHORIZED=yes TASK150_PYTHON='$PY' bash '$ROOT/repo/scripts/run_issue150_server.sh' '$ROOT' '$SOURCE' > '$ROOT/supervisor.log' 2>&1"
```

A duplicate tmux session is not permission to start another worker. Inspect it
with `tmux list-sessions` and `tmux capture-pane -pt task3-150 -S -30`.
If a separate pre-review exception is used, set TASK150_REVIEW_NOTE in the
command's `env` arguments to its issue-comment URL; do not claim peer approval.
The supervisor writes a completion marker only after the guarded campaign
returns successfully, then performs raw analysis, compact verification and export.

## Progress, interruption and resume

```bash
tail -n 30 "$ROOT/supervisor.log"
cat "$ROOT/runs/resources.json"
"$PY" -c 'import json,pathlib; root=pathlib.Path.home()/"task3-issue150/runs/plans"; [(print(p.parent.name,d["status"],sum(j["status"]=="completed" for j in d["jobs"].values()),len(d["jobs"]))) for p in root.glob("*/status.json") for d in [json.loads(p.read_text())]]'
```

The usual game progress 1/1 is one evaluation seed job. Plan statuses show the
whole matrix: reference 320, control 1605, double 1605 jobs. The resource monitor
keeps the cumulative authorized cap, including interruption time; resume does
not grant an additional budget. Never stop/restart an unrelated process.

If the supervisor has exited and the same run is resumable within its original
budget, keep its exact checkout, binding and outputs. Use a fresh tmux session name:

```bash
tmux new-session -d -s task3-150-resume "env TASK150_AUTHORIZED=yes TASK150_PYTHON='$PY' bash '$ROOT/repo/scripts/run_issue150_server.sh' '$ROOT' '$SOURCE' --resume >> '$ROOT/supervisor.log' 2>&1"
```

Do not remove a campaign lock until its owning process is confirmed dead.
Analysis failure is distinct from training failure: preserve its directory and
error; use a new analysis output directory after diagnosis instead of rerunning
training. The supervisor preserves existing analyses and exports.

## Finished result and compact download

The supervisor already analyzes and exports. These are the explicit commands
if those steps need to be performed separately with a new analysis directory:

```bash
"$PY" -m training.task3_double_campaign analyze --binding-dir "$ROOT/binding" --output-root "$ROOT/runs" --analysis-dir "$ROOT/analysis-new"
"$PY" -m training.task3_double_campaign verify --analysis-dir "$ROOT/analysis-new"
"$PY" -m training.task3_double_campaign export --binding-dir "$ROOT/binding" --output-root "$ROOT/runs" --analysis-dir "$ROOT/analysis-new" --archive "$ROOT/issue150-evidence-new.tar.gz"
```

Normal successful output:

```bash
sha256sum "$ROOT/issue150-evidence.tar.gz"
ls -lh "$ROOT/issue150-evidence.tar.gz" "$ROOT/issue150-evidence.tar.gz.manifest.json"
```

Download only those two files first. On local PowerShell, using the server's
working SSH alias/address in place of `workbench`:

```powershell
scp julius@workbench:/home/julius/task3-issue150/issue150-evidence.tar.gz "$HOME/Downloads/"
scp julius@workbench:/home/julius/task3-issue150/issue150-evidence.tar.gz.manifest.json "$HOME/Downloads/"
Get-FileHash "$HOME/Downloads/issue150-evidence.tar.gz" -Algorithm SHA256
```

Share the hash after download. Keep the server originals until review verifies
all required evidence. The previous same-sized campaign's compact archive was
126 MB (874 MB extracted); warning-only logging targets roughly 1-2 GB total useful
outputs plus staging/transfer headroom, not a guaranteed storage upper bound.

## Incomplete campaign or failed analysis

After the campaign has exited, export partial raw evidence without inventing a
scientific decision. This command refuses an active campaign lock and labels the
manifest `partial_unanalyzed`; it does not require or claim a passing analysis.
Preserve the original failure and never remove an active process's lock.

```bash
"$PY" -m training.task3_double_campaign export-incomplete --binding-dir "$ROOT/binding" --output-root "$ROOT/runs" --archive "$ROOT/issue150-partial.tar.gz"
sha256sum "$ROOT/issue150-partial.tar.gz"
```

Download the partial archive and adjacent `.manifest.json` in the same way as
the complete export. Both export paths retain available supervisor/hardware
records for diagnosis. Do not delete originals after either export.
