# Server handoff: Issue 147

Preparation is safe now. Launch only after the owner approves this exact
100,000-training / 3,520-evaluation, 24-CPU-hour / 10-wall-hour / 8-GiB proposal,
the exact source receives peer review, CI passes and the server has a free
12-hour allocation. The old #109 authorization does not cover this new study.

Paste each code block intact. Paths are separate from the finished campaign.
The installed Python 3.13 environment from #109 is reused without changing it.

```bash
export TASK147="$HOME/task3-issue147"
export TASK147_PYTHON="$HOME/task3-issue109/repo/.venv/bin/python"
test ! -e "$TASK147/repo"
mkdir -p "$TASK147"
git clone --branch experiment/147-task3-legal-mask https://github.com/1BlauNitrox/mle-final-project.git "$TASK147/repo"
cd "$TASK147/repo"
git rev-parse HEAD
"$TASK147_PYTHON" --version
"$TASK147_PYTHON" -m training.task3_mask_campaign prepare --parent "$HOME/task3-issue109/binding-parent91/task2-parent.pt" --binding-dir "$TASK147/binding"
"$TASK147_PYTHON" -m training.task3_mask_campaign dry-run --binding-dir "$TASK147/binding" > "$TASK147/preflight.json"
cat "$TASK147/preflight.json"
```

Pin the exact reviewed commit supplied in the PR handoff; do not silently use
a newer branch head. `git checkout --detach FULL_REVIEWED_SHA` can pin a fresh,
clean checkout before preparation. If source changes after preparation, create
a new binding directory; never edit the old binding. If the original parent
is missing, retrieve #91 evidence using #109's instructions and verify PARENT
from README.md; do not substitute another model.

After approval, set `REVIEWED` to that full SHA. This authorization declaration
must reflect the owner's actual decision; it is not granted by copying docs:

```bash
export REVIEWED="FULL_REVIEWED_SHA_FROM_PR_HANDOFF"
export TASK147_AUTHORIZED=yes
command -v tmux
tmux new-session -d -s task3-147 "TASK147_AUTHORIZED='$TASK147_AUTHORIZED' TASK147_PYTHON='$TASK147_PYTHON' bash '$TASK147/repo/scripts/run_issue147_server.sh' '$TASK147' '$REVIEWED' > '$TASK147/supervisor.log' 2>&1"
tail -n 30 "$TASK147/supervisor.log"
```

The script checks source, free RAM/disk and authorization, then trains/evaluates,
analyzes and exports automatically. Leave the server on mains power with sleep
and automatic reboot disabled. Do not run competing training/Docker/evaluation
during this allocation. One-round `1/1` evaluation jobs are expected.

Progress, without changing any process:

```bash
tail -n 30 "$TASK147/supervisor.log"
cat "$TASK147/runs/resources.json"
"$TASK147_PYTHON" -c 'import collections,json,pathlib,os; root=pathlib.Path(os.environ["TASK147"])/"runs/plans"; [(print(p.parent.name,collections.Counter(j["status"] for j in json.load(open(p))["jobs"].values()))) for p in sorted(root.glob("*/status.json"))]'
tmux list-sessions
```

If the launcher has exited and no child experiment process remains, retain its
stderr/status and resume the same immutable campaign with the same environment,
hardware record and budget. Do not resume concurrently or change a running
campaign's source, seeds or parent:

```bash
tmux new-session -d -s task3-147-resume "TASK147_AUTHORIZED='$TASK147_AUTHORIZED' TASK147_PYTHON='$TASK147_PYTHON' bash '$TASK147/repo/scripts/run_issue147_server.sh' '$TASK147' '$REVIEWED' --resume >> '$TASK147/supervisor.log' 2>&1"
```

A retained `.task3-campaign.lock` after a crash requires verifying the original
launcher and all children have exited before removing that lock alone. A budget
breach is not resumable under a larger ceiling. If analysis failed partway and
its output directory exists, preserve that directory and use a new analysis
output name; never overwrite evidence merely to make the wrapper run.

Manual analysis/export if required (same checkout/environment):

```bash
"$TASK147_PYTHON" -m training.task3_mask_campaign analyze --binding-dir "$TASK147/binding" --output-root "$TASK147/runs" --analysis-dir "$TASK147/analysis"
"$TASK147_PYTHON" -m training.task3_mask_campaign verify --analysis-dir "$TASK147/analysis"
"$TASK147_PYTHON" -m training.task3_mask_campaign export --binding-dir "$TASK147/binding" --output-root "$TASK147/runs" --analysis-dir "$TASK147/analysis" --archive "$TASK147/issue147-evidence.tar.gz"
sha256sum "$TASK147/issue147-evidence.tar.gz"
```

Transfer only `issue147-evidence.tar.gz` and
`issue147-evidence.tar.gz.manifest.json` first. Preserve the server originals.
On the receiving machine, verify the archive size/SHA against the manifest,
extract to a new directory and run the compact verifier from the exact source:

```bash
python -c 'import pathlib,json,hashlib; p=pathlib.Path("issue147-evidence.tar.gz"); m=json.load(open(str(p)+".manifest.json")); assert p.stat().st_size==m["size_bytes"] and hashlib.sha256(p.read_bytes()).hexdigest()==m["sha256"]'
mkdir issue147-evidence
tar -xzf issue147-evidence.tar.gz -C issue147-evidence
python -m training.task3_mask_campaign verify --analysis-dir issue147-evidence/analysis
```

After verification, publish archive and manifest as durable review evidence.
Report the exact gate decision even if negative. Do not launch #137 automatically.
