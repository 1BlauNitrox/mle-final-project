#!/usr/bin/env bash
# Run only after peer review and explicit approval of Issue 147's exact budget.
set -euo pipefail
ROOT="${1:?Usage: run_issue147_server.sh ROOT REVIEWED_SHA [--resume]}"
REVIEWED="${2:?Full reviewed SHA required}"
RESUME=()
if [[ "${3:-}" == "--resume" ]]; then RESUME=(--resume); fi
cd "$ROOT/repo"
PYTHON="${TASK147_PYTHON:-$HOME/task3-issue109/repo/.venv/bin/python}"
[[ "${TASK147_AUTHORIZED:-}" == "yes" ]] || { echo 'Explicit Issue 147 protocol/budget approval required.'; exit 2; }
[[ "$(git rev-parse HEAD)" == "$REVIEWED" ]] || { echo 'Source differs from reviewed commit.'; exit 2; }
"$PYTHON" -c 'import psutil,shutil; assert psutil.virtual_memory().available >= 8*1024**3, "Need 8 GiB free memory"; assert shutil.disk_usage(".").free >= 20*1024**3, "Need 20 GiB free disk"'
if [[ ! -e "$ROOT/hardware.txt" ]]; then
  { uname -a; lscpu; free -h; "$PYTHON" --version; } > "$ROOT/hardware.txt"
fi
"$PYTHON" -m training.task3_mask_campaign run --binding-dir "$ROOT/binding" --output-root "$ROOT/runs" --reviewed-commit "$REVIEWED" --authorize-compute --authorized-by Julius --hardware-description "$(cat "$ROOT/hardware.txt")" --available-memory-gib 8 --allocation-hours 12 "${RESUME[@]}"
if [[ ! -e "$ROOT/analysis" ]]; then
  "$PYTHON" -m training.task3_mask_campaign analyze --binding-dir "$ROOT/binding" --output-root "$ROOT/runs" --analysis-dir "$ROOT/analysis"
fi
"$PYTHON" -m training.task3_mask_campaign verify --analysis-dir "$ROOT/analysis"
if [[ ! -e "$ROOT/issue147-evidence.tar.gz" ]]; then
  "$PYTHON" -m training.task3_mask_campaign export --binding-dir "$ROOT/binding" --output-root "$ROOT/runs" --analysis-dir "$ROOT/analysis" --archive "$ROOT/issue147-evidence.tar.gz"
fi
sha256sum "$ROOT/issue147-evidence.tar.gz"
