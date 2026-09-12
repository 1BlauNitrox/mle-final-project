#!/usr/bin/env bash
# Immutable Issue 150 campaign. Explicit authorization is separate from peer review.
set -euo pipefail
ROOT="${1:?Usage: run_issue150_server.sh ROOT SOURCE_SHA [--resume]}"
SOURCE="${2:?Exact source SHA required}"
RESUME=()
if [[ "${3:-}" == "--resume" ]]; then RESUME=(--resume); fi
cd "$ROOT/repo"
PYTHON="${TASK150_PYTHON:-$HOME/task3-issue109/repo/.venv/bin/python}"
[[ "${TASK150_AUTHORIZED:-}" == "yes" ]] || { echo 'Explicit Issue 150 protocol/budget authorization required.'; exit 2; }
[[ "$(git rev-parse HEAD)" == "$SOURCE" ]] || { echo 'Source differs from pinned commit.'; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo 'Source must be clean.'; exit 2; }
if [[ ! -e "$ROOT/run-completed" ]]; then
  "$PYTHON" -c 'import psutil,shutil; assert psutil.virtual_memory().available >= 8*1024**3, "Need 8 GiB free RAM"; assert shutil.disk_usage(".").free >= 8*1024**3, "Need 8 GiB free disk for compact logging"'
  if [[ ! -e "$ROOT/hardware.txt" ]]; then
    { uname -a; lscpu; free -h; "$PYTHON" --version; printf '%s\n' "${TASK150_REVIEW_NOTE:-Review/owner execution decision recorded on Issue 150}"; } > "$ROOT/hardware.txt"
  fi
  "$PYTHON" -m training.task3_double_campaign run --binding-dir "$ROOT/binding" --output-root "$ROOT/runs" --reviewed-commit "$SOURCE" --authorize-compute --authorized-by Julius --hardware-description "$(cat "$ROOT/hardware.txt")" --available-memory-gib 8 --allocation-hours 12 "${RESUME[@]}"
  printf '%s\n' "$SOURCE" > "$ROOT/run-completed"
fi
[[ "$(cat "$ROOT/run-completed")" == "$SOURCE" ]] || { echo 'Completion source mismatch.'; exit 2; }
if [[ ! -e "$ROOT/analysis" ]]; then
  "$PYTHON" -m training.task3_double_campaign analyze --binding-dir "$ROOT/binding" --output-root "$ROOT/runs" --analysis-dir "$ROOT/analysis"
fi
"$PYTHON" -m training.task3_double_campaign verify --analysis-dir "$ROOT/analysis"
if [[ ! -e "$ROOT/issue150-evidence.tar.gz" ]]; then
  "$PYTHON" -m training.task3_double_campaign export --binding-dir "$ROOT/binding" --output-root "$ROOT/runs" --analysis-dir "$ROOT/analysis" --archive "$ROOT/issue150-evidence.tar.gz"
fi
sha256sum "$ROOT/issue150-evidence.tar.gz"
