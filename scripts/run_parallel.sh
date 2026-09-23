#!/usr/bin/env bash
# One-command SCILAWS-PARALLEL run: setup -> task fetch -> preflight -> launch.
#
#   export OPENAI_API_KEY=...
#   JOBS=40 bash scripts/run_parallel.sh
#
# Idempotent and resume-safe: re-run the same command at any time. The venv,
# packages and task tree are reused when already valid, and the launcher skips
# tasks that are already finished (it is the only resume mechanism).
#
# Optional environment:
#   JOBS=N           concurrent tasks (default 40; set to what your API quota sustains)
#   TASKS_ARCHIVE=f  install tasks from a prepacked archive instead of Hugging Face
#   DRY_RUN=1        do everything except launch; print the launch plan instead
# Extra arguments are passed to the launcher (scripts/launch.sh).
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=configs/gpt56_luna.yaml
JOBS="${JOBS:-40}"
say() { printf '\n==> %s\n' "$*"; }
next_steps() {
  cat <<'EOF'

Next commands (from the repository root, any shell):
  bash scripts/monitor.sh --watch 30         # progress
  JOBS=40 bash scripts/run_parallel.sh       # resume after any interruption (same command)
  bash scripts/package_results.sh            # when complete: bundle to send back
EOF
}

# 1. API key (never printed)
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is not set. export OPENAI_API_KEY=... and re-run." >&2
  exit 2
fi

# 2-4. Python >= 3.11 and the virtual environment
py_ok() { "$1" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 11) else 1)' 2>/dev/null; }
if [[ -x .venv/bin/python ]]; then
  py_ok .venv/bin/python || { echo "ERROR: existing .venv uses Python < 3.11; remove .venv and re-run." >&2; exit 2; }
  say "reusing .venv ($(.venv/bin/python -V 2>&1))"
else
  BASE_PY=""
  for cand in python3.13 python3.12 python3.11 python3; do
    if command -v "$cand" >/dev/null 2>&1 && py_ok "$cand"; then BASE_PY="$cand"; break; fi
  done
  [[ -n "$BASE_PY" ]] || { echo "ERROR: need Python >= 3.11 (3.13 recommended) on PATH." >&2; exit 2; }
  say "creating .venv with $BASE_PY ($("$BASE_PY" -V 2>&1))"
  "$BASE_PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# 5. requirements (re-installed only when requirements-runner.txt changes)
REQ_SHA="$(python -c 'import hashlib;print(hashlib.sha256(open("requirements-runner.txt","rb").read()).hexdigest())')"
if [[ "$(cat .venv/.requirements.sha256 2>/dev/null || true)" != "$REQ_SHA" ]]; then
  say "installing requirements-runner.txt"
  python -m pip install -q --upgrade pip
  python -m pip install -q -r requirements-runner.txt
  echo "$REQ_SHA" > .venv/.requirements.sha256
else
  say "requirements already installed"
fi

# 6. task tree (reuse if it verifies; otherwise fetch — fetch itself re-verifies)
if python scripts/fetch_tasks.py --verify-only >/dev/null 2>&1; then
  say "task tree verified (118 tasks) — reusing tasks/"
elif [[ -n "${TASKS_ARCHIVE:-}" ]]; then
  say "installing tasks from $TASKS_ARCHIVE"
  python scripts/fetch_tasks.py --from-archive "$TASKS_ARCHIVE"
else
  say "fetching tasks (pinned Hugging Face revision, ~0.85 GB)"
  python scripts/fetch_tasks.py
fi

# 7. preflight (zero API calls); launch only on READY
say "preflight"
PF_LOG="$(mktemp)"
set +e
python scripts/preflight.py --config "$CONFIG" 2>&1 | tee "$PF_LOG"
PF_RC=${PIPESTATUS[0]}
set -e
if [[ $PF_RC -ne 0 ]] || ! tail -n 1 "$PF_LOG" | grep -qx "READY"; then
  rm -f "$PF_LOG"
  echo "ERROR: preflight did not report READY; not launching (see FAIL lines above)." >&2
  exit 1
fi
rm -f "$PF_LOG"

# 8. launch (or plan) through the one launcher; it skips finished tasks
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  say "DRY_RUN=1: launch plan only (no API calls)"
  bash scripts/launch.sh --config "$CONFIG" --jobs "$JOBS" --dry-run "$@"
  next_steps
  exit 0
fi
say "launching with JOBS=$JOBS (Ctrl-C stops; re-run this script to resume)"
next_steps
set +e
JOBS="$JOBS" bash scripts/launch.sh --config "$CONFIG" "$@"
RC=$?
set -e
next_steps
exit $RC
