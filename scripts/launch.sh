#!/usr/bin/env bash
# Launch (or resume -- same command) the SCILAWS-PARALLEL run.
#   JOBS=40 bash scripts/launch.sh --config configs/gpt56_luna.yaml [--tasks-dir DIR]
# JOBS = concurrent tasks; set it to what your API quota sustains. Run inside tmux/screen.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
exec "$PY" -m runner.launch ${JOBS:+--jobs "$JOBS"} "$@"
