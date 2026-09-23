#!/usr/bin/env bash
# Bundle the finished run into dist/<run>_<UTC>.tar.gz (send this file back).
#   bash scripts/package_results.sh                 # default config configs/gpt56_luna.yaml
#   bash scripts/package_results.sh --dry-run       # summary only
#   bash scripts/package_results.sh --allow-incomplete   # partial bundle (not all tasks terminal)
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-}"
if [[ -z "$PY" && -x .venv/bin/python ]]; then PY=.venv/bin/python; fi   # set up by run_parallel.sh
PY="${PY:-python}"
args=("$@")
if [[ " $* " != *" --config "* && " $* " != *" --run-name "* ]]; then
  args=(--config configs/gpt56_luna.yaml "${args[@]}")
fi
exec "$PY" -m runner.package "${args[@]}"
