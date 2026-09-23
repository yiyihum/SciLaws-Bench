#!/usr/bin/env bash
# Progress of a run (read-only). Default config: configs/gpt56_luna.yaml
#   bash scripts/monitor.sh                      # one snapshot
#   bash scripts/monitor.sh --watch 30           # refresh every 30 s
#   bash scripts/monitor.sh --run-name <name>    # another run dir (e.g. the smoke run)
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
args=("$@")
if [[ " $* " != *" --config "* && " $* " != *" --run-name "* ]]; then
  args=(--config configs/gpt56_luna.yaml "${args[@]}")
fi
exec "$PY" -m runner.monitor "${args[@]}"
