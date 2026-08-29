#!/usr/bin/env bash
# Preview docs/ locally. Bound to localhost; reach it over an SSH tunnel:
#   ssh -N -L 8901:localhost:8901 <host>
set -euo pipefail
PORT="${1:-8901}"
cd "$(cd "$(dirname "$0")/.." && pwd)/docs"
echo "serving $(pwd) on http://localhost:$PORT/"
exec python3 -m http.server "$PORT" --bind 127.0.0.1
