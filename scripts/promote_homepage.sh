#!/usr/bin/env bash
# Pick one homepage design and make it the published site.
#
#   scripts/promote_homepage.sh v3-editorial
#
# Moves docs/<version>/index.html to docs/index.html, rewrites its ../assets/
# references to assets/, and removes the other drafts and the comparison gallery.
set -euo pipefail

VERSION="${1:-}"
DOCS="$(cd "$(dirname "$0")/.." && pwd)/docs"

if [[ -z "$VERSION" || ! -f "$DOCS/$VERSION/index.html" ]]; then
  echo "usage: $0 <version>"
  echo "available:"
  for d in "$DOCS"/v*/; do echo "  $(basename "$d")"; done
  exit 1
fi

sed 's|\.\./assets/|assets/|g' "$DOCS/$VERSION/index.html" > "$DOCS/index.html"
for d in "$DOCS"/v*/; do rm -rf "$d"; done
echo "promoted $VERSION -> docs/index.html (other drafts removed)"
echo "enable GitHub Pages with source = main branch, /docs folder"
