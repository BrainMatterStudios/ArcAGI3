#!/bin/bash
# Launch several PATH-B game servers+workspaces on consecutive ports for a fan-out.
# Usage: pathb_batch.sh <run_tag> <base_port> <game1> [game2 ...]
# Prints one "GAME=<g> WS=<path> PORT=<p>" line per game (for dispatching subagents).
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
TAG="${1:?need run tag}"; BASE="${2:?need base port}"; shift 2
i=0
for G in "$@"; do
  P=$((BASE + i))
  OUT=$("$DIR/pathb_launch.sh" "$G" "$TAG" "$P" 2>&1) || { echo "LAUNCH FAILED for $G:"; echo "$OUT"; i=$((i+1)); continue; }
  WS=$(printf '%s\n' "$OUT" | grep '^WORKSPACE=' | cut -d= -f2-)
  echo "GAME=$G WS=$WS PORT=$P"
  i=$((i+1))
done
