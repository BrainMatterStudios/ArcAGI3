#!/bin/bash
# Stop a PATH-B run's server.
set -euo pipefail
RUNDIR="${1:?need rundir}"
if [ -f "$RUNDIR/server.pid" ]; then
  PID=$(cat "$RUNDIR/server.pid")
  kill "$PID" 2>/dev/null || true
  echo "killed server pid $PID"
fi
