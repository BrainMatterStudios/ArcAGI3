#!/bin/bash
# PATH-B lean launcher: serve one local dev game via astroseger's server (OFFLINE Arcade)
# and set up a fresh agent workspace. Replaces the Docker/proxy/codex controller.
#
# Usage: pathb_launch.sh <game_alias e.g. cd82> <run_tag> [port]
# Prints the run dir + workspace path. The real game id is hidden behind the alias "target".
set -euo pipefail

REPO="/Users/ahmed/Documents/ArcAGI3"
SCAF="$REPO/scratchpad/ewm_pathb"
PY="$REPO/.venv/bin/python"
ENVDIR="$REPO/environment_files"

GAME="${1:?need game alias e.g. cd82}"
TAG="${2:?need run tag}"
PORT="${3:-8879}"

RUNDIR="$SCAF/runs/${GAME}_${TAG}"
WS="$RUNDIR/workspace"
mkdir -p "$RUNDIR"
rm -rf "$WS"
cp -R "$SCAF/src/agent/workspace_init" "$WS"

# convenience wrappers so the coding agent runs everything with the right python + server URL
cat > "$WS/g" <<WRAP
#!/bin/bash
# game client wrapper: ./g move ACTION2   |   ./g move ACTION6 --x 3 --y 4   |   ./g status
export GAME_SERVER_URL="http://127.0.0.1:${PORT}"
exec "$PY" "\$(dirname "\$0")/client/client.py" "\$@"
WRAP
cat > "$WS/verify" <<WRAP
#!/bin/bash
exec "$PY" "\$(dirname "\$0")/verify_world_model.py" "\$@"
WRAP
cat > "$WS/verifyplan" <<WRAP
#!/bin/bash
exec "$PY" "\$(dirname "\$0")/verify_main_planner.py" "\$@"
WRAP
cat > "$WS/plan" <<WRAP
#!/bin/bash
# ./plan --from-current | --from-initial N | --from-attempt N A S
exec "$PY" "\$(dirname "\$0")/run_main_planner.py" "\$@"
WRAP
cat > "$WS/py" <<WRAP
#!/bin/bash
exec "$PY" "\$@"
WRAP
chmod +x "$WS/g" "$WS/verify" "$WS/verifyplan" "$WS/plan" "$WS/py"

# resolve the real OFFLINE game id (e.g. cd82-fb555c5d) for the given alias prefix
REAL_ID=$(cd "$REPO" && PYTHONPATH=src "$PY" - "$GAME" <<'PY'
import sys
from arc_agi import Arcade, OperationMode
c = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
pref = sys.argv[1]
print(next(e.game_id for e in c.get_environments() if e.game_id.startswith(pref)))
PY
)
REAL_ID=$(printf '%s\n' "$REAL_ID" | grep -E '^[a-z0-9]{4}-[a-f0-9]+' | tail -1 | tr -d '[:space:]')
if [ -z "$REAL_ID" ]; then echo "could not resolve game id for $GAME" >&2; exit 1; fi

export ARC_OFFLINE_DIR="$ENVDIR"
export ARC_SERVER_HOST="127.0.0.1"
export ARC_SERVER_PORT="$PORT"
export ARC_SERVER_LOG_PATH="$RUNDIR/server.log"
export GAME_ID_MAPPING_JSON="{\"target\": \"$REAL_ID\"}"

# start server (background)
rm -f "$RUNDIR/server.log"
( cd "$SCAF" && exec "$PY" src/server/server.py ) >"$RUNDIR/server.stdout.log" 2>&1 &
echo $! > "$RUNDIR/server.pid"

# wait for health
for i in $(seq 1 40); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
  sleep 0.25
done
if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "SERVER FAILED TO START — log:" >&2; tail -20 "$RUNDIR/server.stdout.log" >&2; exit 1
fi

# start the game in the workspace client
( cd "$WS" && GAME_SERVER_URL="http://127.0.0.1:$PORT" "$PY" client/client.py start target ) >"$RUNDIR/start.log" 2>&1

echo "RUNDIR=$RUNDIR"
echo "WORKSPACE=$WS"
echo "PORT=$PORT"
echo "REAL_ID=$REAL_ID (hidden as 'target')"
echo "--- start.log tail ---"
tail -8 "$RUNDIR/start.log"
