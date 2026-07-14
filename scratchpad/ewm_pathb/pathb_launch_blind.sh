#!/bin/bash
# BLIND launcher: like pathb_launch.sh but the run dir uses an OPAQUE codename so the
# coding agent cannot learn the real game from its own workspace path (the firewall leak
# that let cd82/dc22 read game source). The codename->game map is kept in a private file
# OUTSIDE the workspace for scoring only.
#
# Usage: pathb_launch_blind.sh <real_game e.g. cd82> <codename e.g. gA> <run_tag> [port]
set -euo pipefail
REPO="/Users/ahmed/Documents/ArcAGI3"
SCAF="$REPO/scratchpad/ewm_pathb"
PY="$REPO/.venv/bin/python"
ENVDIR="$REPO/environment_files"

GAME="${1:?real game}"; CODE="${2:?codename}"; TAG="${3:?run tag}"; PORT="${4:-8890}"

RUNROOT="$SCAF/runs_blind/${CODE}_${TAG}"
WS="$RUNROOT/workspace"
mkdir -p "$RUNROOT"; rm -rf "$WS"
cp -R "$SCAF/src/agent/workspace_init" "$WS"
# private map (agent must not read this; it lives above the workspace)
echo "{\"codename\":\"$CODE\",\"game\":\"$GAME\",\"port\":$PORT}" > "$RUNROOT/.truth.json"

# wrappers (identical to non-blind)
for name in g verify verifyplan plan py; do :; done
cat > "$WS/g" <<WRAP
#!/bin/bash
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
exec "$PY" "\$(dirname "\$0")/run_main_planner.py" "\$@"
WRAP
cat > "$WS/py" <<WRAP
#!/bin/bash
exec "$PY" "\$@"
WRAP
chmod +x "$WS/g" "$WS/verify" "$WS/verifyplan" "$WS/plan" "$WS/py"

REAL_ID=$(cd "$REPO" && PYTHONPATH=src "$PY" - "$GAME" <<'PY' 2>/dev/null
import sys
from arc_agi import Arcade, OperationMode
c = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
print(next(e.game_id for e in c.get_environments() if e.game_id.startswith(sys.argv[1])))
PY
)
REAL_ID=$(printf '%s\n' "$REAL_ID" | grep -E '^[a-z0-9]{4}-[a-f0-9]+' | tail -1 | tr -d '[:space:]')
[ -z "$REAL_ID" ] && { echo "no id for $GAME" >&2; exit 1; }

export ARC_OFFLINE_DIR="$ENVDIR" ARC_SERVER_HOST="127.0.0.1" ARC_SERVER_PORT="$PORT"
export ARC_SERVER_LOG_PATH="$RUNROOT/server.log"
export GAME_ID_MAPPING_JSON="{\"target\": \"$REAL_ID\"}"
rm -f "$RUNROOT/server.log"
( cd "$SCAF" && exec "$PY" src/server/server.py ) >"$RUNROOT/server.stdout.log" 2>&1 &
echo $! > "$RUNROOT/server.pid"
for i in $(seq 1 40); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 0.25; done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || { echo "SERVER FAILED" >&2; tail -20 "$RUNROOT/server.stdout.log" >&2; exit 1; }
( cd "$WS" && GAME_SERVER_URL="http://127.0.0.1:$PORT" "$PY" client/client.py start target ) >"$RUNROOT/start.log" 2>&1
echo "CODE=$CODE WS=$WS PORT=$PORT RUNROOT=$RUNROOT"
