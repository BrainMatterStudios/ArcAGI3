"""Validate the notebook's run_game orchestration path locally against an MLX endpoint,
building the exact /kaggle/working/ewm layout the notebook expects."""
import json, os, shutil, subprocess, sys, time, urllib.request
from pathlib import Path

SCAF = Path("/Users/ahmed/Documents/ArcAGI3/scratchpad/ewm_pathb")
REPO = Path("/Users/ahmed/Documents/ArcAGI3")
ROOT = SCAF / "_gate_test"
PY = sys.executable
BASE_URL = "http://127.0.0.1:8081/v1"
MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"

# build the notebook's ROOT layout
if ROOT.exists(): shutil.rmtree(ROOT)
(ROOT / "src/server").mkdir(parents=True)
shutil.copy(SCAF / "src/server/server.py", ROOT / "src/server/server.py")
shutil.copytree(SCAF / "src/agent/workspace_init", ROOT / "workspace_init")
shutil.copy(SCAF / "ewm_agent.py", ROOT / "ewm_agent.py")
shutil.copy(SCAF / "score_run.py", ROOT / "score_run.py")
for g in ["tu93"]:
    shutil.copytree(REPO / "environment_files" / g, ROOT / "environment_files" / g)

ENVDIR = str(ROOT / "environment_files")
sys.path.insert(0, str(ROOT))

def wait_url(u, t=60):
    t0 = time.time()
    while time.time() - t0 < t:
        try: urllib.request.urlopen(u, timeout=5); return True
        except Exception: time.sleep(1)
    return False

def real_game_id(game):
    from arc_agi import Arcade, OperationMode
    c = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=ENVDIR)
    return next(e.game_id for e in c.get_environments() if e.game_id.startswith(game))

game, port, code = "tu93", 8899, "gT"
runroot = ROOT / f"run_{code}"; ws = runroot / "workspace"
shutil.copytree(ROOT / "workspace_init", ws)
for name, tgt in [("g", "client/client.py"), ("verify", "verify_world_model.py"),
                  ("verifyplan", "verify_main_planner.py"), ("plan", "run_main_planner.py")]:
    (ws / name).write_text(f'#!/bin/bash\nexport GAME_SERVER_URL="http://127.0.0.1:{port}"\nexec "{PY}" "$(dirname "$0")/{tgt}" "$@"\n')
    os.chmod(ws / name, 0o755)
rid = real_game_id(game)
env = {**os.environ, "ARC_OFFLINE_DIR": ENVDIR, "ARC_SERVER_HOST": "127.0.0.1",
       "ARC_SERVER_PORT": str(port), "ARC_SERVER_LOG_PATH": str(runroot / "server.log"),
       "GAME_ID_MAPPING_JSON": json.dumps({"target": rid})}
runroot.mkdir(exist_ok=True)
sp = subprocess.Popen([PY, str(ROOT / "src/server/server.py")], env=env,
                      stdout=open(runroot / "s.log", "w"), stderr=subprocess.STDOUT, cwd=str(ROOT))
assert wait_url(f"http://127.0.0.1:{port}/health", 60), open(runroot / "s.log").read()[-800:]
subprocess.run([PY, str(ws / "client/client.py"), "start", "target"],
               env={**env, "GAME_SERVER_URL": f"http://127.0.0.1:{port}"}, cwd=str(ws), check=True, capture_output=True)
print("server + game started; running agent 5 turns vs MLX...", flush=True)
r = subprocess.run([PY, str(ROOT / "ewm_agent.py"), "--workspace", str(ws), "--base-url", BASE_URL,
                    "--model", MODEL, "--port", str(port), "--max-turns", "5",
                    "--max-ctx-chars", "20000", "--max-tokens", "1024"], capture_output=True, text=True, timeout=600)
sp.kill()
last = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "(none)"
print("agent last line:", last)
import score_run
score_run.REPO = ROOT
sc = score_run.score_game(ws / "client/session", game)
print("SCORE OK:", json.dumps({k: sc[k] for k in ("game", "n_levels", "levels_solved", "game_score")}))
print("=== ORCHESTRATION PATH VALIDATED ===")
shutil.rmtree(ROOT)
