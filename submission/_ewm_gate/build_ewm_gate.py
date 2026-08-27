#!/usr/bin/env python3
"""Build submission/_ewm_gate/ewm-gate.ipynb — the Kaggle Qwen-27B EWM viability gate.

Self-contained (base64-embeds the whole EWM scaffold + ewm_agent + 3 dev games, the
repo's proven no-external-code-dataset pattern). At runtime the notebook:
  1. installs vLLM (offline, from the wheelhouse dataset) + arc_agi/arcengine (competition wheels)
  2. serves Qwen3.6-27B-FP8 on 127.0.0.1:1234 (OpenAI-compatible), prints tok/s
  3. for each embedded dev game (blind): starts the OFFLINE arc_agi flask server + a fresh
     workspace, runs ewm_agent.py driven by the LOCAL Qwen, scores depth-weighted RHAE
  4. prints a summary table: solve rate + RHAE + tokens/game + tok/s

This answers the ONE open deployment question: can a LOCAL model drive verification-by-execution
offline (vs. the Claude proof of ~25/100 mean). Run instructions in README.md.
"""
import base64, json
from pathlib import Path

REPO = Path("/Users/ahmed/Documents/ArcAGI3")
SCAF = REPO / "scratchpad/ewm_pathb"
OUT = REPO / "submission/_ewm_gate"
GAMES = ["tu93", "sb26", "cd82"]  # movement + 2 click puzzles (Claude got 80 / 27.8 / 14.3)

# ---- collect scaffold files (relative path -> bytes) ----
files: dict[str, bytes] = {}

def add(rel: str, path: Path):
    files[rel] = path.read_bytes()

add("src/server/server.py", SCAF / "src/server/server.py")
for p in (SCAF / "src/agent/workspace_init").rglob("*"):
    if p.is_file() and "__pycache__" not in str(p):
        files["workspace_init/" + str(p.relative_to(SCAF / "src/agent/workspace_init"))] = p.read_bytes()
add("ewm_agent.py", SCAF / "ewm_agent.py")
add("score_run.py", SCAF / "score_run.py")
# dev games (their engine .py + metadata.json) so the gate is self-contained offline
for g in GAMES:
    for p in (REPO / "environment_files" / g).rglob("*"):
        if p.is_file():
            files["environment_files/" + str(p.relative_to(REPO / "environment_files"))] = p.read_bytes()

BLOB = {k: base64.b64encode(v).decode() for k, v in files.items()}

# ---- runtime orchestrator embedded in the notebook ----
GATE_MAIN = r'''
import base64, json, os, shutil, subprocess, sys, time, glob, urllib.request
from pathlib import Path

ROOT = Path("/kaggle/working/ewm")
ENVDIR = str(ROOT / "environment_files")
PY = sys.executable
GAMES = %(games)s
QWEN_PORT = 1234
BASE_URL = f"http://127.0.0.1:{QWEN_PORT}/v1"

def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, text=True, capture_output=True, **kw)

def wait_url(url, timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=5); return True
        except Exception:
            time.sleep(3)
    return False

# ---- 1. locate + serve Qwen3.6-27B-FP8 ----
def find_model():
    cands = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/config.json", recursive=True)
             if ("qwen" in p.lower() and "27b" in p.lower()) or "vrfai-qwen3-6-27b" in p.lower()]
    if not cands:  # looser
        cands = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/config.json", recursive=True) if "qwen" in p.lower()]
    assert cands, "Qwen model dir not found under /kaggle/input"
    return sorted(cands, key=len)[0]

MODEL = find_model()
print("[gate] Qwen dir:", MODEL, flush=True)
vlog = open("/kaggle/working/vllm.log", "w")
vproc = subprocess.Popen(
    [PY, "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL,
     "--served-model-name", "qwen", "--host", "127.0.0.1", "--port", str(QWEN_PORT),
     "--max-model-len", "65536", "--gpu-memory-utilization", "0.92",
     # RTX Pro 6000 = ONE 97GB GPU -> TP=1 (copied from the duck's PROVEN setup_commands.json,
     # which scored 0.92 live). TP=4 was for the old nvidiaL4x4 plan and CANNOT serve here.
     # --enforce-eager dropped: it throttles generation, and this gate is a TOKENS-IN-1h test.
     "--tensor-parallel-size", os.environ.get("EWM_TP", "1")],
    stdout=vlog, stderr=subprocess.STDOUT)
print("[gate] serving Qwen (this can take several minutes to load FP8 weights)...", flush=True)
if not wait_url(f"{BASE_URL}/models", 2400):
    print(open("/kaggle/working/vllm.log").read()[-3000:]); raise SystemExit("vLLM failed to serve")
print("[gate] Qwen SERVED. Models:", json.loads(urllib.request.urlopen(f"{BASE_URL}/models").read())["data"][0]["id"], flush=True)

# throughput smoke
t0 = time.time()
body = json.dumps({"model": "qwen", "messages": [{"role": "user",
        "content": "Write a Python function flood_fill(grid, r, c, color) that 4-connected flood fills. Code only."}],
        "max_tokens": 400, "temperature": 0.2}).encode()
req = urllib.request.Request(f"{BASE_URL}/chat/completions", data=body, headers={"Content-Type": "application/json"})
d = json.loads(urllib.request.urlopen(req, timeout=300).read())
nt = d["usage"]["completion_tokens"]; dt = time.time() - t0
print(f"[gate] throughput: {nt/dt:.1f} tok/s ({nt} tok in {dt:.1f}s)", flush=True)

# ---- 2. per-game: OFFLINE flask server + workspace + agent ----
import importlib
try:
    import flask  # noqa
except Exception:
    subprocess.run([PY, "-m", "pip", "install", "--no-index",
                    "--find-links", os.path.dirname(glob.glob('/kaggle/input/**/flask*.whl', recursive=True)[0]),
                    "flask"], check=False)

def real_game_id(game):
    from arc_agi import Arcade, OperationMode
    c = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=ENVDIR)
    return next(e.game_id for e in c.get_environments() if e.game_id.startswith(game))

def run_game(game, port):
    code = {"tu93": "gT", "sb26": "gS", "cd82": "gC"}.get(game, game)
    runroot = ROOT / f"run_{code}"; ws = runroot / "workspace"
    if ws.exists(): shutil.rmtree(ws)
    shutil.copytree(ROOT / "workspace_init", ws)
    # wrappers
    for name, tgt in [("g", "client/client.py"), ("verify", "verify_world_model.py"),
                      ("verifyplan", "verify_main_planner.py"), ("plan", "run_main_planner.py")]:
        (ws / name).write_text(f'#!/bin/bash\nexport GAME_SERVER_URL="http://127.0.0.1:{port}"\nexec "{PY}" "$(dirname "$0")/{tgt}" "$@"\n')
        os.chmod(ws / name, 0o755)
    rid = real_game_id(game)
    env = {**os.environ, "ARC_OFFLINE_DIR": ENVDIR, "ARC_SERVER_HOST": "127.0.0.1",
           "ARC_SERVER_PORT": str(port), "ARC_SERVER_LOG_PATH": str(runroot / "server.log"),
           "GAME_ID_MAPPING_JSON": json.dumps({"target": rid})}
    (runroot / "server.log").unlink(missing_ok=True)
    slog = open(runroot / "server.stdout.log", "w")
    sp = subprocess.Popen([PY, str(ROOT / "src/server/server.py")], env=env, stdout=slog, stderr=subprocess.STDOUT, cwd=str(ROOT))
    if not wait_url(f"http://127.0.0.1:{port}/health", 60):
        print(f"[gate:{game}] server failed"); print(open(runroot / "server.stdout.log").read()[-1500:]); sp.kill(); return None
    subprocess.run([PY, str(ws / "client/client.py"), "start", "target"],
                   env={**env, "GAME_SERVER_URL": f"http://127.0.0.1:{port}"}, cwd=str(ws), check=True,
                   capture_output=True)
    # run the local-model agent loop
    print(f"[gate:{game}] running Qwen-driven EWM agent (code {code})...", flush=True)
    r = subprocess.run([PY, str(ROOT / "ewm_agent.py"), "--workspace", str(ws),
                        "--base-url", BASE_URL, "--model", "qwen", "--port", str(port),
                        "--max-turns", "160", "--max-ctx-chars", "110000", "--obs-cap", "4600",
                        "--max-tokens", "4096", "--temperature", "0.3"],
                       capture_output=True, text=True, timeout=60 * 60)
    sp.kill()
    stats = {}
    for line in r.stdout.strip().splitlines()[::-1]:
        if line.strip().startswith("{"):
            try: stats = json.loads(line); break
            except Exception: pass
    # score
    sys.path.insert(0, str(ROOT))
    import score_run
    # score_run reads baselines from a REPO path; point it at our env dir
    score_run.REPO = ROOT
    sc = score_run.score_game(ws / "client/session", game)
    return {"game": game, "code": code, **sc, "agent": stats}

results = []
for i, g in enumerate(GAMES):
    try:
        res = run_game(g, 8890 + i)
        if res: results.append(res)
        print(f"[gate:{g}] RHAE={res and res['game_score']}  levels={res and res['levels_solved']}/{res and res['n_levels']}  agent={res and res['agent']}", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc(); print(f"[gate:{g}] ERROR {e}", flush=True)

print("\n================ EWM LOCAL-BRAIN GATE SUMMARY (Qwen3.6-27B) ================", flush=True)
print(f"{'game':6} {'levels':8} {'RHAE/100':9} {'out_tok':8} {'sec':6}", flush=True)
tot = 0.0
for r in results:
    a = r.get("agent", {})
    print(f"{r['game']:6} {str(r['levels_solved'])+'/'+str(r['n_levels']):8} {r['game_score']:<9} {a.get('completion_tokens','?'):<8} {a.get('seconds','?')}", flush=True)
    tot += r["game_score"]
print(f"\nMEAN RHAE over {len(results)} games: {tot/max(len(results),1):.2f}/100", flush=True)
print("Claude-brain reference on these: tu93 80.0, sb26 27.8, cd82 14.3 (mean 40.7 on this subset).", flush=True)
print("GATE: Qwen mean >~10 = local-brain deployment is viable -> build selective-EWM submission.", flush=True)
vproc.kill()
'''  % {"games": json.dumps(GAMES)}

# ---- unpack cell ----
UNPACK = r'''
import base64, os, json
from pathlib import Path
ROOT = Path("/kaggle/working/ewm"); ROOT.mkdir(parents=True, exist_ok=True)
BLOB = json.loads(open("/kaggle/working/_ewm_blob.json").read())
for rel, b64 in BLOB.items():
    p = ROOT / rel; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(base64.b64decode(b64))
print(f"[gate] unpacked {len(BLOB)} scaffold files to {ROOT}")
'''

INSTALL = r'''
import glob, os, subprocess, sys
# arc_agi + arcengine from the competition wheels (offline)
wheels = "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels"
if os.path.isdir(wheels):
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--find-links", wheels,
                    "arc-agi", "arcengine"], check=False)
# vLLM from the h100 wheelhouse (offline)
wh = glob.glob("/kaggle/input/**/arc3-vllm-h100-wheelhouse*", recursive=True)
lock = glob.glob("/kaggle/input/**/requirements.lock", recursive=True)
if wh:
    if lock:
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--find-links", wh[0],
                        "-r", lock[0]], check=False)
    else:
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--find-links", wh[0], "vllm"], check=False)
print("[gate] deps installed")
'''

# ---- assemble notebook ----
def code_cell(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)}

# write the blob to a file cell first (too big to inline safely as a py literal)
blob_cell_src = "import json\nopen('/kaggle/working/_ewm_blob.json','w').write(%r)\nprint('blob written')\n" % json.dumps(BLOB)

nb = {
    "cells": [
        {"cell_type": "markdown", "metadata": {},
         "source": ["# EWM local-brain viability gate (Qwen3.6-27B)\n",
                    "Serves Qwen offline and runs the verification-by-execution agent loop on 3 blind dev games. ",
                    "Answers: can a LOCAL model drive the loop (vs the Claude proof, mean ~25/100)?\n"]},
        code_cell(blob_cell_src),
        code_cell(UNPACK),
        code_cell(INSTALL),
        code_cell(GATE_MAIN),
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 5,
}

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "ewm-gate.ipynb").write_text(json.dumps(nb, indent=1))
meta = {
    "id": "ahmedmobasher86/arc-agi-3-ewm-gate",
    "title": "arc-agi-3-ewm-gate",
    "code_file": "ewm-gate.ipynb",
    "language": "python", "kernel_type": "notebook",
    "is_private": True, "enable_gpu": True, "enable_internet": False,
    "dataset_sources": ["driessmit1/arc3-vllm-h100-wheelhouse-v3",
                        "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"],
    "competition_sources": ["arc-prize-2026-arc-agi-3"],
    "kernel_sources": [], "model_sources": [],
    # NOTE: machine_shape is INERT — the accelerator is set by the CLI flag at push time:
    #   kaggle kernels push -p . --accelerator NvidiaRtxPro6000
    # (verified 2026-07-18 by GPU probe: RTX PRO 6000 Blackwell, 97887 MiB)
    "machine_shape": "NvidiaRtxPro6000",
}
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print(f"wrote {OUT/'ewm-gate.ipynb'} ({(OUT/'ewm-gate.ipynb').stat().st_size//1024} KB, {len(BLOB)} embedded files)")
print(f"wrote {OUT/'kernel-metadata.json'}")
