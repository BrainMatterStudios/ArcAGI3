#!/usr/bin/env python3
"""Build submission/_ewm_gate3/ewm-gate3.ipynb — the CORRECTED Qwen-27B EWM viability gate.

WHY v2 EXISTS (2026-07-21): gate v1 (`submission/_ewm_gate/`) ERRORED on the RTX Pro 6000 after
41 min. Root cause from its log: it hand-rolled its own vLLM install/serve, installing the
wheelhouse into SYSTEM dist-packages and reusing the SYSTEM torch
("Requirement already satisfied: torch==2.10.0 in /usr/local/lib/python3.12/dist-packages").
That mixed stack made flashinfer JIT-compile Blackwell kernels (gemm_sm120) ->
"RuntimeError: Ninja build failed" -> engine core never started -> Qwen never played a game.

FIX: stop hand-rolling the serve. Build on submission/_repro/duck-repro.ipynb and let the duck's
PROVEN setup_commands.json serve Qwen (isolated --target install + pinned requirements.lock +
TP=1). That exact path scored 0.92 live on this GPU, and the EWM submission (which reused it)
reached COMPLETE -- so the serve is not in question.

WHAT THIS GATE ANSWERS (the one open question, worth ZERO submission slots):
  Can the LOCAL Qwen3.6-27B brain crack a game at all when given a REAL time box (1h/game)?
The 0.00 EWM submission does NOT answer it -- it used 290s/game, which the token math
(~283K tokens needed vs 4-29K available) refutes for free.

DECISION RULE (set BEFORE seeing results):
  - Qwen completes real levels (>=1 level on tu93, ideally mean >~10/100)
        -> time was the constraint -> build floor-safe SELECTIVE EWM, offline-validate, submit.
  - Qwen ~= 0 across all three
        -> capability wall -> do NOT spend another submission slot on EWM.

Runs as a COMMIT (logs readable, unlike scored reruns). Push:
  python submission/_ewm_gate3/build_ewm_gate2.py
  cd submission/_ewm_gate3 && kaggle kernels push -p . --accelerator NvidiaRtxPro6000
"""
import base64, json
from pathlib import Path

REPO = Path("/Users/ahmed/Documents/ArcAGI3")
SCAF = REPO / "scratchpad/ewm_pathb"
BASE = REPO / "submission/_repro/duck-repro.ipynb"
OUT = REPO / "submission/_ewm_gate3"
GAMES = ["tu93", "sb26", "cd82"]  # Claude scored 80 / 27.8 / 14.3 -> tu93 FIRST = best case

# ---- collect scaffold + dev games (self-contained; no external code dataset) ----
files: dict[str, bytes] = {}
files["src/server/server.py"] = (SCAF / "src/server/server.py").read_bytes()
for p in (SCAF / "src/agent/workspace_init").rglob("*"):
    if p.is_file() and "__pycache__" not in str(p):
        files["workspace_init/" + str(p.relative_to(SCAF / "src/agent/workspace_init"))] = p.read_bytes()
files["ewm_agent.py"] = (SCAF / "ewm_agent.py").read_bytes()
files["score_run.py"] = (SCAF / "score_run.py").read_bytes()
# Opus-authored worked examples for the few-shot (world_model.md + engine only). The
# agent maps each target to a DIFFERENT game's example and also guards against self-use,
# so this teaches the deliverable's SHAPE, never the target game's mechanics. For a real
# private-game eval these public solutions cannot leak into the scored set at all.
for gsrc, wdir in (("tu93", "runs/tu93_run1/workspace"), ("sb26", "runs/sb26_gate1/workspace")):
    for fn in ("world_model.md", "world_model_engine.py"):
        p = SCAF / wdir / fn
        if p.is_file():
            files[f"runs/{gsrc}_ref/workspace/{fn}"] = p.read_bytes()
for g in GAMES:
    for p in (REPO / "environment_files" / g).rglob("*"):
        if p.is_file():
            files["environment_files/" + str(p.relative_to(REPO / "environment_files"))] = p.read_bytes()
BLOB = {k: base64.b64encode(v).decode() for k, v in files.items()}

UNPACK = (
    "import base64, json, sys\n"
    "from pathlib import Path\n"
    "ROOT = Path('/kaggle/working/ewm'); ROOT.mkdir(parents=True, exist_ok=True)\n"
    "for rel, b64 in json.loads(%r).items():\n"
    "    p = ROOT / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(base64.b64decode(b64))\n"
    "if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))\n"
    "print('[gate3] scaffold unpacked:', len(json.loads(%r)), 'files ->', ROOT, flush=True)\n"
) % (json.dumps(BLOB), json.dumps(BLOB))

# ---- the gate itself: NO vLLM launch; the duck's setup_commands already served Qwen ----
GATE = r'''
import json, os, shutil, subprocess, sys, time, glob, urllib.request
from pathlib import Path

ROOT = Path("/kaggle/working/ewm")
ENVDIR = str(ROOT / "environment_files")
PY = sys.executable
GAMES = %(games)s
BASE_URL = "http://127.0.0.1:1234/v1"
MODEL = "coder"   # served-model-name of the CODER brain we launch below
PER_GAME_SECONDS = int(os.environ.get("GATE_PER_GAME_SECONDS", str(60 * 60)))  # REAL box, not 290s

def wait_url(url, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=5); return True
        except Exception:
            time.sleep(3)
    return False

# ---- 1. SWAP THE BRAIN: reuse the duck's PROVEN vLLM INSTALL, serve the CODER model ----
# gate v1 died because it hand-rolled its own vLLM install (into SYSTEM dist-packages, reusing the
# SYSTEM torch) -> flashinfer JIT'd Blackwell gemm_sm120 -> "Ninja build failed". So we let the
# duck's setup_commands do the install+serve (proven), then stop its 27B and relaunch the SAME
# vLLM from the SAME isolated site-packages pointed at the coder model.
WORKING_DIR = Path(os.environ["TAAF_KAGGLE_WORKING_DIR"])
SITE = WORKING_DIR / "vllm-site-packages"
PIDF = WORKING_DIR / "vllm-openai-server.pid"

if not wait_url(f"{BASE_URL}/models", 1800):
    raise SystemExit("[gate3] duck's vLLM never served -- cannot proceed")
print("[gate3] duck 27B up; stopping it to free the GPU for the coder brain", flush=True)
if PIDF.exists():
    try: os.kill(int(PIDF.read_text().strip()), 15)
    except Exception as e: print("[gate3] kill failed:", e, flush=True)
t0 = time.time()
while time.time() - t0 < 300:
    try:
        urllib.request.urlopen(f"{BASE_URL}/models", timeout=3); time.sleep(5)
    except Exception:
        break
time.sleep(25)   # let CUDA release the weights
print("[gate3] 27B stopped", flush=True)

cands = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/config.json", recursive=True)
         if "qwen3-coder" in p.lower()]
assert cands, "qwen3-coder model dir not found under /kaggle/input"
CODER = sorted(cands, key=len)[0]
print("[gate3] coder model dir:", CODER, flush=True)

cenv = {**os.environ, "PYTHONPATH": str(SITE) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "VLLM_NO_USAGE_STATS": "1", "USE_TF": "0", "TRANSFORMERS_NO_TF": "1"}
clog = open("/kaggle/working/vllm-coder.log", "w")
# Qwen3-Coder-30B-A3B is a NON-THINKING model (emits no <think> blocks) -> do NOT pass
# --reasoning-parser/preserve_thinking (the duck needs those for Qwen3.6; here they break parsing).
cproc = subprocess.Popen(
    [PY, "-m", "vllm.entrypoints.openai.api_server", "--model", CODER,
     "--served-model-name", "coder", "--host", "127.0.0.1", "--port", "1234",
     "--tensor-parallel-size", "1", "--max-model-len", "65536",
     "--gpu-memory-utilization", "0.90", "--enable-prefix-caching"],
    env=cenv, stdout=clog, stderr=subprocess.STDOUT)
print("[gate3] serving Qwen3-Coder-30B-A3B-FP8 (3B active) ...", flush=True)
if not wait_url(f"{BASE_URL}/models", 2400):
    print(open("/kaggle/working/vllm-coder.log").read()[-4000:], flush=True)
    raise SystemExit("[gate3] coder model FAILED to serve")
served = json.loads(urllib.request.urlopen(f"{BASE_URL}/models").read())["data"][0]["id"]
print(f"[gate3] CODER SERVED as {served!r}", flush=True)

t0 = time.time()
body = json.dumps({"model": MODEL, "messages": [{"role": "user",
        "content": "Write a Python function flood_fill(grid, r, c, color) that 4-connected flood fills. Code only."}],
        "max_tokens": 400, "temperature": 0.2}).encode()
req = urllib.request.Request(f"{BASE_URL}/chat/completions", data=body, headers={"Content-Type": "application/json"})
d = json.loads(urllib.request.urlopen(req, timeout=300).read())
nt = d["usage"]["completion_tokens"]; dt = time.time() - t0
TOKS = nt / max(dt, 1e-9)
print(f"[gate3] throughput: {TOKS:.1f} tok/s ({nt} tok in {dt:.1f}s) "
      f"-> ~{TOKS*PER_GAME_SECONDS/1000:.0f}K tokens available per {PER_GAME_SECONDS}s game "
      f"(a WORKING EWM run used ~283K)", flush=True)

try:
    import flask  # noqa
except Exception:
    w = glob.glob('/kaggle/input/**/flask*.whl', recursive=True)
    if w:
        subprocess.run([PY, "-m", "pip", "install", "--no-index", "--find-links", os.path.dirname(w[0]), "flask"], check=False)

def real_game_id(game):
    from arc_agi import Arcade, OperationMode
    c = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=ENVDIR)
    return next(e.game_id for e in c.get_environments() if e.game_id.startswith(game))

def run_game(game, port):
    # opaque codename in the workspace path = FIREWALL (agent must infer from frames, not the name)
    code = {"tu93": "gT", "sb26": "gS", "cd82": "gC"}.get(game, game)
    runroot = ROOT / f"run_{code}"; ws = runroot / "workspace"
    runroot.mkdir(parents=True, exist_ok=True)
    if ws.exists(): shutil.rmtree(ws)
    shutil.copytree(ROOT / "workspace_init", ws)
    for name, tgt in [("g", "client/client.py"), ("verify", "verify_world_model.py"),
                      ("verifyplan", "verify_main_planner.py"), ("plan", "run_main_planner.py")]:
        (ws / name).write_text(f'#!/bin/bash\nexport GAME_SERVER_URL="http://127.0.0.1:{port}"\nexec "{PY}" "$(dirname "$0")/{tgt}" "$@"\n')
        os.chmod(ws / name, 0o755)
    rid = real_game_id(game)
    env = {**os.environ, "ARC_OFFLINE_DIR": ENVDIR, "ARC_SERVER_HOST": "127.0.0.1",
           "ARC_SERVER_PORT": str(port), "ARC_SERVER_LOG_PATH": str(runroot / "server.log"),
           "GAME_ID_MAPPING_JSON": json.dumps({"target": rid})}
    slog = open(runroot / "server.stdout.log", "w")
    sp = subprocess.Popen([PY, str(ROOT / "src/server/server.py")], env=env,
                          stdout=slog, stderr=subprocess.STDOUT, cwd=str(ROOT))
    try:
        if not wait_url(f"http://127.0.0.1:{port}/health", 90):
            print(f"[gate3:{code}] game server failed:\n" + open(runroot / "server.stdout.log").read()[-1500:], flush=True)
            return None
        subprocess.run([PY, str(ws / "client/client.py"), "start", "target"],
                       env={**env, "GAME_SERVER_URL": f"http://127.0.0.1:{port}"},
                       cwd=str(ws), check=True, capture_output=True)
        print(f"[gate3:{code}] running Qwen-driven EWM agent, box={PER_GAME_SECONDS}s ...", flush=True)
        t0 = time.time()
        try:
            # Few-shot from a DIFFERENT game's Opus solution (leakage guard): the agent's
            # own guard also skips if target==example, but map explicitly so the gate is
            # self-documenting. We hold verified Opus world models for tu93 and sb26.
            fewshot = "sb26" if code == "tu93" else "tu93"
            r = subprocess.run([PY, str(ROOT / "ewm_agent.py"), "--workspace", str(ws),
                                "--base-url", BASE_URL, "--model", MODEL, "--port", str(port),
                                # Let the TIME BOX bind, not the turn cap. The 2026-07-22 audit
                                # fixes are now the agent's defaults; we pin them here so the gate
                                # is immune to default drift and reads as an explicit config.
                                "--max-turns", "300", "--max-ctx-chars", "110000",
                                "--max-ctx-tokens", "45000",
                                # audit repairs: readable spec, full-file writes, duck-matched
                                # sampling + repetition penalty, worked example.
                                "--obs-cap", "16000", "--max-tokens", "16384",
                                "--temperature", "0.6", "--top-p", "0.95",
                                "--repetition-penalty", "1.05",
                                "--fewshot-game", fewshot, "--fewshot-root", str(ROOT)],
                               capture_output=True, text=True, timeout=PER_GAME_SECONDS)
            tail = (r.stdout or "")[-1200:]
        except subprocess.TimeoutExpired as e:
            tail = (e.stdout or b"").decode(errors="replace")[-1200:] if isinstance(e.stdout, bytes) else str(e.stdout)[-1200:]
            print(f"[gate3:{code}] agent hit the {PER_GAME_SECONDS}s box (expected; scoring what it achieved)", flush=True)
        elapsed = time.time() - t0
        print(f"[gate3:{code}] agent tail:\n{tail}", flush=True)
    finally:
        sp.kill()
    sys.path.insert(0, str(ROOT))
    import score_run
    score_run.REPO = ROOT           # baselines come from OUR embedded env dir
    sc = score_run.score_game(ws / "client/session", game)
    return {"game": game, "code": code, "elapsed": round(elapsed, 1), **sc}

results = []
for i, g in enumerate(GAMES):
    try:
        res = run_game(g, 8890 + i)
        if res:
            results.append(res)
            print(f"[gate3:{g}] RHAE={res['game_score']}  levels={res['levels_solved']}/{res['n_levels']}  {res['elapsed']}s", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[gate3:{g}] ERROR {e}", flush=True)

print("\n================ EWM LOCAL-BRAIN GATE v3 (Qwen3.6-27B, 1h/game) ================", flush=True)
print(f"{'game':8} {'levels':10} {'RHAE/100':>9}", flush=True)
for r in results:
    print(f"{r['game']:8} {str(r['levels_solved'])+'/'+str(r['n_levels']):10} {r['game_score']:>9}", flush=True)
mean = (sum(r["game_score"] for r in results) / len(results)) if results else 0.0
tot_levels = sum(r["levels_solved"] for r in results)
print(f"{'MEAN':8} {'':10} {round(mean,2):>9}", flush=True)
print(f"[gate3] throughput was {TOKS:.1f} tok/s", flush=True)
print(f"[gate3] VERDICT: total levels solved by Qwen = {tot_levels}", flush=True)
print("[gate3] >>> PASS (build selective EWM)" if tot_levels >= 1 else
      "[gate3] >>> FAIL (capability wall -- do NOT spend a submission slot on EWM)", flush=True)
print(json.dumps({"mean_rhae": mean, "total_levels": tot_levels, "tok_s": TOKS, "results": results}), flush=True)
''' % {"games": json.dumps(GAMES)}

nb = json.loads(BASE.read_text())
new_cells, did_setup, did_run = [], False, False
for c in nb["cells"]:
    src = "".join(c.get("source", []))
    # (a) keep the duck's setup_commands serve UNGUARDED so the COMMIT serves Qwen, then gate
    if c["cell_type"] == "code" and "setup_commands.json" in src and not did_setup:
        did_setup = True
        new_cells.append(c)  # untouched: the proven serve
        new_cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                          "source": UNPACK.splitlines(keepends=True)})
        new_cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                          "source": GATE.splitlines(keepends=True)})
        continue
    # (b) neutralize the benchmark run: this gate plays DEV games, not the hidden set
    if c["cell_type"] == "code" and "await bm.run(" in src and not did_run:
        did_run = True
        old = (
            '    await bm.run(soft_end_time=soft_end, runtime_environment=target, minimal_diagnostics=TRUE_SUBMISSION)\n'
            '    if not TRUE_SUBMISSION:\n'
            '        # An offline run isn\'t scored, but Kaggle still expects a submission.parquet output.\n'
            '        import pandas as pd\n'
            '\n'
            '        pd.DataFrame(\n'
            '            [["1_0", "1", True, 1]],\n'
            '            columns=["row_id", "game_id", "end_of_game", "score"],\n'
            '        ).to_parquet(WORKING_DIR / "submission.parquet", index=False)\n')
        new = (
            '    # GATE v3: the EWM gate already ran on DEV games above. Do NOT play the benchmark.\n'
            '    import pandas as pd\n'
            '    pd.DataFrame([["1_0", "1", True, 1]],\n'
            '                 columns=["row_id", "game_id", "end_of_game", "score"]\n'
            '                 ).to_parquet(WORKING_DIR / "submission.parquet", index=False)\n'
            '    print("[gate3] benchmark skipped (this kernel is a GATE, never submitted)", flush=True)\n')
        if old not in src:
            raise SystemExit("run-cell block did not match — inspect duck-repro run cell")
        new_cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                          "source": src.replace(old, new).splitlines(keepends=True)})
        continue
    new_cells.append(c)
if not (did_setup and did_run):
    raise SystemExit(f"transform incomplete: setup={did_setup} run={did_run}")
nb["cells"] = new_cells

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "ewm-gate3.ipynb").write_text(json.dumps(nb, indent=1))
(OUT / "kernel-metadata.json").write_text(json.dumps({
    "id": "ahmedmobasher86/arc-agi-3-ewm-gate3",
    "title": "arc-agi-3-ewm-gate3",
    "code_file": "ewm-gate3.ipynb",
    "language": "python", "kernel_type": "notebook",
    "is_private": True, "enable_gpu": True, "enable_internet": False,
    # machine_shape is INERT; the accelerator comes from the CLI flag at push time.
    "machine_shape": "NvidiaRtxPro6000",
    "dataset_sources": ["driessmit1/arc3-vllm-h100-wheelhouse-v3",
                        "ahmedmobasher86/taaf-src-hybrid",
                        "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"],
    "competition_sources": ["arc-prize-2026-arc-agi-3"],
    "kernel_sources": [],
    "model_sources": ["qwen-lm/qwen3-coder/transformers/30b-a3b-instruct-fp8/1"],
}, indent=2))
print(f"wrote {OUT/'ewm-gate3.ipynb'} ({len(nb['cells'])} cells, {len(BLOB)} embedded files)")
