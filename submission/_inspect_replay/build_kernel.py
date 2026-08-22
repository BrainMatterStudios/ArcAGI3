#!/usr/bin/env python3
"""build_kernel.py — assemble the arc3-inspect-replay notebook.

INSPECTION-PROMPT REPLAY falsifier (docs/RESEARCH-2026-08-22 Addendum 2):
serve Qwen3.5-9B (then 4B if time) on the RTX Pro 6000, replay 136 recorded
27B inspection-call contexts, execute emitted snippets in the rehydrated
sandbox, grade vs the pre-registered kill criteria. NO games, NO submission.

Usage: .venv/bin/python submission/_inspect_replay/build_kernel.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
NB_PATH = HERE / "arc3-inspect-replay.ipynb"

MD_HEADER = """\
# arc3-inspect-replay — INSPECTION-PROMPT REPLAY falsifier (NO games)

Gate on the inspection-routing cascade lever (RESEARCH-2026-08-22 Addendum 2).
Battery: 136 recorded 27B inspection calls (position>=1, 11 games, 6 corpora),
each with the reconstructed chat context, rehydrated sandbox state, and the
27B's recorded snippet output as ground truth (96.5% sandbox fidelity gate at
extraction).

**Pre-registered kill criteria** (either kills the cascade):
- small-model probe-code execution-error rate > 1.5x the 27B's recorded rate
  on the same samples (dead completions count as failures);
- < 70% key-fact recovery of the 27B's printed probe outputs.

| Phase | What | Budget |
|---|---|---|
| boot | GPU assert + input audit | ~2 min |
| 0 | vLLM 0.19 wheelhouse install | ~8 min |
| 1 | serve Qwen3.5-9B bf16 (production parser flags) + smoke | ~8 min |
| 2 | 9B battery (deadline 95 min from start) | ~60-75 min |
| 3 | swap to Qwen3.5-4B + battery (deadline 138 min) | ~30 min |
| 4 | verdict table vs kill criteria | ~1 min |

Runtime cap 2.5 h. Sampling matches go-forward production: temp 0.6,
top_p 0.95, top_k 20, thinking on, reasoning_effort=medium (chat-template
kwarg; the 3.5 template ignores unknown kwargs), max_tokens capped at 16384
as a runtime guard (production sends none).
"""

CELL_IMPORTS = r'''import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

NOTEBOOK_START = time.time()
HARD_DEADLINE = NOTEBOOK_START + 148 * 60      # absolute stop
BATTERY_9B_DEADLINE = NOTEBOOK_START + 95 * 60
BATTERY_4B_DEADLINE = NOTEBOOK_START + 138 * 60
WORKING = Path("/kaggle/working")
RESULTS_PATH = WORKING / "inspect_replay_results.json"
print("start", time.strftime("%H:%M:%S"))
'''

CELL_GPU_ASSERT = r'''# ---------------- FAIL-FAST GPU ASSERT (P100/T4 rehoming dies in minutes) ---
_q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                    capture_output=True, text=True)
GPU_NAME = (_q.stdout or "").strip()
print(f"inspect-replay: GPU = {GPU_NAME!r} (rc={_q.returncode})")
if _q.returncode != 0 or not GPU_NAME:
    raise RuntimeError("WRONG-GPU: nvidia-smi failed — no usable GPU. Aborting fast.")
if "6000" not in GPU_NAME.upper() or "RTX" not in GPU_NAME.upper():
    raise RuntimeError(f"WRONG-GPU: need RTX Pro 6000, got {GPU_NAME!r}. Aborting fast.")
'''

CELL_INPUTS = r'''# ---------------- input audit ------------------------------------------------
def _first_existing(cands):
    for c in cands:
        if Path(c).exists():
            return Path(c)
    return None

ASSETS = _first_existing([
    "/kaggle/input/arc3-inspect-replay-assets",
    "/kaggle/input/datasets/ahmedmobasher86/arc3-inspect-replay-assets",
])
WHEELHOUSE = _first_existing([
    "/kaggle/input/arc3-vllm-h100-wheelhouse-v3",
    "/kaggle/input/datasets/driessmit1/arc3-vllm-h100-wheelhouse-v3",
])
MODEL_9B = _first_existing([
    "/kaggle/input/models/danbth/qwen3-5-9b/transformers/default/1",
    "/kaggle/input/qwen3-5-9b/transformers/default/1",
])
DS_4B = _first_existing([
    "/kaggle/input/qwen3-5-4b/Qwen3.5-4B",
    "/kaggle/input/datasets/limamateus/qwen3-5-4b/Qwen3.5-4B",
])
print("assets    :", ASSETS)
print("wheelhouse:", WHEELHOUSE)
print("model 9B  :", MODEL_9B)
print("model 4B  :", DS_4B)
if ASSETS is None or WHEELHOUSE is None:
    raise RuntimeError("Missing required dataset mounts.")
if MODEL_9B is None:
    # tolerate alternate mount roots
    hits = list(Path("/kaggle/input").rglob("qwen3-5-9b"))
    print("9B search:", hits)
    for h in hits:
        cfgs = list(h.rglob("config.json"))
        if cfgs:
            MODEL_9B = cfgs[0].parent
            break
if MODEL_9B is None:
    raise RuntimeError("Qwen3.5-9B model mount not found.")
if DS_4B is None:
    hits = [p.parent for p in Path("/kaggle/input").rglob("Qwen3.5-4B/config.json")]
    DS_4B = hits[0] if hits else None
print("resolved 9B:", MODEL_9B, "| 4B:", DS_4B)
for tag, p in (("9B", MODEL_9B), ("4B", DS_4B)):
    if p is not None:
        cfg = json.loads((p / "config.json").read_text())
        print(tag, cfg.get("architectures"), "shards:", len(list(p.glob("*.safetensors"))))

os.environ["INSPECT_ASSETS_DIR"] = str(ASSETS)
os.environ["INSPECT_BUNDLE_SRC"] = str(ASSETS)
os.environ["INSPECT_WORKING_DIR"] = str(WORKING)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
'''

CELL_WHEELHOUSE = r'''# ---------------- vLLM 0.19 wheelhouse install (proven serve chain) ---------
SITE_PACKAGES = WORKING / "vllm-site-packages"
requirements = WHEELHOUSE / "requirements.lock"
if not requirements.exists():
    raise FileNotFoundError(f"Missing wheelhouse lock file: {requirements}")
t0 = time.time()
cmd = [sys.executable, "-m", "pip", "install", "--no-index",
       "--find-links", str(WHEELHOUSE), "--requirement", str(requirements),
       "--target", str(SITE_PACKAGES), "--upgrade", "--ignore-installed",
       "--only-binary", ":all:", "--no-compile",
       "--disable-pip-version-check", "--no-warn-conflicts"]
print("installing vLLM wheelhouse ...", flush=True)
r = subprocess.run(cmd, capture_output=True, text=True)
if r.returncode != 0:
    print(r.stdout[-3000:])
    print(r.stderr[-3000:])
    raise RuntimeError("wheelhouse install failed")
print(f"wheelhouse installed in {time.time()-t0:.0f}s")

def vllm_env():
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(SITE_PACKAGES) if not existing else f"{SITE_PACKAGES}{os.pathsep}{existing}"
    env.update({"USE_TF": "0", "TRANSFORMERS_NO_TF": "1",
                "TRANSFORMERS_NO_TORCHVISION": "1", "VLLM_NO_USAGE_STATS": "1"})
    return env
'''

CELL_SERVE = r'''# ---------------- serve helpers ---------------------------------------------
VLLM_HOST, VLLM_PORT = "127.0.0.1", 1234
VLLM_BASE_URL = f"http://{VLLM_HOST}:{VLLM_PORT}/v1"
SERVER_LOG = WORKING / "vllm-openai-server.log"
_server_proc = None

def request_json(url, payload=None, timeout=30):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

def stop_server():
    global _server_proc
    if _server_proc is not None:
        _server_proc.terminate()
        try:
            _server_proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            _server_proc.kill()
        _server_proc = None
        time.sleep(10)

def start_server(model_path, served_name, chat_template=None, wait_s=1200):
    global _server_proc
    stop_server()
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
           "--model", str(model_path), "--served-model-name", served_name,
           "--host", VLLM_HOST, "--port", str(VLLM_PORT),
           "--tensor-parallel-size", "1",
           "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
           "--generation-config", "vllm", "--enable-prefix-caching",
           "--default-chat-template-kwargs", '{"preserve_thinking": true}',
           "--reasoning-parser", "qwen3",
           "--max-model-len", "65536"]
    if chat_template is not None:
        cmd += ["--chat-template", str(chat_template)]
    log = SERVER_LOG.open("w", encoding="utf-8")
    print("starting:", " ".join(cmd), flush=True)
    _server_proc = subprocess.Popen(cmd, env=vllm_env(), stdout=log, stderr=subprocess.STDOUT, text=True)
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if _server_proc.poll() is not None:
            tail = SERVER_LOG.read_text(errors="replace").splitlines()[-40:]
            raise RuntimeError("vLLM server died:\n" + "\n".join(tail))
        try:
            models = request_json(f"{VLLM_BASE_URL}/models", timeout=5)
            print("vLLM ready:", [m.get("id") for m in models.get("data", [])], flush=True)
            return
        except Exception:
            time.sleep(5)
    tail = SERVER_LOG.read_text(errors="replace").splitlines()[-40:]
    raise TimeoutError("vLLM server did not come up:\n" + "\n".join(tail))

def parser_smoke(served_name):
    """Tool-call round trip with production-shaped request. Non-fatal."""
    payload = {
        "model": served_name,
        "messages": [{"role": "user", "content":
                      "Call the python tool with code that prints 2+3."}],
        "tools": [{"type": "function", "function": {
            "name": "python", "description": "Run python code.",
            "parameters": {"type": "object", "properties": {"code": {"type": "string"}},
                           "required": ["code"]}}}],
        "tool_choice": "auto", "temperature": 0.6, "top_p": 0.95, "top_k": 20,
        "max_tokens": 2048,
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "medium"},
    }
    try:
        resp = request_json(f"{VLLM_BASE_URL}/chat/completions", payload, timeout=300)
        msg = resp["choices"][0]["message"]
        ok = bool(msg.get("tool_calls"))
        print(f"parser smoke: tool_calls={ok} finish={resp['choices'][0].get('finish_reason')}")
        if not ok:
            print("  content head:", str(msg.get('content'))[:200])
            print("  reasoning head:", str(msg.get('reasoning_content'))[:200])
        return ok
    except Exception as exc:
        print("parser smoke FAILED:", exc)
        return False
'''

CELL_RUNNER_IMPORT = r'''# ---------------- battery runner ---------------------------------------------
sys.path.insert(0, str(ASSETS))
import runner as inspect_runner
samples, packs = inspect_runner.load_battery()
print(f"battery loaded: {len(samples)} samples, {len(packs)} game packs")
'''

CELL_PHASE_9B = r'''# ---------------- phase 1+2: Qwen3.5-9B --------------------------------------
results = {}
try:
    start_server(MODEL_9B, "Qwen/Qwen3.5-9B")
    smoke_9b = parser_smoke("Qwen/Qwen3.5-9B")
    results = inspect_runner.run_battery(
        "qwen35_9b", VLLM_BASE_URL, "Qwen/Qwen3.5-9B",
        min(BATTERY_9B_DEADLINE, HARD_DEADLINE))
    results.setdefault("qwen35_9b", {}).setdefault("meta", {})["parser_smoke"] = smoke_9b
    inspect_runner._atomic_write(RESULTS_PATH, results)
except Exception as exc:
    print("PHASE-9B FAILED:", type(exc).__name__, exc)
    if SERVER_LOG.exists():
        print("\n".join(SERVER_LOG.read_text(errors="replace").splitlines()[-30:]))
finally:
    stop_server()
'''

CELL_PHASE_4B = r'''# ---------------- phase 3: Qwen3.5-4B (time permitting) ----------------------
try:
    if DS_4B is None:
        print("4B mount missing; skipping 4B arm")
    elif time.time() > BATTERY_4B_DEADLINE - 20 * 60:
        print("not enough budget left for the 4B arm; skipping")
    else:
        tmpl = ASSETS / "qwen35_4b_chat_template.jinja"
        chat_template = tmpl if tmpl.exists() else None
        start_server(DS_4B, "Qwen/Qwen3.5-4B", chat_template=chat_template)
        smoke_4b = parser_smoke("Qwen/Qwen3.5-4B")
        results = inspect_runner.run_battery(
            "qwen35_4b", VLLM_BASE_URL, "Qwen/Qwen3.5-4B",
            min(BATTERY_4B_DEADLINE, HARD_DEADLINE))
        results.setdefault("qwen35_4b", {}).setdefault("meta", {})["parser_smoke"] = smoke_4b
        inspect_runner._atomic_write(RESULTS_PATH, results)
except Exception as exc:
    print("PHASE-4B FAILED:", type(exc).__name__, exc)
    if SERVER_LOG.exists():
        print("\n".join(SERVER_LOG.read_text(errors="replace").splitlines()[-30:]))
finally:
    stop_server()
'''

CELL_VERDICT = r'''# ---------------- phase 4: verdict vs pre-registered kill criteria -----------
try:
    final = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {}
    verdicts = inspect_runner.verdict(final)
    final["_verdicts"] = verdicts
    final["_wall_minutes"] = round((time.time() - NOTEBOOK_START) / 60, 1)
    inspect_runner._atomic_write(RESULTS_PATH, final)
    print("=" * 88)
    print("INSPECTION-PROMPT REPLAY — VERDICT TABLE")
    print("kill criteria: error ratio > 1.5x recorded 27B  OR  fact recovery < 0.70")
    print("=" * 88)
    print(json.dumps(verdicts, indent=1))
    print("=" * 88)
    print(f"total wall: {final['_wall_minutes']} min")
except Exception as exc:
    print("VERDICT PHASE FAILED:", type(exc).__name__, exc)
'''


def code_cell(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


def md_cell(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def main() -> None:
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "cells": [
            md_cell(MD_HEADER),
            code_cell(CELL_IMPORTS),
            code_cell(CELL_GPU_ASSERT),
            code_cell(CELL_INPUTS),
            code_cell(CELL_WHEELHOUSE),
            code_cell(CELL_SERVE),
            code_cell(CELL_RUNNER_IMPORT),
            code_cell(CELL_PHASE_9B),
            code_cell(CELL_PHASE_4B),
            code_cell(CELL_VERDICT),
        ],
    }
    NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", NB_PATH)


if __name__ == "__main__":
    main()
