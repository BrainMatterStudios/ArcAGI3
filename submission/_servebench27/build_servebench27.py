#!/usr/bin/env python3
"""build_servebench27.py — LEAN serving-config bench for the 27B stack.

Question (pre-registered): WHICH of the four V31 serving flags carries the
1.4 -> 2.66 gap? Throughput is the binding constraint (R8), and throughput
needs a load generator, not a 25-game smoke. One kernel session sweeps five
server configs on the exact eval GPU:

  A  v12-baseline   --max-model-len 65536                     (our 1.4 stack)
  B  +fp8kv262k     262144 + --kv-cache-dtype fp8             (V22 dataset)
  C  +mtp3          B + speculative mtp num_speculative_tokens=3
  D  +async         C + --async-scheduling
  E  +nochunk       D + --no-enable-chunked-prefill           (V31, the 2.66 stack)

Per config: relaunch only the vLLM server (weights cached on disk), 60 s
warmup + 420 s measured load at concurrency 28 with prompt sizes matched to
live traffic (4k/12k/20k/28k tokens, shared system prefix, 2 tool defs), then
an 8-prompt greedy probe (temp 0, fixed seed). Recorded evidence per config:
completed turns/min, output tok/s, per-bucket latency, the server log's OWN
lines for kv_cache_dtype / KV-cache token count / speculative / async /
chunked (never trust the launch arg), and last log lines on LAUNCH_FAIL.

Decision rule (pre-registered): a flag "carries" if turns/min >= +10% over its
predecessor config with <= 2/8 greedy-probe divergences vs A.

Boot cells 2/6/8 are copied VERBATIM from the proven-boot public notebook
(romantamrazov/arc-real-agi-solution, papermill-completed 2026-08-17 on this
exact docker image + RTX Pro 6000).
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
SRC_NB = Path(
    "/private/tmp/claude-501/-Users-ahmed-Documents-ArcAGI3/"
    "336208eb-4625-4ff9-b10d-1664cc846e5d/scratchpad/tamrazov/arc-real-agi-solution.ipynb"
)
KERNEL_SLUG = "arc3-servebench27"

SWEEP_SOURCE = r'''# ==== servebench: five-config vLLM sweep, load-gen at concurrency 28 ====
import json as _json
import os
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

BENCH_START = time.time()
MAX_WALL_S = 4.0 * 3600          # hard budget for the whole sweep
WARMUP_S = 60
MEASURE_S = 420
CONCURRENCY = 28
PORT = 1234
BASE_URL = f"http://127.0.0.1:{PORT}/v1"
RESULTS_PATH = Path("/kaggle/working/servebench_results.json")

_prov = _json.loads((WORKING_DIR / "qwen38-model-provenance.json").read_text())
MODEL_PATH = _prov["model_path"]
SERVED = os.environ.get("INFERENCE_ANALYZER_MODEL", "Qwen/Qwen3.8-27B-FP8")
SITE_PACKAGES = WORKING_DIR / "vllm-site-packages"

BASE_ARGS = [
    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
    "--model", MODEL_PATH, "--served-model-name", SERVED,
    "--host", "127.0.0.1", "--port", str(PORT),
    "--tensor-parallel-size", "1", "--enable-auto-tool-choice",
    "--tool-call-parser", "qwen3_coder", "--generation-config", "vllm",
    "--enable-prefix-caching",
    "--default-chat-template-kwargs", '{"preserve_thinking": true}',
    "--reasoning-parser", "qwen3",
]
MTP = ["--speculative-config", '{"method":"mtp","num_speculative_tokens":3}']
CONFIGS = [
    ("A_v12_baseline", ["--max-model-len", "65536"]),
    ("B_fp8kv_262k", ["--max-model-len", "262144", "--kv-cache-dtype", "fp8"]),
    ("C_plus_mtp3", ["--max-model-len", "262144", "--kv-cache-dtype", "fp8", *MTP]),
    ("D_plus_async", ["--max-model-len", "262144", "--kv-cache-dtype", "fp8", *MTP,
                      "--async-scheduling"]),
    ("E_plus_nochunk", ["--max-model-len", "262144", "--kv-cache-dtype", "fp8", *MTP,
                        "--async-scheduling", "--no-enable-chunked-prefill"]),
]

# ---- traffic model: sizes from the measured live profile (chars ~ 3.3/token)
GRID_ROW = " ".join(str((i * 7) % 10) for i in range(64))
def make_payload(tokens: int, salt: int) -> str:
    rows = max(1, int(tokens * 3.3) // (len(GRID_ROW) + 1))
    return f"turn-salt:{salt}\n" + "\n".join(
        f"row{r:03d}: {GRID_ROW}" for r in range(rows))

SYSTEM_PREFIX = ("You are playing an ARC-AGI-3 grid game. Analyse the board, "
                 "maintain a world model, then act via the tool. " + make_payload(2000, 0))
BUCKETS = [("s4k", 4000, 0.20), ("m12k", 12000, 0.35),
           ("l20k", 20000, 0.30), ("xl28k", 28000, 0.15)]
TOOLS = [
    {"type": "function", "function": {"name": "python", "description": "Run python",
     "parameters": {"type": "object", "properties": {"code": {"type": "string"}},
                    "required": ["code"]}}},
    {"type": "function", "function": {"name": "action", "description": "Send game actions",
     "parameters": {"type": "object", "properties": {"actions": {"type": "array",
                    "items": {"type": "string"}}}, "required": ["actions"]}}},
]

_server = {"proc": None, "log": None}

def _port_open() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", PORT), timeout=2):
            return True
    except OSError:
        return False

def kill_server() -> None:
    proc = _server.get("proc")
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
    # the setup-launched server has a pid file instead of a handle
    pid_path = WORKING_DIR / "vllm-openai-server.pid"
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text().strip())
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.kill(pid, sig)
                    time.sleep(3)
                except OSError:
                    break
        except Exception as exc:
            print("pid-file kill warning:", repr(exc), flush=True)
        pid_path.unlink(missing_ok=True)
    deadline = time.time() + 120
    while _port_open() and time.time() < deadline:
        time.sleep(2)
    time.sleep(5)

def spawn_server(name: str, extra: list[str]) -> Path:
    env = os.environ.copy()
    pp = env.get("PYTHONPATH", "")
    if str(SITE_PACKAGES) not in pp.split(os.pathsep):
        env["PYTHONPATH"] = str(SITE_PACKAGES) + (os.pathsep + pp if pp else "")
    env.update({"USE_TF": "0", "TRANSFORMERS_NO_TF": "1",
                "TRANSFORMERS_NO_TORCHVISION": "1", "VLLM_NO_USAGE_STATS": "1"})
    log_path = Path(f"/kaggle/working/servebench_{name}.log")
    proc = subprocess.Popen(BASE_ARGS + extra, env=env,
                            stdout=log_path.open("w"), stderr=subprocess.STDOUT,
                            text=True, start_new_session=True)
    _server["proc"] = proc
    _server["log"] = log_path
    return log_path

def wait_healthy(timeout: float = 600.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        proc = _server.get("proc")
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"{BASE_URL}/models", timeout=5) as r:
                if 200 <= r.status < 300:
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False

def chat(messages, *, max_tokens=700, temperature=0.7, seed=None, timeout=420):
    body = {"model": SERVED, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "tools": TOOLS}
    if seed is not None:
        body["seed"] = seed
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions", data=_json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = _json.load(resp)
    usage = data.get("usage", {})
    return {"latency": time.time() - t0,
            "completion_tokens": usage.get("completion_tokens", 0),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "text": (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""}

def run_load(measure_s: int) -> dict:
    stop_at = time.time() + WARMUP_S + measure_s
    window_start = time.time() + WARMUP_S
    records, errors = [], []
    lock = threading.Lock()

    def worker(wid: int):
        salt = wid * 1000
        while time.time() < stop_at:
            r = (salt * 2654435761) % 100 / 100.0
            acc, bucket = 0.0, BUCKETS[-1]
            for b in BUCKETS:
                acc += b[2]
                if r < acc:
                    bucket = b
                    break
            salt += 1
            msgs = [{"role": "system", "content": SYSTEM_PREFIX},
                    {"role": "user", "content": make_payload(bucket[1], salt)}]
            try:
                out = chat(msgs)
                done = time.time()
                with lock:
                    records.append((bucket[0], out["latency"],
                                    out["completion_tokens"], done >= window_start,
                                    done <= stop_at))
            except Exception as exc:
                with lock:
                    errors.append(repr(exc)[:200])

    threads = [threading.Thread(target=worker, args=(i,), daemon=True)
               for i in range(CONCURRENCY)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=WARMUP_S + measure_s + 480)
    measured = [r for r in records if r[3] and r[4]]
    by_bucket = {}
    for b, lat, ctok, *_ in measured:
        by_bucket.setdefault(b, []).append(lat)
    return {
        "measured_requests": len(measured),
        "turns_per_min": round(len(measured) / (measure_s / 60.0), 2),
        "output_tok_per_s": round(sum(r[2] for r in measured) / measure_s, 1),
        "latency_by_bucket": {b: round(statistics.mean(v), 1)
                              for b, v in sorted(by_bucket.items())},
        "errors": len(errors), "error_sample": errors[:5],
    }

def greedy_probe() -> list[str]:
    outs = []
    for i in range(8):
        msgs = [{"role": "system", "content": "Answer tersely."},
                {"role": "user", "content": f"Probe {i}: list the first {5+i} primes, "
                 "then name the action you would try first in an unknown grid game."}]
        try:
            outs.append(chat(msgs, max_tokens=96, temperature=0.0, seed=1234)["text"])
        except Exception as exc:
            outs.append(f"PROBE_ERROR {exc!r}")
    return outs

def log_evidence(log_path: Path) -> list[str]:
    pats = ("kv_cache_dtype", "KV cache", "speculative", "async_scheduling",
            "chunked_prefill", "num_speculative_tokens")
    hits = []
    try:
        for line in log_path.read_text(errors="replace").splitlines():
            if any(p.lower() in line.lower() for p in pats):
                hits.append(line.strip()[:220])
    except Exception as exc:
        hits.append(f"EVIDENCE_READ_FAIL {exc!r}")
    return hits[:30]

results = {}
for name, extra in CONFIGS:
    left = MAX_WALL_S - (time.time() - BENCH_START)
    if left < 16 * 60:
        results[name] = {"status": "SKIPPED_BUDGET", "seconds_left": int(left)}
        print(f"=== {name}: SKIPPED (budget, {int(left)}s left) ===", flush=True)
        continue
    print(f"=== {name}: relaunching server ({extra}) ===", flush=True)
    kill_server()
    log_path = spawn_server(name, extra)
    if not wait_healthy():
        tail = ""
        try:
            tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-40:])
        except Exception:
            pass
        results[name] = {"status": "LAUNCH_FAIL", "log_tail": tail[-4000:]}
        print(f"=== {name}: LAUNCH_FAIL ===\n{tail[-2000:]}", flush=True)
        continue
    load = run_load(MEASURE_S)
    results[name] = {"status": "OK", "args": extra, **load,
                     "greedy": greedy_probe(), "log_evidence": log_evidence(log_path)}
    print(f"=== {name}: {load['turns_per_min']} turns/min, "
          f"{load['output_tok_per_s']} out tok/s, errors={load['errors']} ===", flush=True)
    RESULTS_PATH.write_text(_json.dumps(results, indent=1))

kill_server()
RESULTS_PATH.write_text(_json.dumps(results, indent=1))
base = results.get("A_v12_baseline", {})
print("\n==== SERVEBENCH SUMMARY ====")
for name, r in results.items():
    if r.get("status") != "OK":
        print(f"{name:16s} {r.get('status')}")
        continue
    delta = ""
    if base.get("status") == "OK" and base.get("turns_per_min"):
        delta = f" ({(r['turns_per_min'] / base['turns_per_min'] - 1) * 100:+.0f}% vs A)"
    g = sum(1 for a, b in zip(r.get("greedy", []),
                              base.get("greedy", [])) if a != b) if base.get("greedy") else "-"
    print(f"{name:16s} turns/min={r['turns_per_min']}{delta} "
          f"out_tok/s={r['output_tok_per_s']} err={r['errors']} greedy_div_vs_A={g}")
print("results ->", RESULTS_PATH)
'''


def main() -> None:
    src = json.loads(SRC_NB.read_text())
    cells = src["cells"]
    boot = [cells[2], cells[6], cells[8]]
    for want, cell in (("WORKING_DIR =", boot[0]), ("_find_bundle_dir", boot[1]),
                       ("setup_commands.json", boot[2])):
        assert want in "".join(cell["source"]), f"boot cell missing marker {want!r}"

    def code_cell(source: str) -> dict:
        return {"cell_type": "code", "execution_count": None, "metadata": {},
                "outputs": [], "source": source}

    def md_cell(source: str) -> dict:
        return {"cell_type": "markdown", "metadata": {}, "source": source}

    def strip(cell: dict) -> dict:
        return {"cell_type": cell["cell_type"], "execution_count": None,
                "metadata": {}, "outputs": [], "source": "".join(cell["source"])}

    nb = {
        "nbformat": 4, "nbformat_minor": 4,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                    "name": "python3"},
                     "language_info": {"name": "python", "version": "3.12"}},
        "cells": [
            md_cell("# arc3-servebench27 — five-config vLLM serving sweep\n"
                    "Which V31 flag carries the 1.4→2.66 gap? Load-gen at concurrency 28, "
                    "no games. Boot cells verbatim from the proven public notebook."),
            strip(boot[0]), strip(boot[1]), strip(boot[2]),
            code_cell(SWEEP_SOURCE),
        ],
    }
    nb_path = HERE / f"{KERNEL_SLUG}.ipynb"
    nb_path.write_text(json.dumps(nb, indent=1))
    (HERE / "kernel-metadata.json").write_text(json.dumps({
        "id": f"ahmedmobasher86/{KERNEL_SLUG}",
        "title": KERNEL_SLUG,
        "code_file": f"{KERNEL_SLUG}.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "dataset_sources": ["driessmit1/arc3-vllm-h100-wheelhouse-v3",
                            "keithtyser/taaf-duck-qwen38-serving-v1"],
        "kernel_sources": [],
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "model_sources": ["foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1"],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:"
                        "57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c",
        "machine_shape": "NvidiaRtxPro6000",
    }, indent=1))
    import hashlib
    code = "\n".join(c["source"] for c in nb["cells"] if c["cell_type"] == "code")
    print("built", KERNEL_SLUG, "code-cell sha256",
          hashlib.sha256(code.encode()).hexdigest()[:16])


if __name__ == "__main__":
    main()
