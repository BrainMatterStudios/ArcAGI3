#!/usr/bin/env python3
"""build_serving_lab.py — one-commit serving probe kernel (NO games, NO submission).

Answers two queued questions in a single GPU commit-run (target <= 2.5 h):

Q1 — LIVE THROUGHPUT (open branch of the live-vs-offline gap): on the Kaggle
RTX Pro 6000, serve Qwen3.8-27B-FP8 with the EXACT scored serve chain (anim
bundle setup_commands.json, patched constants — prefix caching ON, KV bf16,
qwen3_coder parser, max-model-len 65536) and measure sustained per-session
generation tok/min at concurrency 8/16/28 with realistic duck-shaped requests
(15-25k-token prompts incl. base64 board PNGs, thinking ON, 1-3k gen tokens,
>= 10 min measured windows at conc 8 and 28).
Baseline: Modal H100 measured 445 gen-tok/min/session at conc 28.
Decision rule (printed in the log): >= 400 tok/min/session at conc 28 =
throughput hypothesis DEAD; < 300 = throughput IS the live discount.

Q2 — MTP SPECULATIVE DECODING (big-lever queue #2): restart vLLM with
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'
  --no-enable-prefix-caching   (MANDATORY: GDN+MTP+prefix-cache corruption,
  fix PR #47861 is post-0.19 — docs/RESEARCH-2026-08-21-bug-lever-hunt.md)
re-run the load matrix at conc 8/16/28, read spec-decode acceptance from
/metrics (names verified against vLLM v0.19.0 vllm/v1/spec_decode/metrics.py:
vllm:spec_decode_num_drafts / _num_draft_tokens / _num_accepted_tokens /
_num_accepted_tokens_per_pos; log fallback "Avg Draft acceptance rate"),
round-trip duck-style python tool calls through the qwen3_coder parser, and
run a 26k-token long-context soak hunting the FP8+MTP illegal-memory-access
crash (vllm issue #40756). A dead server prints MTP-FATAL, never fails the
kernel. nst=2 arm runs if the clock allows.

Everything is phase-wrapped; partial results always land in
/kaggle/working/serving_lab_results.json.

Usage:
  .venv/bin/python submission/_serving_lab/build_serving_lab.py
  # DO NOT push without Ahmed's go: this build is local-only.
"""
import ast
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).parent
SCAFFOLD = HERE.parent / "_parity_ab" / "scaffold-arc3-duck-v12-with-qwen-3-8-27b.ipynb"
DUCK38_BUILD = HERE.parent / "_duck38_v12" / "build_duck38_v12.py"
KERNEL_SLUG = "arc3-serving-lab"

# Markers that locate the scaffold cells we reuse verbatim (exact scored serve chain).
MARK_IMPORTS = "NOTEBOOK_START_EPOCH = time.time()"
MARK_CONFIG = "# Qwen3.8 / Kaggle input configuration"
MARK_AUDIT = "# Audit the attached inputs that matter for this run."
MARK_SETUP = "TAAF/vLLM setup completed for Qwen3.8"


def _load_attest_cell() -> str:
    spec = importlib.util.spec_from_file_location("build_duck38_v12", DUCK38_BUILD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ATTEST_CELL


MD_HEADER = """\
# arc3-serving-lab — one-commit serving probe (NO games)

Two queued questions, one GPU commit, target <= 2.5 h:

| Phase | What | Budget |
|---|---|---|
| boot | anim-bundle setup (EXACT scored serve chain) + boot attestation | ~20 min |
| A | baseline load matrix conc 8/16/28 (10/8/10 min windows) + parser control | ~36 min |
| B | restart w/ MTP nst=3 + `--no-enable-prefix-caching`; parser round-trip; matrix 8/16/28 (8 min each); acceptance from /metrics | ~42 min |
| C | 26k-token long-context soak (FP8+MTP crash hunt, vllm #40756) | ~9 min |
| D | (time-gated < 110 min elapsed) MTP nst=2 short matrix conc 8/16 | ~22 min |
| final | decision table + verdicts + results JSON | ~2 min |

**Q1 rule:** conc-28 per-session gen tok/min >= 400 = throughput hypothesis DEAD;
< 300 = throughput IS the live discount. (Modal H100 reference: 445.)

**Q2 outputs:** tok/s baseline vs MTP per concurrency, acceptance rates, parser
OK, crash Y/N. MTP runs WITHOUT prefix caching (GDN+MTP+prefix-cache
corruption; fix PR #47861 is post-0.19 — RESEARCH-2026-08-21-bug-lever-hunt).

All measurements: `/kaggle/working/serving_lab_results.json` (saved after every
phase; every phase failure-wrapped). This kernel never plays a game and never
touches the competition rerun path.
"""


CELL_LAB_LIB = r'''# ============================ SERVING LAB LIBRARY ============================
# No games are played in this kernel. Two questions, one commit:
#   Q1 live throughput at conc 8/16/28 on the EXACT scored serve config.
#   Q2 MTP speculative decoding: nst=3 (+2 if time), acceptance, parser, soak.
import base64
import io
import random
import re
import statistics
import threading
import traceback
import urllib.error
import urllib.request

VLLM_HOST = "127.0.0.1"
VLLM_PORT = 1234
VLLM_ROOT = f"http://{VLLM_HOST}:{VLLM_PORT}"
VLLM_API = VLLM_ROOT + "/v1"
VLLM_MAX_MODEL_LEN = 65536
SITE_PACKAGES = WORKING_DIR / "vllm-site-packages"
RESULTS_PATH = WORKING_DIR / "serving_lab_results.json"

LAB_HARD_CAP_MIN = 150.0   # skip any phase starting after this
NST2_GATE_MIN = 110.0      # nst=2 arm only if reached before this
MODAL_H100_TOKMIN = 445.0  # measured gen-tok/min/session at conc 28 (reference)

Q1_RULE = (">=400 tok/min/session at conc 28 = throughput hypothesis DEAD; "
           "<300 = throughput is the live discount")

# Exact scored serve flags (June bundle setup_commands.json, md5 99e4b35d...),
# minus the prefix-caching flag which is parameterized per phase. KV cache
# stays DEFAULT bf16 — we deliberately do NOT copy keithtyser's
# --kv-cache-dtype fp8 (no calibrated KV scales; corruption reports #42179).
BASE_SERVE_FLAGS = [
    "--model", str(QWEN_MODEL_PATH),
    "--served-model-name", QWEN_SERVED_MODEL_NAME,
    "--host", VLLM_HOST,
    "--port", str(VLLM_PORT),
    "--tensor-parallel-size", "1",
    "--enable-auto-tool-choice",
    "--tool-call-parser", "qwen3_coder",
    "--generation-config", "vllm",
    "--default-chat-template-kwargs", '{"preserve_thinking": true}',
    "--reasoning-parser", "qwen3",
    "--max-model-len", str(VLLM_MAX_MODEL_LEN),
]
MTP3_FLAGS = ["--speculative-config",
              '{"method": "mtp", "num_speculative_tokens": 3}',
              "--no-enable-prefix-caching"]
MTP2_FLAGS = ["--speculative-config",
              '{"method": "mtp", "num_speculative_tokens": 2}',
              "--no-enable-prefix-caching"]

CURRENT_SERVER = {"proc": None,
                  "log": str(WORKING_DIR / "vllm-openai-server.log"),
                  "tag": "baseline-scored-flags"}

RESULTS = {
    "meta": {
        "kernel": "arc3-serving-lab",
        "started_utc": datetime.utcnow().isoformat() + "Z",
        "model_path": str(QWEN_MODEL_PATH),
        "served_model_name": QWEN_SERVED_MODEL_NAME,
        "max_model_len": VLLM_MAX_MODEL_LEN,
        "q1_rule": Q1_RULE,
        "modal_h100_tokmin_session_conc28": MODAL_H100_TOKMIN,
        "baseline_flags": BASE_SERVE_FLAGS + ["--enable-prefix-caching"],
        "mtp3_flags": BASE_SERVE_FLAGS + MTP3_FLAGS,
        "mtp2_flags": BASE_SERVE_FLAGS + MTP2_FLAGS,
        "kv_cache_dtype": "default (bf16) — fp8 KV deliberately NOT copied",
        "spec_metric_names_verified": [
            "vllm:spec_decode_num_drafts",
            "vllm:spec_decode_num_draft_tokens",
            "vllm:spec_decode_num_accepted_tokens",
            "vllm:spec_decode_num_accepted_tokens_per_pos",
        ],
    },
    "phases": {},
    "verdicts": {},
}


def elapsed_min():
    return (time.time() - NOTEBOOK_START_EPOCH) / 60.0


def save_results():
    tmp = RESULTS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(RESULTS, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(RESULTS_PATH)


def http_json(url, payload=None, timeout=120):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_text(url, timeout=20):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def server_alive(timeout=5):
    try:
        http_json(VLLM_API + "/models", timeout=timeout)
        return True
    except Exception:
        return False


def vllm_procs():
    out = subprocess.run(["pgrep", "-f", "vllm.entrypoints"], capture_output=True, text=True)
    return [int(x) for x in out.stdout.split() if x.strip().isdigit()]


def gpu_sample():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
        util, mem = out.stdout.strip().splitlines()[0].split(",")
        return {"util_pct": int(util.strip()), "mem_mib": int(mem.strip())}
    except Exception:
        return None


def tail_log_lines(path, max_bytes=524288):
    p = Path(path)
    if not p.exists():
        return []
    with p.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        return handle.read().decode("utf-8", errors="replace").splitlines()


def tail_log(path, n=60):
    return "\n".join(tail_log_lines(path)[-n:])


# ---- Prometheus scrape (metric names verified at vLLM v0.19.0 tag) ----------
METRIC_RE = re.compile(r"^(vllm:[A-Za-z0-9_]+)(?:\{[^}]*\})?\s+([0-9.eE+-]+|NaN|nan)\s*$")
WANTED_PREFIXES = ("vllm:spec_decode", "vllm:generation_tokens",
                   "vllm:prompt_tokens", "vllm:prefix_cache",
                   "vllm:num_preemptions", "vllm:num_requests")


def scrape_metrics():
    try:
        text = http_text(VLLM_ROOT + "/metrics", timeout=25)
    except Exception as exc:
        return {}, [f"scrape-failed: {exc!r}"]
    counters, sample_lines = {}, []
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        base = line.split("{")[0].split(" ")[0]
        if not base.startswith(WANTED_PREFIXES):
            continue
        if len(sample_lines) < 40:
            sample_lines.append(line)
        match = METRIC_RE.match(line)
        if not match:
            continue
        name = match.group(1)
        if name.endswith("_total"):
            name = name[:-len("_total")]
        try:
            counters[name] = counters.get(name, 0.0) + float(match.group(2))
        except ValueError:
            pass
    return counters, sample_lines


def acceptance_delta(before, after):
    def delta(name):
        return after.get(name, 0.0) - before.get(name, 0.0)
    drafts = delta("vllm:spec_decode_num_drafts")
    draft_toks = delta("vllm:spec_decode_num_draft_tokens")
    accepted = delta("vllm:spec_decode_num_accepted_tokens")
    return {
        "num_drafts": drafts,
        "num_draft_tokens": draft_toks,
        "num_accepted_tokens": accepted,
        "acceptance_rate": (accepted / draft_toks) if draft_toks > 0 else None,
        "mean_accepted_per_draft": (accepted / drafts) if drafts > 0 else None,
    }


def running_requests():
    counters, _ = scrape_metrics()
    return counters.get("vllm:num_requests_running")


def drain_inflight(max_wait_s=180):
    # After stop, in-flight requests keep decoding; wait so they don't pollute
    # the next phase's measurement window.
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        active = running_requests()
        if active is None or active <= 0:
            break
        time.sleep(10)


def log_acceptance_lines(n=5):
    # vLLM v0.19.0 vllm/v1/spec_decode/metrics.py logs "Avg Draft acceptance rate: %.1f%%"
    return [ln for ln in tail_log_lines(CURRENT_SERVER["log"])
            if "acceptance rate" in ln][-n:]


# ---- server lifecycle -------------------------------------------------------
def stop_server(reason):
    print(f"serving-lab: stopping vLLM ({reason})", flush=True)
    subprocess.run(["pkill", "-TERM", "-f", "vllm.entrypoints"], check=False)
    deadline = time.time() + 90
    while time.time() < deadline and vllm_procs():
        time.sleep(3)
    if vllm_procs():
        subprocess.run(["pkill", "-9", "-f", "vllm.entrypoints"], check=False)
        time.sleep(10)
    deadline = time.time() + 240
    while time.time() < deadline:
        sample = gpu_sample()
        if sample is not None and sample["mem_mib"] < 8000:
            break
        time.sleep(5)
    print(f"serving-lab: server stopped, gpu={gpu_sample()}", flush=True)


def start_server(extra_flags, tag, timeout_s=1500):
    log_path = WORKING_DIR / f"vllm-{tag}.log"
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
           *BASE_SERVE_FLAGS, *extra_flags]
    env = os.environ.copy()
    pypath = env.get("PYTHONPATH", "")
    if str(SITE_PACKAGES) not in pypath.split(os.pathsep):
        env["PYTHONPATH"] = f"{SITE_PACKAGES}{os.pathsep}{pypath}" if pypath else str(SITE_PACKAGES)
    env.update({"USE_TF": "0", "TRANSFORMERS_NO_TF": "1",
                "TRANSFORMERS_NO_TORCHVISION": "1", "VLLM_NO_USAGE_STATS": "1",
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    print("serving-lab: starting vLLM:", " ".join(cmd), flush=True)
    handle = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
    CURRENT_SERVER.update({"proc": proc, "log": str(log_path), "tag": tag})
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"vLLM ({tag}) died during startup rc={proc.returncode}\n" + tail_log(log_path))
        if server_alive():
            print(f"serving-lab: vLLM ready ({tag})", flush=True)
            return
        time.sleep(5)
    raise TimeoutError(f"vLLM ({tag}) not ready in {timeout_s}s\n" + tail_log(log_path))


# ---- duck-shaped request synthesis -----------------------------------------
PALETTE = [(0, 0, 0), (0, 116, 217), (255, 65, 54), (46, 204, 64),
           (255, 220, 0), (170, 170, 170), (240, 18, 190), (255, 133, 27),
           (128, 219, 255), (135, 12, 37), (105, 58, 183), (63, 81, 181),
           (255, 255, 255)]


def _png_rgb(rows):
    # Minimal pure-stdlib PNG encoder (fallback if PIL is unavailable).
    import struct
    import zlib
    height = len(rows)
    width = len(rows[0])
    raw = b"".join(b"\x00" + b"".join(bytes(px) for px in row) for row in rows)

    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def board_png_b64(rng):
    # 64x64 board upscaled 4x -> 256x256 (matches MULTIMODAL_UPSCALE=4).
    cells, scale = 64, 4
    base = rng.randrange(len(PALETTE))
    grid = [[PALETTE[rng.randrange(len(PALETTE))] if rng.random() < 0.15
             else PALETTE[(base + x // 8 + y // 8) % len(PALETTE)]
             for x in range(cells)] for y in range(cells)]
    try:
        from PIL import Image
        img = Image.new("RGB", (cells, cells))
        for y in range(cells):
            for x in range(cells):
                img.putpixel((x, y), grid[y][x])
        img = img.resize((cells * scale, cells * scale), Image.NEAREST)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        raw = buf.getvalue()
    except Exception:
        rows = []
        for y in range(cells):
            row = []
            for x in range(cells):
                row.extend([grid[y][x]] * scale)
            rows.extend(list(row) for _ in range(scale))
        raw = _png_rgb(rows)
    return base64.b64encode(raw).decode("ascii")


WORDS = ("grid cluster border sprite agent portal key door wall floor toggle "
         "rotate mirror count color region path move click reward level frame "
         "delta pixel row col mask object pattern rule hypothesis verify plan "
         "act observe anchor cursor palette symmetry adjacency corridor").split()


def make_transcript(rng, approx_tokens):
    lines, tokens, step = [], 0, 0
    while tokens < approx_tokens:
        step += 1
        words = " ".join(rng.choice(WORDS) for _ in range(24))
        lines.append(f"[turn {step:04d}] obs: {words}. delta_pixels="
                     f"{rng.randrange(900)} score={rng.randrange(7)}")
        tokens += 34
    return "\n".join(lines)


SYSTEM_TEXT = ("You are an ARC-AGI-3 game-playing analyst. Maintain a world "
               "model, goal model and action model from board observations, "
               "then choose the next batch of actions. Think carefully.\n"
               + make_transcript(random.Random(7), 1800))

PYTHON_TOOL = [{
    "type": "function",
    "function": {
        "name": "python",
        "description": ("Run Python code against the current game state. The "
                        "snippet is ephemeral and is not saved across calls."),
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string",
                                    "description": "Python code to run."}},
            "required": ["code"],
        },
    },
}]


def _strip_images(user_msg):
    content = [part for part in user_msg["content"] if part.get("type") != "image_url"]
    return {"role": "user", "content": content}


class Session:
    # One synthetic duck "game worker": a growing conversation whose prefix is
    # stable across turns (what makes prefix caching realistic in Phase A).
    def __init__(self, idx, seed, images_per_req=5):
        self.rng = random.Random(seed)
        self.idx = idx
        self.images_per_req = images_per_req
        self.turns = []
        self.base_context = make_transcript(self.rng, 11000 + self.rng.randrange(4000))
        self.last_prompt_tokens = None
        self._pending_user = None

    def _new_user_turn(self):
        text = ("[turn] board updated; analyze the change and choose the next "
                "actions.\n" + make_transcript(self.rng, 400))
        content = [{"type": "text", "text": text},
                   {"type": "image_url", "image_url": {
                       "url": "data:image/png;base64," + board_png_b64(self.rng)}}]
        return {"role": "user", "content": content}

    def build_messages(self):
        msgs = [{"role": "system", "content": SYSTEM_TEXT},
                {"role": "user", "content": [{"type": "text", "text":
                    "Game transcript so far:\n" + self.base_context}]},
                {"role": "assistant", "content": "Understood. World model initialized."}]
        total = len(self.turns)
        for i, (user_msg, assistant_msg) in enumerate(self.turns):
            if total - i > self.images_per_req:
                user_msg = _strip_images(user_msg)
            msgs.append(user_msg)
            msgs.append(assistant_msg)
        self._pending_user = self._new_user_turn()
        msgs.append(self._pending_user)
        return msgs

    def record(self, assistant_text, usage):
        reply = (assistant_text or "").strip()[:1500] or "(thinking only)"
        self.turns.append((self._pending_user, {"role": "assistant", "content": reply}))
        tokens = usage.get("prompt_tokens")
        self.last_prompt_tokens = tokens
        while tokens and tokens > 24500 and len(self.turns) > 2:
            self.turns.pop(0)
            tokens -= 900


def chat_request(session, max_tokens, timeout=1500):
    payload = {
        "model": QWEN_SERVED_MODEL_NAME,
        "messages": session.build_messages(),
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    started = time.time()
    resp = http_json(VLLM_API + "/chat/completions", payload, timeout=timeout)
    latency = time.time() - started
    usage = resp.get("usage") or {}
    message = resp["choices"][0]["message"]
    session.record(message.get("content") or "", usage)
    return {"t_end": time.time(), "latency_s": latency,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "had_reasoning": bool(message.get("reasoning_content"))}


# ---- load phase -------------------------------------------------------------
def per_session_of(phase):
    if not phase:
        return None
    metric = phase.get("gen_tok_min_session_metric")
    return metric if metric is not None else phase.get("gen_tok_min_session_usage")


def run_load_phase(name, conc, warmup_s, measure_s):
    if elapsed_min() > LAB_HARD_CAP_MIN:
        print(f"serving-lab: SKIP {name} — past hard cap ({elapsed_min():.0f} min)", flush=True)
        RESULTS["phases"][name] = {"skipped": "hard-cap"}
        save_results()
        return None
    if not server_alive(15):
        print(f"serving-lab: SKIP {name} — server not alive", flush=True)
        RESULTS["phases"][name] = {"skipped": "server-dead"}
        save_results()
        return None
    print(f"\nserving-lab: === {name} (conc={conc}, warmup={warmup_s}s, "
          f"measure={measure_s}s, elapsed={elapsed_min():.1f} min) ===", flush=True)
    events, errors = [], []
    lock = threading.Lock()
    stop = threading.Event()
    gpu_samples = []

    def sampler():
        while not stop.is_set():
            sample = gpu_sample()
            if sample:
                gpu_samples.append(sample)
            stop.wait(30)

    def worker(i):
        session = Session(i, seed=(hash(name) & 0xFFFF) * 100 + i)
        while not stop.is_set():
            max_tok = session.rng.choice((1024, 2048, 3072))
            try:
                record = chat_request(session, max_tok)
                with lock:
                    events.append(record)
            except urllib.error.HTTPError as exc:
                try:
                    body = exc.read().decode("utf-8", errors="replace")[:400]
                except Exception:
                    body = ""
                with lock:
                    errors.append({"t": time.time(), "code": exc.code, "body": body})
                if "image" in body.lower() or "multi" in body.lower():
                    session.images_per_req = 1
                stop.wait(3)
            except Exception as exc:
                with lock:
                    errors.append({"t": time.time(), "err": repr(exc)[:200]})
                stop.wait(5)

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(conc)]
    threads.append(threading.Thread(target=sampler, daemon=True))
    for thread in threads:
        thread.start()
    time.sleep(warmup_s)
    metrics_before, _ = scrape_metrics()
    t0 = time.time()
    server_died = False
    while time.time() - t0 < measure_s:
        time.sleep(15)
        if not server_alive(10) and not vllm_procs():
            server_died = True
            print(f"serving-lab: SERVER DIED during {name}", flush=True)
            break
    t1 = time.time()
    metrics_after, metric_lines = scrape_metrics()
    stop.set()
    for thread in threads:
        thread.join(timeout=2)
    if not server_died:
        drain_inflight()

    window = [e for e in events if t0 <= e["t_end"] <= t1]
    minutes = max((t1 - t0) / 60.0, 0.01)
    usage_gen = sum(e["completion_tokens"] for e in window)
    gen_delta = metrics_after.get("vllm:generation_tokens", 0.0) - \
        metrics_before.get("vllm:generation_tokens", 0.0)
    latencies = sorted(e["latency_s"] for e in window)
    prompts = [e["prompt_tokens"] for e in window]
    result = {
        "conc": conc,
        "warmup_s": warmup_s,
        "measure_min": round(minutes, 2),
        "requests_in_window": len(window),
        "requests_total": len(events),
        "errors": len(errors),
        "error_samples": errors[:5],
        "gen_tok_min_session_metric": round(gen_delta / minutes / conc, 1) if gen_delta > 0 else None,
        "gen_tok_min_session_usage": round(usage_gen / minutes / conc, 1),
        "gen_tok_s_aggregate_metric": round(gen_delta / (minutes * 60.0), 1) if gen_delta > 0 else None,
        "gen_tok_s_aggregate_usage": round(usage_gen / (minutes * 60.0), 1),
        "mean_latency_s": round(statistics.fmean(latencies), 1) if latencies else None,
        "p50_latency_s": round(statistics.median(latencies), 1) if latencies else None,
        "prompt_tokens_mean": round(statistics.fmean(prompts)) if prompts else None,
        "prompt_tokens_min": min(prompts) if prompts else None,
        "prompt_tokens_max": max(prompts) if prompts else None,
        "completion_tokens_mean": round(statistics.fmean(
            [e["completion_tokens"] for e in window])) if window else None,
        "acceptance": acceptance_delta(metrics_before, metrics_after),
        "acceptance_log_lines": log_acceptance_lines(),
        "spec_metric_lines_sample": metric_lines[:8],
        "gpu_samples_tail": gpu_samples[-6:],
        "server_died": server_died,
        "server_alive_at_end": server_alive(),
        "server_tag": CURRENT_SERVER["tag"],
    }
    RESULTS["phases"][name] = result
    save_results()
    print(f"serving-lab: {name}: {result['gen_tok_min_session_metric'] or result['gen_tok_min_session_usage']}"
          f" gen-tok/min/session ({len(window)} reqs, {len(errors)} errs, "
          f"prompt~{result['prompt_tokens_mean']}, p50 lat {result['p50_latency_s']}s)", flush=True)
    if result["acceptance"]["num_draft_tokens"] > 0:
        print(f"serving-lab: {name}: acceptance_rate="
              f"{result['acceptance']['acceptance_rate']:.3f} "
              f"mean_accepted_per_draft={result['acceptance']['mean_accepted_per_draft']:.2f}",
              flush=True)
    return result


# ---- parser round-trip ------------------------------------------------------
def parser_roundtrip(tag):
    name = f"parser_roundtrip_{tag}"
    if not server_alive(15):
        RESULTS["phases"][name] = {"skipped": "server-dead"}
        save_results()
        return
    prompts = [
        ("auto", "The board has an unknown number of red pixels. Use the python "
                 "tool to inspect: call it with code that prints grid[0][0]."),
        ("auto", "You must act now. Emit a python tool call whose code prints "
                 "the string 'quack' and nothing else."),
        ("forced", "Count from 1 to 3 using the python tool."),
    ]
    outcomes = []
    for mode, prompt in prompts:
        payload = {
            "model": QWEN_SERVED_MODEL_NAME,
            "messages": [
                {"role": "system", "content":
                    "You are an ARC-AGI-3 analyst. Use the python tool to act."},
                {"role": "user", "content": prompt},
            ],
            "tools": PYTHON_TOOL,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 1200,
            "chat_template_kwargs": {"enable_thinking": True},
        }
        if mode == "forced":
            payload["tool_choice"] = {"type": "function", "function": {"name": "python"}}
        try:
            resp = http_json(VLLM_API + "/chat/completions", payload, timeout=600)
            message = resp["choices"][0]["message"]
            tool_calls = message.get("tool_calls") or []
            ok, detail = False, ""
            if tool_calls:
                fn = tool_calls[0].get("function", {})
                args_ok = False
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                    args_ok = isinstance(args.get("code"), str)
                except Exception:
                    args = None
                ok = fn.get("name") == "python" and args_ok
                detail = f"name={fn.get('name')} args_parse={'OK' if args_ok else 'FAIL'}"
            else:
                detail = (f"no tool_calls; finish={resp['choices'][0].get('finish_reason')}; "
                          f"content_head={(message.get('content') or '')[:80]!r}")
            outcomes.append({"mode": mode, "ok": ok, "detail": detail})
        except Exception as exc:
            outcomes.append({"mode": mode, "ok": False, "detail": repr(exc)[:250]})
    ok_count = sum(1 for o in outcomes if o["ok"])
    verdict = "PASS" if ok_count >= 2 else "FAIL"
    RESULTS["phases"][name] = {"ok_count": ok_count, "total": len(outcomes),
                               "verdict": verdict, "outcomes": outcomes}
    save_results()
    print(f"serving-lab: {name}: {verdict} ({ok_count}/{len(outcomes)} tool calls parsed)", flush=True)


# ---- long-context soak (FP8+MTP crash hunt, vllm #40756) --------------------
CRASH_PATTERNS = ("illegal memory access", "CUDA error", "device-side assert",
                  "core dumped", "Engine core initialization failed",
                  "Watchdog caught")


def long_context_soak(tag, conc=6, duration_s=480, target_prompt_tokens=26000):
    name = f"{tag}_soak_longctx"
    if elapsed_min() > LAB_HARD_CAP_MIN:
        RESULTS["phases"][name] = {"skipped": "hard-cap"}
        save_results()
        return
    if not server_alive(15):
        RESULTS["phases"][name] = {"skipped": "server-dead-before-soak",
                                   "verdict": "MTP-FATAL (server already dead before soak)"}
        save_results()
        print("serving-lab: MTP-FATAL — server dead before long-context soak", flush=True)
        return
    print(f"\nserving-lab: === {name} (conc={conc}, {duration_s}s, "
          f"~{target_prompt_tokens}-token prompts) ===", flush=True)
    events, errors = [], []
    lock = threading.Lock()
    stop = threading.Event()
    counter = {"n": 0}

    def worker(i):
        while not stop.is_set():
            with lock:
                counter["n"] += 1
                seq = counter["n"]
            rng = random.Random(90000 + seq)
            text = make_transcript(rng, target_prompt_tokens)
            messages = [
                {"role": "system", "content": SYSTEM_TEXT},
                {"role": "user", "content": [
                    {"type": "text", "text": "Full game transcript:\n" + text
                        + "\nAnalyze deeply and produce a plan."},
                    {"type": "image_url", "image_url": {
                        "url": "data:image/png;base64," + board_png_b64(rng)}},
                ]},
            ]
            payload = {"model": QWEN_SERVED_MODEL_NAME, "messages": messages,
                       "temperature": 0.6, "top_p": 0.95, "top_k": 20,
                       "max_tokens": 2048,
                       "chat_template_kwargs": {"enable_thinking": True}}
            started = time.time()
            try:
                resp = http_json(VLLM_API + "/chat/completions", payload, timeout=1500)
                usage = resp.get("usage") or {}
                with lock:
                    events.append({"latency_s": round(time.time() - started, 1),
                                   "prompt_tokens": usage.get("prompt_tokens", 0),
                                   "completion_tokens": usage.get("completion_tokens", 0)})
            except Exception as exc:
                with lock:
                    errors.append({"t": time.time(), "err": repr(exc)[:200]})
                stop.wait(5)

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(conc)]
    for thread in threads:
        thread.start()
    t0 = time.time()
    server_died = False
    while time.time() - t0 < duration_s:
        time.sleep(15)
        if not server_alive(10) and not vllm_procs():
            server_died = True
            print(f"serving-lab: SERVER DIED during {name}", flush=True)
            break
    stop.set()
    for thread in threads:
        thread.join(timeout=2)
    crash_lines = [ln for ln in tail_log_lines(CURRENT_SERVER["log"])
                   if any(pat in ln for pat in CRASH_PATTERNS)][:10]
    alive = server_alive()
    if server_died or (not alive and not vllm_procs()):
        verdict = "MTP-FATAL (server died during long-context soak — vllm #40756 class)"
    elif crash_lines:
        verdict = "SUSPECT (crash strings in server log, server still alive)"
    else:
        verdict = "SURVIVED"
    result = {"conc": conc, "duration_s": duration_s,
              "requests_ok": len(events), "errors": len(errors),
              "error_samples": errors[:5],
              "prompt_tokens_max": max((e["prompt_tokens"] for e in events), default=None),
              "mean_latency_s": round(statistics.fmean(
                  [e["latency_s"] for e in events]), 1) if events else None,
              "server_alive_at_end": alive, "crash_lines": crash_lines,
              "verdict": verdict}
    RESULTS["phases"][name] = result
    save_results()
    print(f"serving-lab: {name}: {verdict} ({len(events)} ok, {len(errors)} errs, "
          f"max prompt {result['prompt_tokens_max']})", flush=True)


print(f"serving-lab: library ready, elapsed {elapsed_min():.1f} min")
save_results()
'''


CELL_PHASE_A = r'''# ===================== PHASE A — BASELINE (scored config) ====================
# Q1: sustained per-session generation tok/min on the EXACT scored serve chain
# (server booted by the bundle setup above: prefix caching ON, KV bf16).
try:
    parser_roundtrip("baseline")
    for _name, _conc, _warm, _meas in [("baseline_conc8", 8, 90, 600),
                                       ("baseline_conc16", 16, 90, 480),
                                       ("baseline_conc28", 28, 90, 600)]:
        run_load_phase(_name, _conc, _warm, _meas)
    _b28 = RESULTS["phases"].get("baseline_conc28") or {}
    _rate28 = per_session_of(_b28)
    print("\nserving-lab: Q1 DECISION RULE:", Q1_RULE)
    print(f"serving-lab: Modal H100 reference: {MODAL_H100_TOKMIN} gen-tok/min/session at conc 28")
    if _rate28 is None:
        _q1 = "NO-DATA (conc-28 baseline phase did not complete)"
    elif _rate28 >= 400:
        _q1 = f"THROUGHPUT HYPOTHESIS DEAD ({_rate28} >= 400 tok/min/session at conc 28)"
    elif _rate28 < 300:
        _q1 = f"THROUGHPUT IS THE LIVE DISCOUNT ({_rate28} < 300 tok/min/session at conc 28)"
    else:
        _q1 = f"INCONCLUSIVE BAND ({_rate28} in 300-400 tok/min/session at conc 28)"
    RESULTS["verdicts"]["q1_throughput"] = _q1
    print("serving-lab: Q1 VERDICT:", _q1, flush=True)
except Exception:
    traceback.print_exc()
    RESULTS["verdicts"].setdefault("q1_throughput", "PHASE-A-ERROR (see traceback)")
save_results()
'''


CELL_MTP_BOOT = r'''# ===================== PHASE B BOOT — MTP nst=3 restart ======================
# MANDATORY: --no-enable-prefix-caching. GDN+MTP+prefix-cache corrupts
# generations on vLLM 0.19 (fix PR #47861 is post-0.19) —
# docs/RESEARCH-2026-08-21-bug-lever-hunt.md. Everything else = scored flags.
try:
    if elapsed_min() > LAB_HARD_CAP_MIN:
        raise RuntimeError(f"hard cap reached ({elapsed_min():.0f} min) — MTP boot skipped")
    stop_server("switch to MTP nst=3, prefix caching OFF")
    start_server(MTP3_FLAGS, tag="mtp3")
    _probe_before, _ = scrape_metrics()
    _probe_session = Session(999, seed=999)
    chat_request(_probe_session, 256, timeout=900)
    _probe_after, _probe_lines = scrape_metrics()
    _probe = acceptance_delta(_probe_before, _probe_after)
    _active = (_probe["num_draft_tokens"] or 0) > 0
    RESULTS["phases"]["mtp3_boot"] = {"ok": True, "spec_decode_active": _active,
                                      "probe": _probe,
                                      "spec_metric_lines": _probe_lines[:10]}
    if _active:
        _rate_txt = (f"{_probe['acceptance_rate']:.3f}"
                     if _probe["acceptance_rate"] is not None else "n/a")
        print(f"serving-lab: MTP ACTIVE — probe acceptance_rate={_rate_txt}", flush=True)
    else:
        print("serving-lab: WARNING — MTP INERT: no spec_decode draft tokens "
              "after probe request; check flags/metrics names", flush=True)
except Exception as exc:
    traceback.print_exc()
    RESULTS["phases"]["mtp3_boot"] = {"ok": False, "error": repr(exc)[:600]}
    RESULTS["verdicts"]["q2_mtp_boot"] = "MTP-FATAL (BOOT) — vLLM failed to start with MTP nst=3"
    print("serving-lab: MTP-FATAL (BOOT) — verdict recorded, kernel continues", flush=True)
save_results()
'''


CELL_PHASE_B = r'''# ================ PHASE B — MTP nst=3 load matrix + parser ==================
try:
    if RESULTS["phases"].get("mtp3_boot", {}).get("ok"):
        parser_roundtrip("mtp3")
        for _name, _conc, _warm, _meas in [("mtp3_conc8", 8, 90, 480),
                                           ("mtp3_conc16", 16, 90, 480),
                                           ("mtp3_conc28", 28, 90, 480)]:
            run_load_phase(_name, _conc, _warm, _meas)
        if not server_alive(10) and not vllm_procs():
            RESULTS["verdicts"]["q2_crash"] = "MTP-FATAL (LOAD-MATRIX) — server died during nst=3 matrix"
            print("serving-lab: MTP-FATAL (LOAD-MATRIX)", flush=True)
    else:
        print("serving-lab: skipping MTP load matrix — nst=3 boot failed", flush=True)
except Exception:
    traceback.print_exc()
save_results()
'''


CELL_SOAK = r'''# ============ PHASE C — 26k-token long-context soak (crash hunt) ============
# Hunts the FP8+MTP illegal-memory-access crash (vllm issue #40756). With
# prefix caching OFF every request is a full ~26k-token prefill — worst case.
try:
    if RESULTS["phases"].get("mtp3_boot", {}).get("ok"):
        long_context_soak("mtp3", conc=6, duration_s=480, target_prompt_tokens=26000)
        _soak = RESULTS["phases"].get("mtp3_soak_longctx", {})
        if _soak.get("verdict"):
            RESULTS["verdicts"]["q2_crash"] = _soak["verdict"]
    else:
        print("serving-lab: skipping soak — nst=3 boot failed", flush=True)
except Exception:
    traceback.print_exc()
save_results()
'''


CELL_NST2 = r'''# ============== PHASE D — MTP nst=2 (time-gated salvage arm) ================
# vLLM SpeculativeConfig warns nst>1 re-runs the single MTP layer and can
# lower acceptance — nst=2 is the fallback arm if nst=3 acceptance is poor.
try:
    if elapsed_min() > NST2_GATE_MIN:
        print(f"serving-lab: skipping nst=2 arm — elapsed {elapsed_min():.0f} min "
              f"> gate {NST2_GATE_MIN:.0f}", flush=True)
        RESULTS["phases"]["mtp2_boot"] = {"skipped": "time-gate"}
    else:
        stop_server("switch to MTP nst=2, prefix caching OFF")
        start_server(MTP2_FLAGS, tag="mtp2")
        RESULTS["phases"]["mtp2_boot"] = {"ok": True}
        for _name, _conc, _warm, _meas in [("mtp2_conc8", 8, 60, 300),
                                           ("mtp2_conc16", 16, 60, 300)]:
            run_load_phase(_name, _conc, _warm, _meas)
except Exception as exc:
    traceback.print_exc()
    RESULTS["phases"]["mtp2_boot"] = {"ok": False, "error": repr(exc)[:600]}
    print("serving-lab: nst=2 arm failed — recorded, kernel continues", flush=True)
save_results()
'''


CELL_FINAL = r'''# ===================== FINAL — decision table + verdicts =====================
def _cell(value, width=12):
    text = "-" if value is None else str(value)
    return text.rjust(width)


def _acc(phase):
    acc = (phase or {}).get("acceptance") or {}
    rate = acc.get("acceptance_rate")
    return f"{rate:.3f}" if isinstance(rate, float) else None


print("\n" + "=" * 88)
print("SERVING LAB DECISION TABLE — gen-tok/min/session (metric-based; usage fallback)")
print("=" * 88)
print(f"{'conc':>6} {'baseline':>12} {'mtp3':>12} {'mtp3/base':>12} "
      f"{'mtp3 accept':>12} {'mtp2':>12} {'mtp2 accept':>12}")
for _conc in (8, 16, 28):
    _base = RESULTS["phases"].get(f"baseline_conc{_conc}")
    _mtp3 = RESULTS["phases"].get(f"mtp3_conc{_conc}")
    _mtp2 = RESULTS["phases"].get(f"mtp2_conc{_conc}")
    _b, _m3, _m2 = per_session_of(_base), per_session_of(_mtp3), per_session_of(_mtp2)
    _ratio = round(_m3 / _b, 2) if _b and _m3 else None
    print(f"{_conc:>6} {_cell(_b)} {_cell(_m3)} {_cell(_ratio)} "
          f"{_cell(_acc(_mtp3))} {_cell(_m2)} {_cell(_acc(_mtp2))}")

print("-" * 88)
print("AGGREGATE tok/s (all sessions summed):")


def _agg(phase):
    if not phase:
        return None
    metric = phase.get("gen_tok_s_aggregate_metric")
    return metric if metric is not None else phase.get("gen_tok_s_aggregate_usage")


print(f"{'conc':>6} {'baseline':>12} {'mtp3':>12} {'mtp2':>12}")
for _conc in (8, 16, 28):
    print(f"{_conc:>6} {_cell(_agg(RESULTS['phases'].get(f'baseline_conc{_conc}')))} "
          f"{_cell(_agg(RESULTS['phases'].get(f'mtp3_conc{_conc}')))} "
          f"{_cell(_agg(RESULTS['phases'].get(f'mtp2_conc{_conc}')))}")

print("-" * 88)
print("Q1 RULE:   ", Q1_RULE)
print("Q1 VERDICT:", RESULTS["verdicts"].get("q1_throughput", "NO-DATA"))

_parser_base = RESULTS["phases"].get("parser_roundtrip_baseline", {})
_parser_mtp = RESULTS["phases"].get("parser_roundtrip_mtp3", {})
print(f"PARSER:     baseline {_parser_base.get('verdict', 'N/A')} "
      f"({_parser_base.get('ok_count', '-')}/{_parser_base.get('total', '-')}) | "
      f"mtp3 {_parser_mtp.get('verdict', 'N/A')} "
      f"({_parser_mtp.get('ok_count', '-')}/{_parser_mtp.get('total', '-')})")

_crash = RESULTS["verdicts"].get("q2_crash") or RESULTS["verdicts"].get("q2_mtp_boot")
_fatal = bool(_crash and _crash.startswith("MTP-FATAL"))
_crash_yn = "Y" if _fatal else "N"
RESULTS["verdicts"]["q2_crash_yn"] = _crash_yn
print(f"CRASH Y/N:  {_crash_yn} ({_crash or 'no crash observed'})")

_mtp3_boot = RESULTS["phases"].get("mtp3_boot", {})
if _mtp3_boot.get("ok") and not _mtp3_boot.get("spec_decode_active"):
    print("WARNING:    MTP booted but spec_decode counters never moved (MTP INERT)")

_m3_28, _b_28 = (per_session_of(RESULTS["phases"].get("mtp3_conc28")),
                 per_session_of(RESULTS["phases"].get("baseline_conc28")))
_m3_8, _b_8 = (per_session_of(RESULTS["phases"].get("mtp3_conc8")),
               per_session_of(RESULTS["phases"].get("baseline_conc8")))
if not _mtp3_boot.get("ok"):
    print("Q2 VERDICT: MTP-FATAL at boot — no MTP measurements")
elif _fatal:
    print("Q2 VERDICT:", _crash)
else:
    if _m3_8 and _b_8:
        _gain8 = _m3_8 / _b_8
        _gain28 = (_m3_28 / _b_28) if (_m3_28 and _b_28) else None
        _q2 = (f"MTP nst=3 speedup: conc8 {_gain8:.2f}x"
               + (f", conc28 {_gain28:.2f}x" if _gain28 else ", conc28 n/a"))
    else:
        _q2 = "MTP matrix incomplete — see phases JSON"
    RESULTS["verdicts"]["q2_mtp_speedup"] = _q2
    print("Q2 VERDICT:", _q2)

RESULTS["meta"]["finished_utc"] = datetime.utcnow().isoformat() + "Z"
RESULTS["meta"]["total_elapsed_min"] = round(elapsed_min(), 1)
save_results()
print("=" * 88)
print(f"serving-lab: DONE in {elapsed_min():.1f} min — results JSON: {RESULTS_PATH}")
print("=" * 88, flush=True)
'''


def _code_cell(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


def _md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": source.splitlines(keepends=True)}


def main() -> None:
    scaffold = json.loads(SCAFFOLD.read_text())

    def find_cell(marker: str) -> str:
        hits = [c for c in scaffold["cells"]
                if c["cell_type"] == "code" and marker in "".join(c["source"])]
        assert len(hits) == 1, (marker, len(hits))
        return "".join(hits[0]["source"])

    imports_cell = find_cell(MARK_IMPORTS)
    config_cell = find_cell(MARK_CONFIG)
    audit_cell = find_cell(MARK_AUDIT)
    setup_cell = find_cell(MARK_SETUP)
    attest_cell = _load_attest_cell()

    cells = [
        _md_cell(MD_HEADER),
        _code_cell(imports_cell),
        _code_cell(config_cell),
        _code_cell(audit_cell),
        _code_cell(setup_cell),   # boots vLLM with the exact scored flags
        _code_cell(attest_cell),  # doctrine v2: weights verified before measuring
        _code_cell(CELL_LAB_LIB),
        _code_cell(CELL_PHASE_A),
        _code_cell(CELL_MTP_BOOT),
        _code_cell(CELL_PHASE_B),
        _code_cell(CELL_SOAK),
        _code_cell(CELL_NST2),
        _code_cell(CELL_FINAL),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    # Validate: every code cell must be syntactically valid python.
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        try:
            ast.parse(src)
        except SyntaxError as exc:
            raise SystemExit(f"cell {i} failed ast.parse: {exc}") from exc

    joined = "\n".join("".join(c["source"]) for c in notebook["cells"])
    assert "attest: OK" in joined
    assert joined.index("attest: OK") < joined.index("SERVING LAB LIBRARY")
    assert "--no-enable-prefix-caching" in joined
    assert '"method": "mtp"' in joined
    # The fp8 KV flag must never appear as an actual flag-list item (comments
    # referencing it are fine) — we keep KV bf16.
    assert '"--kv-cache-dtype"' not in joined

    nb_path = HERE / f"{KERNEL_SLUG}.ipynb"
    nb_path.write_text(json.dumps(notebook, indent=1) + "\n")

    (HERE / "kernel-metadata.json").write_text(json.dumps({
        "id": f"ahmedmobasher86/{KERNEL_SLUG}",
        "title": KERNEL_SLUG,
        "code_file": f"{KERNEL_SLUG}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "machine_shape": "NvidiaRtxPro6000",
        "dataset_sources": [
            "driessmit1/arc3-vllm-h100-wheelhouse-v3",
            "jakobbrggen/taaf-kaggle-source-anim-20260807-anim",
        ],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": ["foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1"],
    }, indent=2) + "\n")

    import hashlib
    code = "\n".join("".join(c["source"]) for c in notebook["cells"]
                     if c["cell_type"] == "code")
    print("built", KERNEL_SLUG, "code-cell sha256",
          hashlib.sha256(code.encode()).hexdigest())
    print("cells:", len(notebook["cells"]), "| notebook:", nb_path)


if __name__ == "__main__":
    main()
