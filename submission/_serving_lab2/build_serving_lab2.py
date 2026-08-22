#!/usr/bin/env python3
"""build_serving_lab2.py — serving-lab-2: vLLM 0.27.1 + DFlash2 lane probe
(NO games, NO submission).

Three questions, one GPU commit-run (hard cap ~2.5 h):

Q1 — 0.27 THROUGHPUT: boot vLLM 0.27.1 from the saltb0x wheelhouse
(saltb0x/arc3-vllm-wheelhouse-v0271-cu129, wheels + PR#52816 DFlash2 overlay)
serving the same FOYSAL Qwen3.8-27B-FP8 repack with the scored parser flags,
and re-run the duck-shaped load matrix at conc 8/28. 0.19 reference:
1626 tok/min/session @conc8, 642.6 @conc28 (arc3-serving-lab, 08-21).

Q2 — DFLASH2 SPECULATIVE DECODING: mount bbucxi/qwen3-8-27b-dflash2 (mirror of
z-lab/Qwen3.8-27B-DFlash2 — config.json git-oid attested EQUAL offline;
model.safetensors size-equal, official LFS sha256 recorded for in-kernel
verification), enable --speculative-config '{"method": "dflash", ...,
"num_speculative_tokens": 7}' (exact schema from the z-lab model card and
vllm v0.27.1 vllm/config/speculative.py) with prefix caching OFF (GDN rule +
vllm issue #52317 spec-decode+prefix-cache startup crash). Measure conc
8/16/28 + acceptance. DECISION RULE: dflash2 conc-28 per-session tok/min
>= 1.25 x 642.6 = 803.3 → LANE OPEN.

Q3 — SM120 FP8 QUALITY BATTERY: fixed 12-prompt battery (2 real sk48
dead-completion reproducers from packv22 transcripts + 10 seeded duck-style
prompts), greedy temp-0 + fixed seed, run on the 0.19 stack FIRST (leg A,
before the upgrade) and again on 0.27 (leg B; leg C on DFlash2 time-gated).
Verdict: dead-completion / degenerate-loop rate difference between stacks.

Phase order: boot 0.19 via the anim bundle (exact scored serve chain), run
leg A + a brief conc-28 reproduction of the 642.6 baseline (~20 min), THEN
install 0.27.1 into a separate site-packages dir, copy the DFlash2 overlay,
and run phases 1-3. Parser round-trip runs FIRST on every stack; a 0.27
parser failure is recorded PARSER-FATAL and throughput still runs. Every
phase is try/except-wrapped; results JSON is atomically rewritten after each
phase; a fail-fast GPU assert dies on P100 before any heavy setup.

Usage:
  .venv/bin/python submission/_serving_lab2/build_serving_lab2.py
"""
import ast
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).parent
SCAFFOLD = HERE.parent / "_parity_ab" / "scaffold-arc3-duck-v12-with-qwen-3-8-27b.ipynb"
DUCK38_BUILD = HERE.parent / "_duck38_v12" / "build_duck38_v12.py"
DEAD_REPRO = HERE / "dead_repro.json"
V1_CARRY = HERE / "v1_results_carry.json"
KERNEL_SLUG = "arc3-serving-lab2"

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
# arc3-serving-lab2 v3 — vLLM 0.27.1 + DFlash2 lane probe (NO games)

**Boot-failure history:** v1 died on flashinfer `check_cuda_arch` (image
exports pre-sm75 `TORCH_CUDA_ARCH_LIST`; device is sm_120). v2, with the arch
pinned to `12.0`, died on `_normalize_cuda_arch` → `SM 12.x requires CUDA >=
12.9` (flashinfer parses `nvcc --version` from the image's pre-12.9
`/usr/local/cuda`). **v3 aimed fix (code-verified against flashinfer
0.6.16.post3):** `FLASHINFER_CUDA_ARCH_LIST=12.0f` — an explicit suffix is
stored as-is, skipping both checks; `CUDA_HOME` → the pip `nvidia/cuda_nvcc`
dir, which verifiably ships NO `bin/nvcc` (only `ptxas` + `nvvm`), so any
remaining version probe falls back to `torch.version.cuda` = 12.9 from the
cu129 torch wheel; precompiled kernels come from the `flashinfer_cubin` wheel.
Sampler-off fallback ladder widened to catch both prior error strings. The
0.19 leg (parser PASS 3/3, battery leg A 12/12 scored 0 dead, conc-28 699.1
tok/min/session) is carried from v1 verbatim — no bundle setup in v3.
A THIRD distinct blocker closes the lane for real.

| Phase | What | Budget |
|---|---|---|
| boot | GPU assert + input audit + weight attestation (no 0.19 serve chain) | ~4 min |
| 0 | carry v1's 0.19-leg results into the results JSON | ~0 min |
| 1 | install vLLM 0.27.1 (saltb0x wheelhouse) + PR#52816 DFlash2 overlay; boot with scored parser flags (adaptive, deviations logged); parser round-trip FIRST; battery leg B | ~35 min |
| 2 | 0.27 baseline throughput conc 8/28, duck-shaped load | ~26 min |
| 3 | DFlash2 draft attest (sha256 vs official z-lab LFS oid) + boot (`method: dflash`, nst=7, prefix caching OFF) + parser + matrix conc 28/8/16 + acceptance; battery leg C time-gated | ~42 min |
| final | decision table + verdicts + results JSON | ~2 min |

**Q1/Q2 rule:** DFlash2 conc-28 per-session gen tok/min >= 1.25 x 642.6 = **803.2 → LANE OPEN**.
0.19 refs (arc3-serving-lab, 08-21): 1626 @conc8, 1140 @conc16, 642.6 @conc28.

**Q3:** identical greedy (temp 0, fixed seed) 12-prompt battery on 0.19 vs 0.27
(vs DFlash2 if time): dead-completion / degenerate-loop rate per stack. The two
reproducer prompts are REAL sk48 turns that dead-completed live (one replicated
6x at 12:45-13:09 on 08-22 packv22 traces).

Spec-decode runs with `--no-enable-prefix-caching` (GDN rule; vllm #52317
prefix-cache+spec-decode startup crash). Every measurement lands in
`/kaggle/working/serving_lab2_results.json` after every phase. This kernel
never plays a game and never touches the competition rerun path.
"""


CELL_GPU_ASSERT = r'''# ===================== FAIL-FAST GPU ASSERT (before any setup) ==============
# This lab is only meaningful on the RTX Pro 6000 pool. Die IMMEDIATELY on a
# P100/T4 rehoming so the slot costs minutes, not hours.
_gpu_query = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
    capture_output=True, text=True)
GPU_NAME = (_gpu_query.stdout or "").strip()
print(f"serving-lab2: GPU = {GPU_NAME!r} (rc={_gpu_query.returncode})")
if _gpu_query.returncode != 0 or not GPU_NAME:
    raise RuntimeError("WRONG-GPU: nvidia-smi failed — no usable GPU. Aborting fast.")
_gpu_upper = GPU_NAME.upper()
if "6000" not in _gpu_upper or "RTX" not in _gpu_upper:
    raise RuntimeError(
        f"WRONG-GPU: expected an RTX Pro 6000, got {GPU_NAME!r}. Aborting fast "
        "so the session dies in minutes (P100/T4 rehoming).")
print("serving-lab2: GPU assert PASS", flush=True)
'''


CELL_AUDIT2 = r'''# ============== serving-lab2 extra input audit (fail fast) ==================
# The 0.27 wheelhouse and the DFlash2 draft must be attached, else phases 1-3
# are impossible — die before the ~20-min bundle setup.
def _resolve_mount(owner, slug):
    for cand in (Path("/kaggle/input") / slug,
                 Path("/kaggle/input/datasets") / owner / slug):
        if cand.is_dir():
            return cand
    return None


WHEELHOUSE_0271 = _resolve_mount("saltb0x", "arc3-vllm-wheelhouse-v0271-cu129")
DFLASH2_DRAFT_DIR = _resolve_mount("bbucxi", "qwen3-8-27b-dflash2")
print("serving-lab2: wheelhouse 0.27.1:", WHEELHOUSE_0271)
print("serving-lab2: dflash2 draft:   ", DFLASH2_DRAFT_DIR)
if WHEELHOUSE_0271 is None:
    raise RuntimeError("saltb0x/arc3-vllm-wheelhouse-v0271-cu129 is not attached")
if DFLASH2_DRAFT_DIR is None:
    raise RuntimeError("bbucxi/qwen3-8-27b-dflash2 is not attached")
_wh_wheels = sorted(p.name for p in WHEELHOUSE_0271.glob("*.whl"))
print(f"serving-lab2: wheelhouse has {len(_wh_wheels)} wheels; "
      f"vllm wheels: {[w for w in _wh_wheels if w.startswith('vllm')]}")
assert any(w.startswith("vllm-0.27.1") for w in _wh_wheels), "vllm 0.27.1 wheel missing"
assert (WHEELHOUSE_0271 / "ovl_manifest.json").is_file(), "DFlash2 overlay manifest missing"
assert (DFLASH2_DRAFT_DIR / "config.json").is_file(), "dflash2 draft config.json missing"
assert (DFLASH2_DRAFT_DIR / "model.safetensors").is_file(), "dflash2 draft weights missing"
'''


# The lab library. @@DEAD_REPRO@@ is replaced by the builder with a repr() of
# the compact dead-completion reproducer JSON.
CELL_LAB_LIB2_TEMPLATE = r'''# =========================== SERVING LAB 2 LIBRARY ==========================
# No games are played in this kernel. Three questions, one commit:
#   Q1 vLLM 0.27.1 throughput vs the 0.19 scored stack (refs 1626/642.6).
#   Q2 DFlash2 speculative decoding lane (>=1.25x at conc 28 = LANE OPEN).
#   Q3 SM120 FP8 quality battery: dead-completion rate 0.19 vs 0.27 (vs dflash2).
import base64
import hashlib
import io
import random
import re
import shutil
import statistics
import threading
import traceback
import urllib.error
import urllib.request
import zlib

VLLM_HOST = "127.0.0.1"
VLLM_PORT = 1234
VLLM_ROOT = f"http://{VLLM_HOST}:{VLLM_PORT}"
VLLM_API = VLLM_ROOT + "/v1"
VLLM_MAX_MODEL_LEN = 65536
SITE_PACKAGES_019 = WORKING_DIR / "vllm-site-packages"
SITE_PACKAGES_0271 = Path("/tmp/vllm-site-packages-0271")
RESULTS_PATH = WORKING_DIR / "serving_lab2_results.json"

LAB_HARD_CAP_MIN = 150.0   # skip any phase starting after this
DFLASH_C16_GATE_MIN = 128.0
BATTERY_C_GATE_MIN = 132.0

# 0.19 references measured by arc3-serving-lab (08-21, this GPU pool, same load
# generator, metric-based per-session tok/min).
V019_REF = {8: 1626.0, 16: 1140.1, 28: 642.6}
LANE_RULE_MULT = 1.25
LANE_RULE_TOKMIN = round(V019_REF[28] * LANE_RULE_MULT, 1)  # 803.3
LANE_RULE = (f"dflash2 conc-28 per-session tok/min >= {LANE_RULE_TOKMIN} "
             f"(1.25 x 0.19 baseline 642.6) = LANE OPEN")

# Official z-lab/Qwen3.8-27B-DFlash2 fingerprints (HF API, fetched 2026-08-22).
# The mounted bbucxi dataset is a third-party mirror with no manifest of its
# own; provenance is attested by comparing against these official values.
DFLASH2_OFFICIAL_SAFETENSORS_SHA256 = \
    "67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c"
DFLASH2_OFFICIAL_SAFETENSORS_SIZE = 3848817896
DFLASH2_OFFICIAL_CONFIG_GIT_OID = "79279cc5665bced6f3cdaa2095a2ffe819497b2e"

# Exact scored serve flags (June bundle setup_commands.json), minus the
# prefix-caching flag which is parameterized per phase. KV cache stays DEFAULT
# bf16. On 0.27 the same surface is attempted first; any flag 0.27 rejects is
# dropped by the adaptive boot and logged as a deviation.
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
# DFlash2 config schema verified against vllm v0.27.1 vllm/config/speculative.py
# ("dflash" in SpeculativeMethod; parallel_drafting auto-set) and the z-lab
# model card's vLLM example (num_speculative_tokens: 7 = block_size 8 minus the
# anchor). The draft config carries no top-level n_predict, so nst is explicit.
DFLASH2_SPEC_FLAGS_TEMPLATE = [
    "--speculative-config",
    None,  # filled at boot with the resolved draft path
    "--no-enable-prefix-caching",
]


def dflash2_flags():
    spec = {"method": "dflash", "model": str(DFLASH2_DRAFT_DIR),
            "num_speculative_tokens": 7}
    flags = list(DFLASH2_SPEC_FLAGS_TEMPLATE)
    flags[1] = json.dumps(spec)
    return flags


CURRENT_SERVER = {"proc": None,
                  "log": str(WORKING_DIR / "vllm-openai-server.log"),
                  "tag": "v019-scored-flags",
                  "site_packages": str(SITE_PACKAGES_019)}

RESULTS = {
    "meta": {
        "kernel": "arc3-serving-lab2",
        "started_utc": datetime.utcnow().isoformat() + "Z",
        "gpu": GPU_NAME,
        "model_path": str(QWEN_MODEL_PATH),
        "served_model_name": QWEN_SERVED_MODEL_NAME,
        "max_model_len": VLLM_MAX_MODEL_LEN,
        "v019_reference_tokmin_session": {str(k): v for k, v in V019_REF.items()},
        "lane_rule": LANE_RULE,
        "baseline_flags_scored": BASE_SERVE_FLAGS + ["--enable-prefix-caching"],
        "dflash2_official_sha256": DFLASH2_OFFICIAL_SAFETENSORS_SHA256,
        "dflash2_official_config_git_oid": DFLASH2_OFFICIAL_CONFIG_GIT_OID,
        "kv_cache_dtype": "default (bf16)",
        "deviations": [],
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


def log_deviation(text):
    print(f"serving-lab2: DEVIATION: {text}", flush=True)
    RESULTS["meta"]["deviations"].append(text)
    save_results()


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


# ---- Prometheus scrape ------------------------------------------------------
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
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        active = running_requests()
        if active is None or active <= 0:
            break
        time.sleep(10)


def log_acceptance_lines(n=5):
    return [ln for ln in tail_log_lines(CURRENT_SERVER["log"])
            if "acceptance" in ln.lower()][-n:]


# ---- server lifecycle -------------------------------------------------------
def stop_server(reason):
    print(f"serving-lab2: stopping vLLM ({reason})", flush=True)
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
    print(f"serving-lab2: server stopped, gpu={gpu_sample()}", flush=True)


UNRECOGNIZED_RE = re.compile(r"unrecognized arguments?:\s*(.+)")
BADARG_RE = re.compile(r"error: argument (--[A-Za-z0-9-]+)")

# v1 root cause: Kaggle image exports a pre-sm75 TORCH_CUDA_ARCH_LIST →
# flashinfer check_cuda_arch() raised "requires sm75+". v2 root cause: with
# the arch pinned to "12.0", CompilationContext._normalize_cuda_arch(12,0)
# calls is_cuda_version_at_least("12.9"), which parses `nvcc --version` from
# /usr/local/cuda (image toolkit < 12.9) → "SM 12.x requires CUDA >= 12.9".
# v3 fix (all code-verified against flashinfer 0.6.16.post3):
#  - FLASHINFER_CUDA_ARCH_LIST="12.0f": an explicit suffix is stored AS-IS by
#    CompilationContext.__init__ (no _normalize_cuda_arch, no version check),
#    and check_cuda_arch() passes on major 12 >= 8 — both prior failure modes
#    bypassed deterministically.
#  - CUDA_HOME → the pip nvidia/cuda_nvcc dir. VERIFIED: that wheel ships NO
#    bin/nvcc (only bin/ptxas + nvvm/libdevice), so flashinfer's
#    get_cuda_version() hits FileNotFoundError and falls back to
#    torch.version.cuda (12.9 from the cu129 torch wheel) for any remaining
#    is_cuda_version_at_least() call — while ptxas/nvvm ARE found. JIT-less
#    kernels come from the flashinfer_cubin wheel the house ships.
ARCH_ENV_FIX = {"TORCH_CUDA_ARCH_LIST": "12.0+PTX",
                "FLASHINFER_CUDA_ARCH_LIST": "12.0f"}


def cuda_env_fix(site_packages):
    cuda_home = Path(site_packages) / "nvidia" / "cuda_nvcc"
    runtime_lib = Path(site_packages) / "nvidia" / "cuda_runtime" / "lib"
    env = dict(ARCH_ENV_FIX)
    env["CUDA_HOME"] = str(cuda_home)
    env["CUDA_PATH"] = str(cuda_home)
    env["PATH"] = f"{cuda_home}/bin{os.pathsep}" + os.environ.get("PATH", "")
    env["LD_LIBRARY_PATH"] = (f"{runtime_lib}{os.pathsep}"
                              + os.environ.get("LD_LIBRARY_PATH", ""))
    return env


FLASHINFER_ERR_MARKERS = ("FlashInfer requires", "sm75", "flashinfer.jit",
                          "requires CUDA", "check_cuda_arch",
                          "compilation_context")


def _strip_flag(flags, flag_name):
    """Remove flag_name (and its value, if the next item is not another flag)."""
    out, i = [], 0
    while i < len(flags):
        if flags[i] == flag_name:
            if i + 1 < len(flags) and not str(flags[i + 1]).startswith("--"):
                i += 2
            else:
                i += 1
            continue
        out.append(flags[i])
        i += 1
    return out


def _launch(cmd, env, log_path):
    handle = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT,
                            text=True)


def start_server(flags, tag, site_packages, timeout_s=1800, adapt_max=4):
    """Boot vLLM from `site_packages` with `flags`; adaptively drop flags the
    CLI rejects and fall back off the FlashInfer sampler on the sm75 JIT
    error (deviations logged). Raises on unrecoverable boot failure — with
    the FULL log tail printed to stdout so the kernel log always carries the
    root cause (v1 lesson: repr() truncation ate the EngineCore traceback)."""
    flags = list(flags)
    extra_env = cuda_env_fix(site_packages)
    attempts = 0
    while True:
        attempts += 1
        log_path = WORKING_DIR / f"vllm-{tag}-try{attempts}.log"
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", *flags]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(site_packages)
        env.update({"USE_TF": "0", "TRANSFORMERS_NO_TF": "1",
                    "TRANSFORMERS_NO_TORCHVISION": "1", "VLLM_NO_USAGE_STATS": "1",
                    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        env.update(extra_env)
        print(f"serving-lab2: starting vLLM ({tag}, try {attempts}, "
              f"extra_env={extra_env}):", " ".join(cmd), flush=True)
        proc = _launch(cmd, env, log_path)
        CURRENT_SERVER.update({"proc": proc, "log": str(log_path), "tag": tag,
                               "site_packages": str(site_packages)})
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if proc.poll() is not None:
                log_lines = tail_log_lines(log_path)
                log_txt = "\n".join(log_lines)
                bad = None
                match = UNRECOGNIZED_RE.search(log_txt)
                if match:
                    bad = match.group(1).strip().split()[0]
                else:
                    match = BADARG_RE.search(log_txt)
                    if match:
                        bad = match.group(1)
                if bad and bad.startswith("--") and attempts <= adapt_max:
                    log_deviation(f"{tag}: flag {bad} rejected by this vLLM — "
                                  "dropped and re-booted")
                    flags = _strip_flag(flags, bad)
                    break  # retry outer loop
                if (any(m in log_txt for m in FLASHINFER_ERR_MARKERS)
                        and extra_env.get("VLLM_USE_FLASHINFER_SAMPLER") != "0"
                        and attempts <= adapt_max):
                    log_deviation(f"{tag}: FlashInfer JIT failure in boot log — "
                                  "retrying with VLLM_USE_FLASHINFER_SAMPLER=0")
                    extra_env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
                    break  # retry outer loop
                print(f"serving-lab2: {tag} BOOT FAILURE — full log tail "
                      f"({log_path.name}):", flush=True)
                print("\n".join(log_lines[-150:]), flush=True)
                CURRENT_SERVER["last_boot_log_tail"] = log_lines[-150:]
                raise RuntimeError(
                    f"vLLM ({tag}) died during startup rc={proc.returncode}; "
                    f"full tail printed above; log persisted at {log_path}")
            if server_alive():
                print(f"serving-lab2: vLLM ready ({tag}) with flags: "
                      + " ".join(flags) + f" extra_env={extra_env}", flush=True)
                CURRENT_SERVER["extra_env"] = dict(extra_env)
                return flags
            time.sleep(5)
        else:
            log_lines = tail_log_lines(log_path)
            print(f"serving-lab2: {tag} BOOT TIMEOUT — full log tail:", flush=True)
            print("\n".join(log_lines[-150:]), flush=True)
            CURRENT_SERVER["last_boot_log_tail"] = log_lines[-150:]
            raise TimeoutError(f"vLLM ({tag}) not ready in {timeout_s}s; "
                               f"log persisted at {log_path}")


# ---- vLLM 0.27.1 install + DFlash2 overlay ---------------------------------
def install_vllm_0271():
    """pip-install vLLM 0.27.1 offline from the saltb0x wheelhouse into a
    dedicated site-packages dir, then copy the PR#52816 DFlash2 overlay files
    per ovl_manifest.json. Returns an info dict (raises on failure)."""
    if SITE_PACKAGES_0271.exists():
        shutil.rmtree(SITE_PACKAGES_0271)
    SITE_PACKAGES_0271.mkdir(parents=True)
    pip_log = WORKING_DIR / "pip-install-0271.log"
    cmd = [sys.executable, "-m", "pip", "install", "--no-index",
           "--find-links", str(WHEELHOUSE_0271),
           "--target", str(SITE_PACKAGES_0271),
           "vllm==0.27.1", "flashinfer-python", "flashinfer-cubin"]
    print("serving-lab2: pip install:", " ".join(cmd), flush=True)
    with pip_log.open("w", encoding="utf-8") as handle:
        rc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT,
                            timeout=1800).returncode
    if rc != 0:
        raise RuntimeError("pip install of vLLM 0.27.1 failed rc=%d\n%s"
                           % (rc, tail_log(pip_log, 40)))

    manifest = json.loads((WHEELHOUSE_0271 / "ovl_manifest.json").read_text())
    overlay_shas = {}
    for src_name, rel_dest in sorted(manifest.items()):
        src = WHEELHOUSE_0271 / src_name
        dest = SITE_PACKAGES_0271 / rel_dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = src.read_bytes()
        dest.write_bytes(data)
        overlay_shas[rel_dest] = hashlib.sha256(data).hexdigest()[:16]
        # Make sure new packages have __init__ chains importable.
    print(f"serving-lab2: overlay copied ({len(overlay_shas)} files)", flush=True)

    probe_code = (
        "import json, os\n"
        "import vllm, torch\n"
        "info = {'vllm': vllm.__version__, 'torch': torch.__version__,\n"
        "        'torch_cuda': torch.version.cuda}\n"
        "try:\n"
        "    import flashinfer\n"
        "    info['flashinfer'] = flashinfer.__version__\n"
        "except Exception as e:\n"
        "    info['flashinfer'] = 'IMPORT-FAIL: ' + repr(e)[:120]\n"
        "try:\n"
        "    import flashinfer_cubin\n"
        "    info['flashinfer_cubin'] = getattr(flashinfer_cubin, '__version__', 'present')\n"
        "except Exception as e:\n"
        "    info['flashinfer_cubin'] = 'IMPORT-FAIL: ' + repr(e)[:120]\n"
        "sp = os.environ['SL2_SP']\n"
        "nvcc_dir = os.path.join(sp, 'nvidia', 'cuda_nvcc', 'bin')\n"
        "info['cuda_nvcc_bin'] = sorted(os.listdir(nvcc_dir)) if os.path.isdir(nvcc_dir) else 'MISSING'\n"
        "print(json.dumps(info))\n")
    ver = subprocess.run(
        [sys.executable, "-c", probe_code],
        env={**os.environ, "PYTHONPATH": str(SITE_PACKAGES_0271),
             "SL2_SP": str(SITE_PACKAGES_0271)},
        capture_output=True, text=True, timeout=600)
    try:
        stack = json.loads((ver.stdout or "").strip().splitlines()[-1])
    except Exception:
        raise RuntimeError(f"stack probe failed rc={ver.returncode}: "
                           f"{(ver.stdout or '')[-300:]} / {(ver.stderr or '')[-400:]}")
    print(f"serving-lab2: stack probe = {stack}", flush=True)
    if not stack["vllm"].startswith("0.27.1"):
        raise RuntimeError(f"vLLM version check failed: {stack['vllm']!r}")
    if not str(stack.get("torch_cuda", "")).startswith("12.9"):
        # The whole CUDA_HOME fallback strategy rests on torch reporting 12.9.
        log_deviation(f"torch.version.cuda={stack.get('torch_cuda')!r} — not a "
                      "cu129 build; flashinfer version fallback may misreport")
    return {"version": stack["vllm"], "stack_probe": stack,
            "overlay_files": overlay_shas, "pip_log": str(pip_log)}


def attest_dflash2_draft():
    """Attest the mounted third-party DFlash2 mirror against the official
    z-lab fingerprints. Records everything; hard-fails only on a config
    architecture mismatch (wrong model class ⇒ meaningless measurement)."""
    cfg_raw = (DFLASH2_DRAFT_DIR / "config.json").read_bytes()
    cfg = json.loads(cfg_raw)
    git_oid = hashlib.sha1(b"blob %d\x00" % len(cfg_raw) + cfg_raw).hexdigest()
    st_path = DFLASH2_DRAFT_DIR / "model.safetensors"
    st_size = st_path.stat().st_size
    sha = hashlib.sha256()
    with st_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            sha.update(chunk)
    st_sha = sha.hexdigest()
    report = {
        "mirror": "bbucxi/qwen3-8-27b-dflash2 (third-party, no official manifest)",
        "official": "z-lab/Qwen3.8-27B-DFlash2 (mirror of incoai/Qwen3.8-27B-DFlash2)",
        "config_architectures": cfg.get("architectures"),
        "config_git_oid": git_oid,
        "config_git_oid_matches_official":
            git_oid == DFLASH2_OFFICIAL_CONFIG_GIT_OID,
        "safetensors_size": st_size,
        "safetensors_size_matches_official":
            st_size == DFLASH2_OFFICIAL_SAFETENSORS_SIZE,
        "safetensors_sha256": st_sha,
        "safetensors_sha256_matches_official":
            st_sha == DFLASH2_OFFICIAL_SAFETENSORS_SHA256,
        "dflash_config": cfg.get("dflash_config"),
        "num_target_layers": cfg.get("num_target_layers"),
        "hidden_size": cfg.get("hidden_size"),
    }
    assert cfg.get("architectures") == ["DFlash2DraftModel"], (
        f"dflash2 attest FAIL: architectures {cfg.get('architectures')}")
    assert cfg.get("hidden_size") == 5120 and cfg.get("num_target_layers") == 64, (
        "dflash2 attest FAIL: draft does not target a 64L/5120h model")
    verdict = ("ATTESTED-OFFICIAL-BYTES" if report["safetensors_sha256_matches_official"]
               and report["config_git_oid_matches_official"]
               else "THIRD-PARTY-UNATTESTED (fingerprint mismatch vs z-lab)")
    report["verdict"] = verdict
    print(f"serving-lab2: dflash2 draft attest: {verdict} "
          f"(sha256 {st_sha[:16]}..., config oid match "
          f"{report['config_git_oid_matches_official']})", flush=True)
    return report


# ---- duck-shaped request synthesis (identical to arc3-serving-lab) ----------
PALETTE = [(0, 0, 0), (0, 116, 217), (255, 65, 54), (46, 204, 64),
           (255, 220, 0), (170, 170, 170), (240, 18, 190), (255, 133, 27),
           (128, 219, 255), (135, 12, 37), (105, 58, 183), (63, 81, 181),
           (255, 255, 255)]


def _png_rgb(rows):
    import struct
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
        print(f"serving-lab2: SKIP {name} — past hard cap ({elapsed_min():.0f} min)", flush=True)
        RESULTS["phases"][name] = {"skipped": "hard-cap"}
        save_results()
        return None
    if not server_alive(15):
        print(f"serving-lab2: SKIP {name} — server not alive", flush=True)
        RESULTS["phases"][name] = {"skipped": "server-dead"}
        save_results()
        return None
    print(f"\nserving-lab2: === {name} (conc={conc}, warmup={warmup_s}s, "
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
            print(f"serving-lab2: SERVER DIED during {name}", flush=True)
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
    print(f"serving-lab2: {name}: {result['gen_tok_min_session_metric'] or result['gen_tok_min_session_usage']}"
          f" gen-tok/min/session ({len(window)} reqs, {len(errors)} errs, "
          f"prompt~{result['prompt_tokens_mean']}, p50 lat {result['p50_latency_s']}s)", flush=True)
    if result["acceptance"]["num_draft_tokens"] > 0:
        print(f"serving-lab2: {name}: acceptance_rate="
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
        return "SKIPPED"
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
    print(f"serving-lab2: {name}: {verdict} ({ok_count}/{len(outcomes)} tool calls parsed)", flush=True)
    return verdict


# ---- SM120 quality battery --------------------------------------------------
# 2 REAL dead-completion reproducers (sk48, packv22 traces of 2026-08-22; the
# harness there is text/ascii-based, so the battery is text-only) + 10 seeded
# duck-style prompts. Greedy temp 0, fixed seed → cross-stack comparable.
_DEAD_REPRO_JSON = @@DEAD_REPRO@@
DEAD_REPRO = json.loads(_DEAD_REPRO_JSON)


def _battery_prompts():
    items = []
    for key in ("reproducer_a", "reproducer_b"):
        rep = DEAD_REPRO[key]
        items.append({
            "id": key,
            "source": rep["source"],
            "max_tokens": 9216,
            "messages": [
                {"role": "system", "content": rep["system"]},
                {"role": "user", "content": rep["user"]},
            ],
        })
    for i in range(10):
        rng = random.Random(4200 + i)
        transcript = make_transcript(rng, 6000 + 600 * i)
        items.append({
            "id": f"duckstyle_{i:02d}",
            "source": "seeded synthetic duck-style prompt",
            "max_tokens": 2048,
            "messages": [
                {"role": "system", "content": SYSTEM_TEXT},
                {"role": "user", "content":
                    "Game transcript so far:\n" + transcript
                    + "\n\nAnalyze the latest state and use the python tool to "
                      "run your world-model update and choose the next action "
                      "batch. You must end with exactly one python tool call."},
            ],
        })
    return items


BATTERY = _battery_prompts()


def _degenerate(text):
    tail = (text or "")[-3000:]
    if len(tail) < 1500:
        return False
    ratio = len(zlib.compress(tail.encode("utf-8", "replace"))) / len(tail)
    return ratio < 0.07


def run_battery(name, wall_cap_min=14.0, conc=3):
    if not server_alive(15):
        RESULTS["phases"][name] = {"skipped": "server-dead"}
        save_results()
        return
    print(f"\nserving-lab2: === {name} (12 prompts, greedy temp 0, seed 1234, "
          f"cap {wall_cap_min} min, elapsed {elapsed_min():.1f}) ===", flush=True)
    t_start = time.time()
    results = {}
    lock = threading.Lock()
    queue = list(BATTERY)

    def worker():
        while True:
            with lock:
                if not queue:
                    return
                if (time.time() - t_start) / 60.0 > wall_cap_min:
                    while queue:
                        item = queue.pop()
                        results[item["id"]] = {"skipped": "battery-wall-cap"}
                    return
                item = queue.pop(0)
            payload = {
                "model": QWEN_SERVED_MODEL_NAME,
                "messages": item["messages"],
                "tools": PYTHON_TOOL,
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": 1234,
                "max_tokens": item["max_tokens"],
                "chat_template_kwargs": {"enable_thinking": True},
            }
            started = time.time()
            try:
                resp = http_json(VLLM_API + "/chat/completions", payload, timeout=1400)
                choice = resp["choices"][0]
                message = choice["message"]
                content = message.get("content") or ""
                reasoning = message.get("reasoning_content") or ""
                tool_calls = message.get("tool_calls") or []
                args_blob = "".join(
                    (tc.get("function") or {}).get("arguments") or ""
                    for tc in tool_calls)
                finish = choice.get("finish_reason")
                dead = (finish == "stop" and not tool_calls
                        and not content.strip())
                record = {
                    "finish_reason": finish,
                    "tool_call_count": len(tool_calls),
                    "content_chars": len(content),
                    "reasoning_chars": len(reasoning),
                    "completion_tokens": (resp.get("usage") or {}).get("completion_tokens"),
                    "latency_s": round(time.time() - started, 1),
                    "dead_completion": dead,
                    "degenerate_loop": _degenerate(reasoning) or _degenerate(content),
                    "output_sha256": hashlib.sha256(
                        (reasoning + "\x00" + content + "\x00" + args_blob)
                        .encode("utf-8", "replace")).hexdigest()[:16],
                    "reasoning_head": reasoning[:160],
                    "content_head": content[:160],
                }
            except Exception as exc:
                record = {"error": repr(exc)[:300]}
            with lock:
                results[item["id"]] = record
                done = len(results)
            tag = record.get("finish_reason", "ERR")
            print(f"serving-lab2: {name} [{done}/12] {item['id']}: finish={tag} "
                  f"tools={record.get('tool_call_count')} "
                  f"dead={record.get('dead_completion')} "
                  f"degen={record.get('degenerate_loop')}", flush=True)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(conc)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=wall_cap_min * 60 + 1500)
    scored = [r for r in results.values() if "finish_reason" in r]
    summary = {
        "prompts_total": len(BATTERY),
        "prompts_scored": len(scored),
        "prompts_error": sum(1 for r in results.values() if "error" in r),
        "prompts_skipped": sum(1 for r in results.values() if "skipped" in r),
        "dead_completions": sum(1 for r in scored if r.get("dead_completion")),
        "degenerate_loops": sum(1 for r in scored if r.get("degenerate_loop")),
        "tool_call_prompts": sum(1 for r in scored if r.get("tool_call_count")),
        "finish_length": sum(1 for r in scored if r.get("finish_reason") == "length"),
        "wall_min": round((time.time() - t_start) / 60.0, 1),
        "items": results,
    }
    RESULTS["phases"][name] = summary
    save_results()
    print(f"serving-lab2: {name}: dead={summary['dead_completions']} "
          f"degen={summary['degenerate_loops']} tools={summary['tool_call_prompts']}"
          f"/{summary['prompts_scored']} scored (errors {summary['prompts_error']}, "
          f"skipped {summary['prompts_skipped']})", flush=True)


print(f"serving-lab2: library ready, elapsed {elapsed_min():.1f} min "
      f"(battery: {len(BATTERY)} prompts, "
      f"repro sys+user chars: "
      f"{[len(DEAD_REPRO[k]['system']) + len(DEAD_REPRO[k]['user']) for k in ('reproducer_a', 'reproducer_b')]})")
save_results()
'''


CELL_V1_CARRY_TEMPLATE = r'''# ====== PHASE 0 — 0.19 LEG CARRIED FROM v1 (run 2026-08-22, this GPU pool) ==
# v1 (arc3-serving-lab2 version 1) completed the whole 0.19 leg on the exact
# scored serve chain before its 0.27 boot died on the FlashInfer sm75 JIT
# check. Those measurements are banked verbatim here so the final table and
# the battery cross-stack comparison still work; v2 does NOT re-run the
# ~20-min bundle setup or the 0.19 leg. Cross-session caveat noted in-place.
_V1_CARRY = json.loads(@@V1_CARRY@@)
for _pname in ("parser_roundtrip_v019", "battery_v019", "v019_conc28_brief"):
    _entry = dict(_V1_CARRY["phases"][_pname])
    _entry["carried_from"] = "arc3-serving-lab2 v1 (same GPU pool, 2026-08-22)"
    RESULTS["phases"][_pname] = _entry
RESULTS["verdicts"]["v019_conc28_reproduction"] = (
    "699.1 vs 642.6 ref (1.09x — REPRODUCES; measured in v1, carried)")
RESULTS["meta"]["v1_carry_note"] = (
    "0.19-leg numbers measured by v1 of this kernel; battery leg A greedy "
    "hashes are cross-session, so exact-match rates vs leg B are indicative "
    "only (batching nondeterminism applies within a session anyway)")
print("serving-lab2: carried v1 0.19-leg results:",
      "parser", RESULTS["phases"]["parser_roundtrip_v019"].get("verdict"),
      "| battery dead:", RESULTS["phases"]["battery_v019"].get("dead_completions"),
      "| conc28 brief:", RESULTS["phases"]["v019_conc28_brief"].get("gen_tok_min_session_metric"))
save_results()
'''


CELL_UPGRADE = r'''# ========= PHASE 1a — INSTALL vLLM 0.27.1 + DFlash2 OVERLAY + BOOT ==========
# Wheelhouse: saltb0x/arc3-vllm-wheelhouse-v0271-cu129 (vLLM 0.27.1 cu129
# wheels + PR#52816 DFlash2 overlay files; README instructs copying overlay
# over site-packages after install). 0.27 boot uses the scored parser flag
# surface; any flag 0.27 rejects is dropped adaptively and logged.
V027_OK = False
try:
    if elapsed_min() > LAB_HARD_CAP_MIN:
        raise RuntimeError(f"hard cap reached ({elapsed_min():.0f} min)")
    stop_server("switch 0.19 → 0.27.1")
    _info = install_vllm_0271()
    RESULTS["phases"]["v027_install"] = {"ok": True, **_info}
    save_results()
    _flags_used = start_server(
        BASE_SERVE_FLAGS + ["--enable-prefix-caching"],
        tag="v027-baseline", site_packages=SITE_PACKAGES_0271)
    RESULTS["phases"]["v027_boot"] = {"ok": True, "flags_used": _flags_used}
    V027_OK = True
    print("serving-lab2: vLLM 0.27.1 UP with scored parser flags", flush=True)
except Exception as exc:
    traceback.print_exc()
    RESULTS["phases"]["v027_boot"] = {
        "ok": False, "error": repr(exc)[:2000],
        "boot_log_tail": CURRENT_SERVER.get("last_boot_log_tail", [])[-150:]}
    RESULTS["verdicts"]["q1_v027"] = "V027-BOOT-FATAL — 0.27.1 failed to install/boot"
    print("serving-lab2: V027-BOOT-FATAL — verdict recorded, kernel continues", flush=True)
save_results()
'''


CELL_PHASE1B = r'''# ===== PHASE 1b — 0.27 PARSER ROUND-TRIP (FIRST) + QUALITY BATTERY LEG B ====
# Parser check runs BEFORE any load test (0.19→0.27 parser behavior shifts,
# vllm #39056/#42021). A failure = PARSER-FATAL recorded; throughput still runs.
try:
    if V027_OK:
        _pv = parser_roundtrip("v027")
        if _pv == "FAIL":
            RESULTS["verdicts"]["v027_parser"] = (
                "PARSER-FATAL — qwen3_coder tool calls do not round-trip on "
                "0.27.1 (throughput phases still run; quality still informative)")
            print("serving-lab2: PARSER-FATAL on 0.27", flush=True)
        else:
            RESULTS["verdicts"]["v027_parser"] = f"parser {_pv} on 0.27.1"
        run_battery("battery_v027", wall_cap_min=14.0)
    else:
        print("serving-lab2: skipping 0.27 parser/battery — boot failed", flush=True)
except Exception:
    traceback.print_exc()
save_results()
'''


CELL_PHASE2 = r'''# ============== PHASE 2 — 0.27 BASELINE THROUGHPUT (no spec decode) =========
try:
    if V027_OK:
        run_load_phase("v027_conc8", 8, 90, 480)
        run_load_phase("v027_conc28", 28, 90, 600)
        _c8 = per_session_of(RESULTS["phases"].get("v027_conc8"))
        _c28 = per_session_of(RESULTS["phases"].get("v027_conc28"))
        _r8 = f"{_c8 / V019_REF[8]:.2f}x" if _c8 else "n/a"
        _r28 = f"{_c28 / V019_REF[28]:.2f}x" if _c28 else "n/a"
        RESULTS["verdicts"]["q1_v027"] = (
            f"0.27.1 vs 0.19: conc8 {_c8} ({_r8}), conc28 {_c28} ({_r28})")
        print("serving-lab2: Q1 (0.27 baseline):", RESULTS["verdicts"]["q1_v027"], flush=True)
    else:
        print("serving-lab2: skipping 0.27 throughput — boot failed", flush=True)
except Exception:
    traceback.print_exc()
save_results()
'''


CELL_PHASE3 = r'''# ========= PHASE 3 — DFLASH2 SPECULATIVE DECODING (the lane probe) ==========
# Draft: bbucxi/qwen3-8-27b-dflash2, attested in-kernel against the official
# z-lab fingerprints. Spec config from the z-lab card + v0.27.1
# speculative.py: {"method": "dflash", "num_speculative_tokens": 7}. Prefix
# caching OFF is MANDATORY (GDN rule; vllm #52317 startup crash). conc28 runs
# FIRST — it carries the decision rule.
DFLASH2_OK = False
try:
    if not V027_OK:
        raise RuntimeError("0.27 stack unavailable — DFlash2 lane untestable")
    if elapsed_min() > LAB_HARD_CAP_MIN:
        raise RuntimeError(f"hard cap reached ({elapsed_min():.0f} min)")
    RESULTS["phases"]["dflash2_attest"] = attest_dflash2_draft()
    save_results()
    stop_server("switch 0.27 baseline → 0.27+DFlash2")
    _flags_used = start_server(
        BASE_SERVE_FLAGS + dflash2_flags(),
        tag="dflash2", site_packages=SITE_PACKAGES_0271)
    _probe_before, _ = scrape_metrics()
    _probe_session = Session(999, seed=999)
    chat_request(_probe_session, 256, timeout=900)
    _probe_after, _probe_lines = scrape_metrics()
    _probe = acceptance_delta(_probe_before, _probe_after)
    _active = (_probe["num_draft_tokens"] or 0) > 0
    RESULTS["phases"]["dflash2_boot"] = {
        "ok": True, "flags_used": _flags_used, "spec_decode_active": _active,
        "probe": _probe, "spec_metric_lines": _probe_lines[:10],
        "acceptance_log_lines": log_acceptance_lines()}
    DFLASH2_OK = True
    if _active:
        print(f"serving-lab2: DFLASH2 ACTIVE — probe acceptance_rate="
              f"{(_probe['acceptance_rate'] or 0):.3f}", flush=True)
    else:
        print("serving-lab2: WARNING — DFlash2 booted but spec_decode counters "
              "never moved (check /metrics names on 0.27 + acceptance log lines)",
              flush=True)
except Exception as exc:
    traceback.print_exc()
    RESULTS["phases"]["dflash2_boot"] = {
        "ok": False, "error": repr(exc)[:2000],
        "boot_log_tail": CURRENT_SERVER.get("last_boot_log_tail", [])[-150:]}
    RESULTS["verdicts"]["q2_dflash2"] = "DFLASH2-FATAL (BOOT) — lane CLOSED this probe"
    print("serving-lab2: DFLASH2-FATAL (BOOT) — recorded, kernel continues", flush=True)
save_results()

try:
    if DFLASH2_OK:
        parser_roundtrip("dflash2")
        run_load_phase("dflash2_conc28", 28, 90, 480)
        run_load_phase("dflash2_conc8", 8, 90, 480)
        if elapsed_min() <= DFLASH_C16_GATE_MIN:
            run_load_phase("dflash2_conc16", 16, 90, 480)
        else:
            RESULTS["phases"]["dflash2_conc16"] = {"skipped": "time-gate"}
        if not server_alive(10) and not vllm_procs():
            RESULTS["verdicts"]["q2_dflash2_crash"] = (
                "DFLASH2-FATAL (LOAD) — server died during matrix")
            print("serving-lab2: DFLASH2-FATAL (LOAD)", flush=True)
        if elapsed_min() <= BATTERY_C_GATE_MIN and server_alive(10):
            run_battery("battery_dflash2", wall_cap_min=10.0)
        else:
            RESULTS["phases"]["battery_dflash2"] = {"skipped": "time-gate"}
except Exception:
    traceback.print_exc()
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


print("\n" + "=" * 92)
print("SERVING LAB 2 DECISION TABLE — gen-tok/min/session (metric-based; usage fallback)")
print("=" * 92)
print(f"{'conc':>6} {'0.19 ref':>12} {'0.19 brief':>12} {'0.27':>12} "
      f"{'0.27/0.19':>10} {'dflash2':>12} {'df2/0.19':>10} {'df2 accept':>11}")
for _conc in (8, 16, 28):
    _brief = per_session_of(RESULTS["phases"].get("v019_conc28_brief")) if _conc == 28 else None
    _v27 = per_session_of(RESULTS["phases"].get(f"v027_conc{_conc}"))
    _df2p = RESULTS["phases"].get(f"dflash2_conc{_conc}")
    _df2 = per_session_of(_df2p)
    _ref = V019_REF[_conc]
    _r27 = round(_v27 / _ref, 2) if _v27 else None
    _rdf = round(_df2 / _ref, 2) if _df2 else None
    print(f"{_conc:>6} {_cell(_ref)} {_cell(_brief)} {_cell(_v27)} "
          f"{_cell(_r27, 10)} {_cell(_df2)} {_cell(_rdf, 10)} {_cell(_acc(_df2p), 11)}")

print("-" * 92)
_pv = {tag: RESULTS["phases"].get(f"parser_roundtrip_{tag}", {})
       for tag in ("v019", "v027", "dflash2")}
print("PARSER:  " + " | ".join(
    f"{tag} {p.get('verdict', 'N/A')} ({p.get('ok_count', '-')}/{p.get('total', '-')})"
    for tag, p in _pv.items()))

print("-" * 92)
print("QUALITY BATTERY (12 prompts, greedy temp 0, seed 1234; dead = finish=stop,")
print("no tool call, no content — the live sk48 failure signature):")
print(f"{'leg':>16} {'scored':>8} {'dead':>6} {'degen':>6} {'tool-calls':>11} {'length':>7}")
for _leg in ("battery_v019", "battery_v027", "battery_dflash2"):
    _b = RESULTS["phases"].get(_leg) or {}
    if "prompts_scored" not in _b:
        print(f"{_leg:>16} {'(skipped)':>8}")
        continue
    print(f"{_leg:>16} {_cell(_b['prompts_scored'], 8)} {_cell(_b['dead_completions'], 6)} "
          f"{_cell(_b['degenerate_loops'], 6)} {_cell(_b['tool_call_prompts'], 11)} "
          f"{_cell(_b['finish_length'], 7)}")

_ba = RESULTS["phases"].get("battery_v019") or {}
_bb = RESULTS["phases"].get("battery_v027") or {}
if "prompts_scored" in _ba and "prompts_scored" in _bb:
    _da, _db = _ba["dead_completions"], _bb["dead_completions"]
    _shared = [k for k in (_ba.get("items") or {})
               if "output_sha256" in (_ba["items"].get(k) or {})
               and "output_sha256" in ((_bb.get("items") or {}).get(k) or {})]
    _matched = sum(1 for k in _shared
                   if _ba["items"][k]["output_sha256"] == _bb["items"][k]["output_sha256"])
    RESULTS["verdicts"]["q3_quality"] = (
        f"dead-completions 0.19={_da} vs 0.27={_db} "
        f"({'0.27 BETTER' if _db < _da else '0.27 WORSE' if _db > _da else 'NO DIFF'}); "
        f"greedy outputs identical on {_matched}/{len(_shared)} shared prompts")
else:
    RESULTS["verdicts"]["q3_quality"] = "battery incomplete — see phases JSON"
print("Q3 VERDICT:", RESULTS["verdicts"]["q3_quality"])

print("-" * 92)
print("LANE RULE:", LANE_RULE)
_df28 = per_session_of(RESULTS["phases"].get("dflash2_conc28"))
_boot = RESULTS["phases"].get("dflash2_boot", {})
if not _boot.get("ok"):
    _lane = "LANE CLOSED (DFlash2 did not boot this probe)"
elif _df28 is None:
    _lane = "LANE NO-DATA (dflash2 conc-28 phase did not complete)"
elif _df28 >= LANE_RULE_TOKMIN:
    _lane = (f"LANE OPEN — dflash2 conc28 {_df28} >= {LANE_RULE_TOKMIN} "
             f"({_df28 / V019_REF[28]:.2f}x the 0.19 baseline)")
else:
    _lane = (f"LANE CLOSED — dflash2 conc28 {_df28} < {LANE_RULE_TOKMIN} "
             f"({_df28 / V019_REF[28]:.2f}x the 0.19 baseline)")
RESULTS["verdicts"]["q2_dflash2_lane"] = _lane
print("LANE VERDICT:", _lane)
if _boot.get("ok") and not _boot.get("spec_decode_active"):
    print("WARNING: DFlash2 booted but spec_decode counters never moved")

_att = RESULTS["phases"].get("dflash2_attest") or {}
print("DRAFT PROVENANCE:", _att.get("verdict", "not attested"))
if RESULTS["meta"]["deviations"]:
    print("DEVIATIONS:")
    for _d in RESULTS["meta"]["deviations"]:
        print("  -", _d)

RESULTS["meta"]["finished_utc"] = datetime.utcnow().isoformat() + "Z"
RESULTS["meta"]["total_elapsed_min"] = round(elapsed_min(), 1)
save_results()
print("=" * 92)
print(f"serving-lab2: DONE in {elapsed_min():.1f} min — results JSON: {RESULTS_PATH}")
print("=" * 92, flush=True)
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
    attest_cell = _load_attest_cell()

    # v2: no 0.19 server is booted (its leg is carried from v1), so strip the
    # attest cell's greedy decode fingerprint (it queries the live server) and
    # keep the pure weight-signature checks.
    marker = "# Greedy decode fingerprint"
    assert marker in attest_cell
    attest_cell = attest_cell.split(marker)[0] + (
        'print("attest: OK — official Qwen3.8-FP8 weight signature verified '
        '(decode fingerprint skipped: no 0.19 server in v2)")\n')

    dead_repro = json.loads(DEAD_REPRO.read_text(encoding="utf-8"))
    for key in ("reproducer_a", "reproducer_b"):
        assert dead_repro[key]["system"] and dead_repro[key]["user"], key
    lab_cell = CELL_LAB_LIB2_TEMPLATE.replace(
        "@@DEAD_REPRO@@", repr(json.dumps(dead_repro, separators=(",", ":"))))
    assert "@@" not in lab_cell

    v1_full = json.loads(V1_CARRY.read_text(encoding="utf-8"))
    v1_carry = {"phases": {name: v1_full["phases"][name] for name in
                           ("parser_roundtrip_v019", "battery_v019",
                            "v019_conc28_brief")}}
    assert v1_carry["phases"]["v019_conc28_brief"]["gen_tok_min_session_metric"] == 699.1
    assert v1_carry["phases"]["battery_v019"]["prompts_scored"] == 12
    carry_cell = CELL_V1_CARRY_TEMPLATE.replace(
        "@@V1_CARRY@@", repr(json.dumps(v1_carry, separators=(",", ":"))))
    assert "@@" not in carry_cell

    cells = [
        _md_cell(MD_HEADER),
        _code_cell(imports_cell),
        _code_cell(CELL_GPU_ASSERT),   # fail-fast: dies on P100 in minutes
        _code_cell(config_cell),
        _code_cell(audit_cell),
        _code_cell(CELL_AUDIT2),       # fail-fast: 0.27 wheelhouse + draft mounts
        _code_cell(attest_cell),       # doctrine v2: weights verified before measuring
        _code_cell(lab_cell),
        _code_cell(carry_cell),        # 0.19 leg banked from v1 — no bundle setup
        _code_cell(CELL_UPGRADE),
        _code_cell(CELL_PHASE1B),
        _code_cell(CELL_PHASE2),
        _code_cell(CELL_PHASE3),
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

    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        try:
            ast.parse(src)
        except SyntaxError as exc:
            raise SystemExit(f"cell {i} failed ast.parse: {exc}") from exc

    joined = "\n".join("".join(c["source"]) for c in notebook["cells"])
    # Ordering + content invariants (hard-won):
    assert joined.index("WRONG-GPU") < joined.index(MARK_CONFIG)
    assert joined.index("attest: OK") < joined.index("SERVING LAB 2 LIBRARY")
    assert '"method": "dflash"' in joined
    assert "--no-enable-prefix-caching" in joined
    assert "num_speculative_tokens" in joined
    assert "67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c" in joined
    assert "reproducer_a" in joined and "reproducer_b" in joined
    assert '"--kv-cache-dtype"' not in joined  # KV stays bf16
    # v2/v3 invariants: arch + CUDA_HOME fixes + v1 carry present, no bundle setup:
    assert "TORCH_CUDA_ARCH_LIST" in joined
    assert '"12.0f"' in joined            # suffix bypasses both version checks
    assert '"CUDA_HOME"' in joined        # nvcc-less pip dir → torch fallback
    assert "VLLM_USE_FLASHINFER_SAMPLER" in joined
    assert '"requires CUDA"' in joined and '"check_cuda_arch"' in joined
    assert "carried_from" in joined and "699.1" in joined
    assert MARK_SETUP not in joined  # the 0.19 serve chain is NOT booted in v2
    # v1 carry must land before the 0.27 upgrade cell:
    assert joined.index("carried v1 0.19-leg results") < joined.index("_info = install_vllm_0271()")
    # Parser round-trip must precede the 0.27 load phases:
    assert joined.index('parser_roundtrip("v027")') < joined.index('"v027_conc8"')

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
        "enable_tpu": False,
        "enable_internet": False,
        "machine_shape": "NvidiaRtxPro6000",
        "keywords": ["gpu"],
        "dataset_sources": [
            "driessmit1/arc3-vllm-h100-wheelhouse-v3",
            "jakobbrggen/taaf-kaggle-source-anim-20260807-anim",
            "saltb0x/arc3-vllm-wheelhouse-v0271-cu129",
            "bbucxi/qwen3-8-27b-dflash2",
        ],
        "kernel_sources": [],
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
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
