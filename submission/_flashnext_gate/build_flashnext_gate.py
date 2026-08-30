#!/usr/bin/env python3
"""build_flashnext_gate.py — can sonpham's Qwen3.8-Flash-Next NVFP4 package be
SERVED on ONE Kaggle RTX Pro 6000, and how fast is it on duck-shaped load
vs our Qwen3.8-27B-FP8 baseline? (NO games, NO submission.)

What their package actually is (decoded 2026-08-30 from the part-A
source-bundle — kaggle_flashnext_setup.py + kaggle_flashnext_preconverted_setup.py
+ FLASHNEXT_PACKAGE_LOCK.json + container-requirements.freeze):

  * Runtime = a pinned **vLLM dev wheel** `0.1.dev20073+g8e685d198`
    (torch 2.13.0+cu130, transformers 5.15.1, CUDA 13.0) shipped as the
    6 GB `flashnext-gcp-container-site-packages.tar.zst` (extracted
    site-packages of container vllm/vllm-openai@sha256:fc120ece...).
    The model card's SGLang qwen4_exp instructions are for the upstream HF
    checkpoint; sonpham's Kaggle package does NOT use SGLang.
  * Single-GPU mechanism = `--tensor-parallel-size 1` +
    `--gpu-memory-utilization 0.96` + **`VLLM_PLE_CPU_OFFLOAD=1`**: the ten
    model-plebf16-*.safetensors PLE n-gram tables (~104 GB BF16) live in host
    RAM; GPU-resident weights are ~82 GB of the 186 GB payload. Their
    FLASHNEXT_PACKAGE_LOCK.json records this exact serving shape (tp1,
    max_num_seqs 22, max_model_len 32768, kv auto/BF16, mtp_enabled false,
    ple_cpu_offload true) as the config behind their GCP mean 9.63.
  * Their preconverted wrapper verifies a PRIVATE Kaggle MODEL mount
    (sonphamorg/qwen3-8-flash-next-plebf16-gcp-exact — 403 for us), so this
    kernel assembles the same flat model view by symlink-union of the three
    public serving-part datasets and then launches THEIR argv + env verbatim.

Kernel phases (every phase failure-wrapped; partial results always land in
/kaggle/working/flashnext_gate_results.json):
  0     nvidia-smi (driver FIRST), df -h, free -g, /kaggle/input mounts,
        fail-fast RTX-Pro-6000 assert
  1     ASSEMBLE — verify+extract their runtime tarball with their bundled
        zstd (both sha-pinned), import-check the pinned vLLM, symlink-union
        serving-part-000/001/002 into one model dir, index-coverage assert,
        PLE-conversion provenance check vs FLASHNEXT_GCP_MODEL_INFO.json
  2     BOOT — their launch_server argv verbatim (tp1, util 0.96, seqs 22,
        32k, qwen3_xml parser), serving_env() verbatim incl.
        VLLM_PLE_CPU_OFFLOAD=1; 1800 s readiness (their wait_server number);
        one pre-registered retry rung (util 0.92 / 16k / 12 seqs) on failure
  3     SMOKE — /v1/models; parser round-trip (python tool, qwen3_xml);
        one-image multimodal probe; 20-prompt tool battery (parse-rate gate);
        20-prompt greedy battery (sanity)
  4     LOAD — duck-shaped sessions (lab1/lab4 generator, stable regime):
        conc-1 probe, conc-8, conc-28; churn conc-28 if time allows;
        gen tok/s aggregate + tok/min/session + prefix-hit + KV usage +
        spec-decode counters (expected zero: their MTP_ENABLED=0)
  5     VERDICT — pre-registered gate + comparison table vs the 27B baseline
        (serving_lab3 2026-08-26, same GPU pool)

Pre-registered gate (README.md):
  GATE PASS = boots on 1 GPU (either rung) AND stable conc-28 gen tok/s
  aggregate >= 450 (>= ~1.5x the 27B's 296.8) AND tool-call parse >= 0.95.

Load generator + measurement core are REUSED from arc3-serving-lab (lab1) via
block replacement, same recipe as lab4 — see _lab_library().

Usage:
  .venv/bin/python submission/_flashnext_gate/build_flashnext_gate.py
  .venv/bin/python submission/_flashnext_gate/validate_flashnext_gate.py
  # DO NOT push without Ahmed's go: this build is local-only.
"""
import ast
import hashlib
import json
import runpy
from pathlib import Path

HERE = Path(__file__).parent
LAB1_BUILD = HERE.parent / "_serving_lab" / "build_serving_lab.py"
KERNEL_SLUG = "arc3-flashnext-gate"

# ---- pins copied from their kaggle_flashnext_setup.py (part A source-bundle) --
EXPECTED_RUNTIME_SHA256 = "c06a78d59a74ac278dc2278d26dde6c70c48a4e28bb91fd4fbbefff4484e10f3"
EXPECTED_ZSTD_SHA256 = "7c5468b370f7c47eda07281e3437fafc568f95d10420051e3aa522709f9342c5"
EXPECTED_VERSIONS_LINE = "0.1.dev20073+g8e685d198 2.13.0+cu130 5.15.1 13.0"
MODEL_ID = "RadixArk/Qwen3.8-Flash-Next-NVFP4"
MODEL_REVISION = "7b719225242aacd3dbd3f9407468c2ee9a9d2594"
RUNTIME_ARCHIVE = "flashnext-gcp-container-site-packages.tar.zst"
EXPECTED_SAFETENSOR_FILES = 206      # 192 expert + 4 bf16 + 10 plebf16
EXPECTED_PAYLOAD_MIN_BYTES = 186_000_000_000

# ---- 27B baseline (submission/_serving_lab3/results-2026-08-26.json, same GPU pool) --
BASELINE_27B = {
    "source": "arc3-serving-lab3 2026-08-26 (vLLM 0.19, 27B-FP8, RTX Pro 6000)",
    "conc1": {"agg_tok_s": 40.6, "tok_min_session": 2435.0},
    "conc8": {"agg_tok_s": 215.9, "tok_min_session": 1619.4},
    "conc28": {"agg_tok_s": 296.8, "tok_min_session": 636.1, "prompt_mean": 21539},
}
GATE_CONC28_TOK_S = 450.0
GATE_TOOL_PARSE = 0.95


MD_HEADER = """\
# arc3-flashnext-gate — serve sonpham's Flash-Next NVFP4 on ONE RTX Pro 6000 (NO games, NO submission)

Single question, one commit, budget <= 2.5 h: does the sonphamorg
arc3-flashnext-serving package (Qwen3.8-Flash-Next NVFP4, ~180B MoE, 186 GB
payload) boot on one 96 GB GPU via their own vLLM-dev + PLE-CPU-offload
recipe, and how fast is it on duck-shaped load vs our 27B-FP8 baseline?

| Phase | What | Budget |
|---|---|---|
| 0 | nvidia-smi driver FIRST, df/free, mounts, RTX-Pro-6000 fail-fast assert | 1 min |
| 1 | verify + extract their runtime tarball (their zstd, both sha-pinned); symlink-union serving-part-000/001/002; index-coverage + PLE provenance checks | ~10-15 min |
| 2 | boot THEIR argv verbatim (tp1, util 0.96, 22 seqs, 32k, kv auto, qwen3_xml) with VLLM_PLE_CPU_OFFLOAD=1; 1800 s readiness; one retry rung (0.92/16k/12) | ~10-25 min |
| 3 | /v1/models; parser round-trip; one-image multimodal probe; 20-prompt tool battery; 20-prompt greedy battery | ~10 min |
| 4 | stable conc-1 probe / conc-8 / conc-28; churn conc-28 if time — tok/s agg, tok/min/session, prefix hit, KV usage, spec counters | ~30 min |
| 5 | pre-registered gate + table vs 27B baseline | 1 min |

**Pre-registered decision rules**

* **GATE PASS** = boots on 1 GPU (either rung) AND stable conc-28 gen tok/s
  aggregate **>= 450** (>= ~1.5x the 27B's 296.8) AND tool-call parse **>= 0.95**.
* Boot on the reduced rung only (util 0.92 / 16k ctx / 12 seqs) = boot PASS
  with a DEGRADED-CONFIG flag (their exact config did not fit).
* Spec-decode counters are expected ZERO (their package ships MTP_ENABLED=0).

27B baseline for scale (lab3 08-26, same GPU): conc-28 296.8 tok/s agg /
636.1 tok/min/session at prompt ~21.5k; conc-8 215.9 / 1619.4; conc-1 40.6.
"""


CELL_IMPORTS = r'''# ================= flashnext-gate — imports, results sink, DRIVER + HOST + MOUNTS FIRST =================
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

NOTEBOOK_START_EPOCH = time.time()
WORKING_DIR = Path("/kaggle/working")
WORKING_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = WORKING_DIR / "flashnext_gate_results.json"
RESULTS = {
    "meta": {"kernel": "arc3-flashnext-gate",
             "started_utc": datetime.utcnow().isoformat() + "Z"},
    "boots": {},
    "phases": {},
    "verdicts": {},
}


def elapsed_min():
    return (time.time() - NOTEBOOK_START_EPOCH) / 60.0


def save_results():
    tmp = RESULTS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(RESULTS, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(RESULTS_PATH)


def host_snapshot(tag):
    snap = {}
    try:
        mem = dict((ln.split(":", 1)[0], ln.split(":", 1)[1].strip())
                   for ln in Path("/proc/meminfo").read_text().splitlines() if ":" in ln)
        snap["mem_total"] = mem.get("MemTotal")
        snap["mem_available"] = mem.get("MemAvailable")
    except Exception as exc:
        snap["meminfo_error"] = repr(exc)[:120]
    for mount in ("/kaggle/working", "/kaggle/tmp", "/tmp", "/"):
        try:
            usage = shutil.disk_usage(mount)
            snap[mount] = f"free {usage.free / 1e9:.1f} / total {usage.total / 1e9:.1f} GB"
        except Exception:
            snap[mount] = "n/a"
    RESULTS["meta"].setdefault("host", {})[tag] = snap
    return snap


# ---- DRIVER VERSION FIRST, then host + mounts ----
_first = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
     "--format=csv,noheader"], capture_output=True, text=True)
GPU_LINE = (_first.stdout or "").strip()
print("flashnext-gate: nvidia-smi name,driver,memory =", repr(GPU_LINE),
      "rc=", _first.returncode, flush=True)
_smi_full = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
print("\n".join((_smi_full.stdout or "").splitlines()[:12]), flush=True)
_parts = [p.strip() for p in GPU_LINE.split(",")] if GPU_LINE else []
GPU_NAME = _parts[0] if _parts else ""
DRIVER_VERSION = _parts[1] if len(_parts) > 1 else ""
RESULTS["meta"]["gpu_name"] = GPU_NAME
RESULTS["meta"]["driver_version"] = DRIVER_VERSION
RESULTS["meta"]["python"] = sys.version.split()[0]

for _cmd in (["df", "-h"], ["free", "-g"]):
    _r = subprocess.run(_cmd, capture_output=True, text=True)
    print(f"flashnext-gate: {' '.join(_cmd)}\n" + (_r.stdout or "")[:1500], flush=True)
print("flashnext-gate: host snapshot", json.dumps(host_snapshot("start"), indent=1), flush=True)

_tree = subprocess.run(["find", "/kaggle/input", "-maxdepth", "3", "-mindepth", "1",
                        "-not", "-path", "*/serving-part-00*/*",
                        "-not", "-path", "*/source-bundle/src*"],
                       capture_output=True, text=True)
print("flashnext-gate: /kaggle/input tree (trimmed):\n" + (_tree.stdout or "")[:5000], flush=True)
RESULTS["meta"]["input_tree_head"] = (_tree.stdout or "").splitlines()[:80]
save_results()

# ---- FAIL-FAST GPU ASSERT: only the RTX Pro 6000 pool answers the question ----
if _first.returncode != 0 or not GPU_NAME:
    raise RuntimeError("WRONG-GPU: nvidia-smi failed — no usable GPU. Aborting fast.")
if "6000" not in GPU_NAME.upper() or "RTX" not in GPU_NAME.upper():
    raise RuntimeError(f"WRONG-GPU: expected an RTX Pro 6000, got {GPU_NAME!r}. Aborting fast.")
print(f"flashnext-gate: GPU assert PASS ({GPU_NAME}, driver {DRIVER_VERSION}, "
      f"python {RESULTS['meta']['python']})", flush=True)
'''


CELL_ASSEMBLE = r'''# ================= Phase 1 — ASSEMBLE: their runtime + symlink-union model view =================
# Pins below are copied verbatim from sonpham's kaggle_flashnext_setup.py
# (part-A source-bundle). Their preconverted wrapper verifies a PRIVATE Kaggle
# MODEL mount, so we assemble the identical flat view from the three public
# serving-part datasets and reuse everything else of their serve chain.
EXPECTED_RUNTIME_SHA256 = "@@RUNTIME_SHA@@"
EXPECTED_ZSTD_SHA256 = "@@ZSTD_SHA@@"
EXPECTED_VERSIONS_LINE = "@@VERSIONS_LINE@@"
RUNTIME_ARCHIVE = "@@RUNTIME_ARCHIVE@@"
QWEN_SERVED_MODEL_NAME = "@@MODEL_ID@@"
MODEL_REVISION = "@@MODEL_REVISION@@"
INPUT_ROOT = Path("/kaggle/input")

# scratch root: the extracted runtime is ~15 GB — keep it OFF /kaggle/working
# (its ~20 GB doubles as the preserved-output volume).
def _pick_scratch():
    best, best_free = Path("/tmp"), 0
    for cand in (Path("/kaggle/tmp"), Path("/kaggle/temp"), Path("/tmp")):
        try:
            cand.mkdir(parents=True, exist_ok=True)
            free = shutil.disk_usage(str(cand)).free
        except Exception:
            continue
        if free > best_free:
            best, best_free = cand, free
    return best, best_free


SCRATCH_ROOT, _scratch_free = _pick_scratch()
RUNTIME_ROOT = SCRATCH_ROOT / "flashnext-gcp-runtime"
SITE_PACKAGES = RUNTIME_ROOT / "dist-packages"
MODEL_DIR = SCRATCH_ROOT / "flashnext-model"
QWEN_MODEL_PATH = MODEL_DIR
print(f"flashnext-gate: scratch root {SCRATCH_ROOT} (free {_scratch_free / 1e9:.1f} GB)", flush=True)
RESULTS["meta"]["scratch_root"] = str(SCRATCH_ROOT)


def sha256_file(path):
    digest = __import__("hashlib").sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def serving_env():
    # VERBATIM from their kaggle_flashnext_setup.py serving_env() — including
    # VLLM_PLE_CPU_OFFLOAD=1 (the single-GPU mechanism: ~104 GB of PLE n-gram
    # tables live in host RAM) and TORCH_CUDA_ARCH_LIST=12.0f (Blackwell).
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(SITE_PACKAGES)
        if not current_pythonpath
        else f"{SITE_PACKAGES}{os.pathsep}{current_pythonpath}"
    )
    packaged_library_dirs = sorted(
        str(path) for path in (SITE_PACKAGES / "nvidia").glob("*/lib") if path.is_dir()
    )
    system_library_dirs = [
        "/usr/local/nvidia/lib64",
        "/usr/local/cuda/lib64",
        "/usr/local/nvidia/lib",
    ]
    current_ld = [entry for entry in env.get("LD_LIBRARY_PATH", "").split(os.pathsep) if entry]
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        dict.fromkeys(packaged_library_dirs + system_library_dirs + current_ld)
    )
    current_path = [entry for entry in env.get("PATH", "").split(os.pathsep) if entry]
    env["PATH"] = os.pathsep.join(
        dict.fromkeys(["/usr/local/nvidia/bin", "/usr/local/cuda/bin"] + current_path)
    )
    env.update(
        {
            "USE_TF": "0",
            "TRANSFORMERS_NO_TF": "1",
            "TRANSFORMERS_NO_TORCHVISION": "1",
            "VLLM_NO_USAGE_STATS": "1",
            "VLLM_ENABLE_CUDA_COMPATIBILITY": "0",
            "VLLM_PLE_CPU_OFFLOAD": "1",
            "VLLM_PLE_OFFLOAD_READY_TIMEOUT": "1800",
            "PYTORCH_ALLOC_CONF": "expandable_segments:False",  # v2: True needs pidfd_getfd CUDA-IPC, blocked by Kaggle seccomp (v1 boot death in the PLE offload worker)
            "HF_HUB_OFFLINE": "1",
        }
    )
    # v3: "12.0f" made the tarball's flashinfer filter out every major-12 arch
    # ("No supported CUDA architectures found for major versions [12]" in the
    # sm120 fused-MoE JIT). Ladder now controls the arch env: None = unset
    # (flashinfer derives SM 12.0 from the device), or an explicit string.
    env.pop("TORCH_CUDA_ARCH_LIST", None)
    arch = ARCH_OVERRIDE.get("value")
    if arch:
        env["TORCH_CUDA_ARCH_LIST"] = arch
    return env


ARCH_OVERRIDE = {"value": None}


RUNTIME_OK = False
ASSEMBLE = {}
try:
    _t0 = time.time()
    # ---- locate mounts (any nesting: /kaggle/input/<slug> or /kaggle/input/datasets/<user>/<slug>)
    _shards = {}
    for _name in ("serving-part-000", "serving-part-001", "serving-part-002"):
        _hits = sorted({p.resolve() for p in INPUT_ROOT.rglob(_name) if p.is_dir()})
        if len(_hits) != 1:
            raise FileNotFoundError(f"expected exactly one mounted {_name}, got {_hits}")
        _shards[_name] = _hits[0]
    _bundles = sorted({p.parent.resolve() for p in INPUT_ROOT.rglob("source-bundle/zstd")})
    if len(_bundles) != 1:
        raise FileNotFoundError(f"expected exactly one source-bundle, got {_bundles}")
    BUNDLE_DIR = _bundles[0]
    _archives = sorted({p.resolve() for p in INPUT_ROOT.rglob(RUNTIME_ARCHIVE)})
    if not _archives:
        raise FileNotFoundError(f"runtime archive {RUNTIME_ARCHIVE} not mounted")
    _preferred = [p for p in _archives if "runtime-exact" in str(p)]
    RUNTIME_TARBALL = (_preferred or _archives)[0]
    ASSEMBLE["shard_dirs"] = {k: str(v) for k, v in _shards.items()}
    ASSEMBLE["bundle_dir"] = str(BUNDLE_DIR)
    ASSEMBLE["runtime_tarball"] = str(RUNTIME_TARBALL)
    print("flashnext-gate: shards", ASSEMBLE["shard_dirs"], flush=True)
    print("flashnext-gate: bundle", BUNDLE_DIR, "| tarball", RUNTIME_TARBALL, flush=True)

    # ---- their runtime install, verbatim semantics (sha-pinned tarball + zstd) ----
    _sha = sha256_file(RUNTIME_TARBALL)
    if _sha != EXPECTED_RUNTIME_SHA256:
        raise RuntimeError(f"runtime archive drift: {_sha} != {EXPECTED_RUNTIME_SHA256}")
    _zstd_src = BUNDLE_DIR / "zstd"
    _zsha = sha256_file(_zstd_src)
    if _zsha != EXPECTED_ZSTD_SHA256:
        raise RuntimeError(f"bundled zstd drift: {_zsha} != {EXPECTED_ZSTD_SHA256}")
    _tools = SCRATCH_ROOT / "packaging-tools"
    _tools.mkdir(parents=True, exist_ok=True)
    ZSTD_TOOL = _tools / "zstd"
    shutil.copy2(_zstd_src, ZSTD_TOOL)
    ZSTD_TOOL.chmod(0o755)
    shutil.rmtree(RUNTIME_ROOT, ignore_errors=True)
    RUNTIME_ROOT.mkdir(parents=True)
    _tx = time.time()
    subprocess.run(["tar", f"--use-compress-program={ZSTD_TOOL}", "-xf",
                    str(RUNTIME_TARBALL), "-C", str(RUNTIME_ROOT)], check=True)
    ASSEMBLE["extract_s"] = round(time.time() - _tx, 1)
    if not (SITE_PACKAGES / "vllm").is_dir():
        raise RuntimeError(f"extracted runtime incomplete: {SITE_PACKAGES}")
    _probe = subprocess.run(
        [sys.executable, "-c",
         ("import platform,torch,transformers,vllm; "
          "print(platform.python_version(),vllm.__version__,torch.__version__,"
          "transformers.__version__,torch.version.cuda)")],
        env=serving_env(), capture_output=True, text=True)
    ASSEMBLE["runtime_import"] = (_probe.stdout or "").strip()[:200]
    print("flashnext-gate: exact runtime import:", ASSEMBLE["runtime_import"], flush=True)
    if _probe.returncode:
        raise RuntimeError(f"exact runtime import failed:\n{(_probe.stderr or '')[-3000:]}")
    if EXPECTED_VERSIONS_LINE not in (_probe.stdout or ""):
        raise RuntimeError(f"exact serving versions drifted: {ASSEMBLE['runtime_import']}")

    # ---- symlink-union model view (their zero-copy layout, dataset-mount flavour) ----
    shutil.rmtree(MODEL_DIR, ignore_errors=True)
    MODEL_DIR.mkdir(parents=True)
    _seen = {}
    for _name in sorted(_shards):
        for _member in sorted(_shards[_name].iterdir()):
            if not _member.is_file():
                raise RuntimeError(f"unexpected nested entry in {_name}: {_member}")
            if _member.name in _seen:
                raise RuntimeError(f"duplicate file across shards: {_member.name} "
                                   f"({_seen[_member.name]} vs {_name})")
            _seen[_member.name] = _name
            if _member.name == "model.safetensors.index.json":
                shutil.copy2(_member, MODEL_DIR / _member.name)
            else:
                (MODEL_DIR / _member.name).symlink_to(_member)
    _st_files = sorted(MODEL_DIR.glob("*.safetensors"))
    _st_bytes = sum(p.stat().st_size for p in _st_files)
    ASSEMBLE["model_files"] = len(_seen)
    ASSEMBLE["safetensor_files"] = len(_st_files)
    ASSEMBLE["safetensor_bytes"] = _st_bytes
    if len(_st_files) != @@EXPECTED_SAFETENSOR_FILES@@:
        raise RuntimeError(f"expected @@EXPECTED_SAFETENSOR_FILES@@ safetensors, got {len(_st_files)}")
    if _st_bytes < @@EXPECTED_PAYLOAD_MIN_BYTES@@:
        raise RuntimeError(f"payload too small: {_st_bytes}")
    _index = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())
    _needed = sorted(set((_index.get("weight_map") or {}).values()))
    _missing = [n for n in _needed if not (MODEL_DIR / n).is_file()]
    ASSEMBLE["index_files_referenced"] = len(_needed)
    if _missing:
        raise RuntimeError(f"index references missing files: {_missing[:8]} "
                           f"(+{max(0, len(_missing) - 8)} more)")
    if list(MODEL_DIR.glob("model-plefp8-*.safetensors")):
        raise RuntimeError("serving view still contains FP8 PLE source files")

    # ---- PLE-conversion provenance vs the GCP winner (their check, hashes-by-manifest) ----
    _expected = json.loads((BUNDLE_DIR / "FLASHNEXT_GCP_MODEL_INFO.json").read_text())["ple_conversion"]
    _observed = json.loads((MODEL_DIR / "ple-bf16-conversion.json").read_text())
    _exp_files = {i["target"]: (int(i["target_bytes"]), i["target_sha256"]) for i in _expected["files"]}
    _obs_files = {i["target"]: (int(i["target_bytes"]), i["target_sha256"]) for i in _observed["files"]}
    if _obs_files != _exp_files or _observed.get("index_sha256") != _expected.get("index_sha256"):
        raise RuntimeError("PLE conversion provenance differs from the GCP winner")
    for _tname, (_tbytes, _sha_unused) in _exp_files.items():
        if (MODEL_DIR / _tname).stat().st_size != _tbytes:
            raise RuntimeError(f"PLE file size drift: {_tname}")
    ASSEMBLE["ple_files_verified"] = len(_exp_files)
    _cfg = json.loads((MODEL_DIR / "config.json").read_text())
    ASSEMBLE["architectures"] = _cfg.get("architectures")
    ASSEMBLE["model_type"] = _cfg.get("model_type")
    ASSEMBLE["assemble_s"] = round(time.time() - _t0, 1)
    RUNTIME_OK = True
    print(f"flashnext-gate: ASSEMBLE OK in {ASSEMBLE['assemble_s']} s — "
          f"{len(_st_files)} safetensors / {_st_bytes / 1e9:.1f} GB, "
          f"arch {ASSEMBLE['architectures']}, elapsed {elapsed_min():.1f} min", flush=True)
except Exception as _exc:
    traceback.print_exc()
    ASSEMBLE["error"] = repr(_exc)[:800]
    RESULTS["verdicts"]["boot"] = "ASSEMBLE-FAILED: " + repr(_exc)[:300]
    print("flashnext-gate: ASSEMBLE FAILED — every later phase will be skipped", flush=True)
RESULTS["meta"]["assemble"] = ASSEMBLE
RESULTS["meta"]["model_path"] = str(QWEN_MODEL_PATH)
RESULTS["meta"]["served_model_name"] = QWEN_SERVED_MODEL_NAME
host_snapshot("after_assemble")
save_results()
'''


# ---------------------------------------------------------------------------
# Library blocks that REPLACE their lab1 counterparts (block replacement keeps
# the duck-shaped generator + measurement core verbatim — lab4 recipe).
# ---------------------------------------------------------------------------
LIB_HEADER = r'''# ============================ SERVING LAB LIBRARY (flashnext-gate) ============================
# No games are played in this kernel. Load generator + measurement core are the
# arc3-serving-lab / lab3 code (the pattern re-run on this exact GPU on 08-26)
# adapted for:
#   - sonpham's pinned vLLM-dev serve chain (their argv + serving_env verbatim)
#   - regime-aware sessions (stable = primary; churn = optional extra)
#   - prefix-cache hit rate + KV usage per measured window
#   - a tool-call parse battery (gate: parse >= 0.95 with qwen3_xml)
'''

LIB_CONSTANTS = r'''VLLM_HOST = "127.0.0.1"
VLLM_PORT = 1234
VLLM_ROOT = f"http://{VLLM_HOST}:{VLLM_PORT}"
VLLM_API = VLLM_ROOT + "/v1"
VLLM_MAX_MODEL_LEN = 32768   # their launch_server value

LAB_HARD_CAP_MIN = 135.0     # no NEW phase starts after this (kernel budget 150 min)
CHURN_GATE_MIN = 100.0       # optional churn conc-28 arm only before this

# ---- their launch_server argv VERBATIM (kaggle_flashnext_setup.py, kv "auto") ----
FLASH_SERVE_FLAGS = [
    "--model", str(QWEN_MODEL_PATH),
    "--served-model-name", QWEN_SERVED_MODEL_NAME,
    "--host", VLLM_HOST,
    "--port", str(VLLM_PORT),
    "--tensor-parallel-size", "1",
    "--distributed-executor-backend", "mp",
    "--gpu-memory-utilization", "0.96",
    "--max-model-len", "32768",
    "--max-num-seqs", "22",
    "--max-num-batched-tokens", "6144",
    "--kv-cache-dtype", "auto",
    "--enable-prefix-caching",
    "--no-enable-flashinfer-autotune",
    "--enable-auto-tool-choice",
    "--tool-call-parser", "qwen3_xml",
    "--generation-config", "vllm",
    "--default-chat-template-kwargs", '{"preserve_thinking": true}',
    "--reasoning-parser", "qwen3",
]
# pre-registered retry rung if their exact config fails (OOM or otherwise):
REDUCED_SERVE_FLAGS = [
    "--model", str(QWEN_MODEL_PATH),
    "--served-model-name", QWEN_SERVED_MODEL_NAME,
    "--host", VLLM_HOST,
    "--port", str(VLLM_PORT),
    "--tensor-parallel-size", "1",
    "--distributed-executor-backend", "mp",
    "--gpu-memory-utilization", "0.92",
    "--max-model-len", "16384",
    "--max-num-seqs", "12",
    "--max-num-batched-tokens", "6144",
    "--kv-cache-dtype", "auto",
    "--enable-prefix-caching",
    "--no-enable-flashinfer-autotune",
    "--enable-auto-tool-choice",
    "--tool-call-parser", "qwen3_xml",
    "--generation-config", "vllm",
    "--default-chat-template-kwargs", '{"preserve_thinking": true}',
    "--reasoning-parser", "qwen3",
]
# (tag, flags, TORCH_CUDA_ARCH_LIST override): None = unset -> flashinfer
# derives SM 12.0 from the device; "12.0a" = the CUTLASS sm120a spelling.
BOOT_LADDER = [("gcp_exact", FLASH_SERVE_FLAGS, None),
               ("gcp_exact_arch120a", FLASH_SERVE_FLAGS, "12.0a"),
               ("reduced", REDUCED_SERVE_FLAGS, None)]

BASELINE_27B = @@BASELINE_27B@@
GATE_CONC28_TOK_S = @@GATE_CONC28_TOK_S@@
GATE_TOOL_PARSE = @@GATE_TOOL_PARSE@@

IMAGES_ENABLED = {"on": True}   # flipped off if the server rejects image input

CURRENT_SERVER = {"proc": None, "log": str(WORKING_DIR / "vllm-gcp_exact.log"),
                  "tag": "none", "flags": []}

RESULTS["meta"].update({
    "max_model_len": VLLM_MAX_MODEL_LEN,
    "flash_serve_flags": FLASH_SERVE_FLAGS,
    "reduced_serve_flags": REDUCED_SERVE_FLAGS,
    "baseline_27b": BASELINE_27B,
    "serving_env_keys": ["VLLM_PLE_CPU_OFFLOAD=1", "VLLM_PLE_OFFLOAD_READY_TIMEOUT=1800",
                         "TORCH_CUDA_ARCH_LIST per boot ladder (None -> device detect; 12.0a fallback)",
                         "PYTORCH_ALLOC_CONF=expandable_segments:False"],
    "decision_rules": {
        "gate_pass": (f"boots on 1 GPU (either rung) AND stable conc-28 gen tok/s aggregate >= "
                      f"{GATE_CONC28_TOK_S} AND tool-call parse >= {GATE_TOOL_PARSE}"),
        "reduced_rung": "boot PASS with DEGRADED-CONFIG flag",
        "spec_decode": "expected zero — their package ships MTP_ENABLED=0",
    },
})


'''

LIB_SERVER = r'''# ---- server lifecycle (their serving_env, own process group) --------------------
ENGINE_LINE_PATTERNS = {
    "engine_config": "Initializing a V1 LLM engine",
    "non_default_args": "non-default args",
    "attention_backend": "attention backend",
    "kv_cache_size": "GPU KV cache size",
    "max_concurrency": "Maximum concurrency for",
    "kv_cache_memory": "Available KV cache memory",
    "model_load": "Model loading took",
    "ple_lower": "ple",
    "offload": "offload",
    "cuda_oom": "CUDA out of memory",
    "torch_oom": "OutOfMemoryError",
    "cuda_error": "CUDA error",
    "illegal_memory": "illegal memory access",
    "engine_dead": "EngineDeadError",
    "traceback": "Traceback (most recent call last)",
}


def capture_engine_lines(log_path, max_hits=4):
    lines = tail_log_lines(log_path, max_bytes=4_000_000)
    found = {}
    for key, needle in ENGINE_LINE_PATTERNS.items():
        if key == "ple_lower":
            hits = [ln.strip()[:1200] for ln in lines
                    if ("ple" in ln.lower() and ("offload" in ln.lower() or "PLE" in ln))]
        else:
            hits = [ln.strip()[:1200] for ln in lines if needle in ln]
        if hits:
            found[key] = {"count": len(hits), "lines": hits[:max_hits]}
    backend = None
    for ln in lines:
        m = re.search(r"Using (\S+) attention backend", ln)
        if m and backend is None:
            backend = m.group(1)
    found["attention_backend_name"] = backend
    found["oomish"] = bool(found.get("cuda_oom") or found.get("torch_oom"))
    return found


def print_engine_lines(tag, found):
    print(f"flashnext-gate: [{tag}] engine lines:", flush=True)
    for key in ("engine_config", "attention_backend", "kv_cache_size", "max_concurrency",
                "kv_cache_memory", "model_load", "ple_lower", "offload"):
        for ln in (found.get(key) or {}).get("lines", [])[:2]:
            print(f"  {key}: {ln[:1100]}", flush=True)
    print(f"  attention backend : {found.get('attention_backend_name')}", flush=True)
    for key in ("cuda_oom", "torch_oom", "cuda_error", "illegal_memory", "engine_dead"):
        for ln in (found.get(key) or {}).get("lines", [])[:2]:
            print(f"  !! {key}: {ln[:600]}", flush=True)


def stop_server(reason):
    print(f"flashnext-gate: stopping vLLM ({reason})", flush=True)
    proc = CURRENT_SERVER.get("proc")
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass
    subprocess.run(["pkill", "-TERM", "-f", "vllm.entrypoints"], check=False)
    deadline = time.time() + 90
    while time.time() < deadline and vllm_procs():
        time.sleep(3)
    if vllm_procs():
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
        subprocess.run(["pkill", "-9", "-f", "vllm.entrypoints"], check=False)
        time.sleep(10)
    deadline = time.time() + 240
    while time.time() < deadline:
        sample = gpu_sample()
        if sample is not None and sample["mem_mib"] < 8000:
            break
        time.sleep(5)
    CURRENT_SERVER["proc"] = None
    print(f"flashnext-gate: server stopped, gpu={gpu_sample()}", flush=True)


def served_ids():
    try:
        return [m.get("id") for m in http_json(VLLM_API + "/models", timeout=5).get("data", [])]
    except Exception:
        return None


def start_server(flags, tag, timeout_s=1800):
    """Boot the pinned vLLM with the given full argv in THEIR serving_env().
    1800 s readiness = their wait_server + VLLM_PLE_OFFLOAD_READY_TIMEOUT
    number. Raises on death/timeout with the log tail printed; the boot is
    recorded under RESULTS['boots'][tag] either way."""
    log_path = WORKING_DIR / f"vllm-{tag}.log"
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", *flags]
    print(f"flashnext-gate: starting vLLM [{tag}] (elapsed {elapsed_min():.1f} min):",
          " ".join(cmd), flush=True)
    handle = log_path.open("w", encoding="utf-8")
    t0 = time.time()
    proc = subprocess.Popen(cmd, env=serving_env(), stdout=handle, stderr=subprocess.STDOUT,
                            text=True, start_new_session=True)
    CURRENT_SERVER.update({"proc": proc, "log": str(log_path), "tag": tag, "flags": flags})
    boot = {"tag": tag, "flags": flags, "started_utc": datetime.utcnow().isoformat() + "Z"}
    RESULTS["boots"][tag] = boot
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            lines = tail_log_lines(log_path)
            print(f"flashnext-gate: [{tag}] BOOT FAILURE rc={proc.returncode} — log tail:", flush=True)
            print("\n".join(lines[-150:]), flush=True)
            boot.update({"ok": False, "rc": proc.returncode, "boot_s": round(time.time() - t0, 1),
                         "log_tail": lines[-60:], "engine_lines": capture_engine_lines(log_path)})
            print_engine_lines(tag, boot["engine_lines"])
            save_results()
            raise RuntimeError(f"vLLM [{tag}] died during startup rc={proc.returncode}")
        if server_alive():
            ids = served_ids()
            found = capture_engine_lines(log_path)
            boot.update({"ok": True, "boot_s": round(time.time() - t0, 1), "served_ids": ids,
                         "engine_lines": found, "gpu_after": gpu_sample(),
                         "host_after": host_snapshot(f"after_boot_{tag}")})
            print(f"flashnext-gate: vLLM ready [{tag}] in {boot['boot_s']} s, served ids {ids}",
                  flush=True)
            print_engine_lines(tag, found)
            print(f"flashnext-gate: [{tag}] gpu after boot {boot['gpu_after']} | "
                  f"host {boot['host_after']}", flush=True)
            if ids != [QWEN_SERVED_MODEL_NAME]:
                print(f"flashnext-gate: WARNING served ids {ids} != [{QWEN_SERVED_MODEL_NAME}]",
                      flush=True)
            save_results()
            return flags
        time.sleep(10)
    lines = tail_log_lines(log_path)
    print(f"flashnext-gate: [{tag}] BOOT TIMEOUT — log tail:", flush=True)
    print("\n".join(lines[-150:]), flush=True)
    boot.update({"ok": False, "rc": None, "timeout": True, "boot_s": round(time.time() - t0, 1),
                 "log_tail": lines[-60:], "engine_lines": capture_engine_lines(log_path)})
    save_results()
    raise TimeoutError(f"vLLM [{tag}] not ready in {timeout_s}s")


def reset_prefix_cache():
    # vLLM API server: POST /reset_prefix_cache. Best effort — recorded, never fatal.
    try:
        req = urllib.request.Request(VLLM_ROOT + "/reset_prefix_cache", data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status
    except Exception as exc:
        return ("failed: " + repr(exc))[:120]


'''

LIB_SESSION = r'''CHURN_CAP_TOKENS = 22000    # shipped duck: pop the oldest block past this (32k ctx headroom)
STABLE_CAP_TOKENS = 26000   # hysteresis regime: one big cut, rarely (26000 + 3072 gen < 32768)
SEED_TURNS = 12             # pre-seeded history: FIRST request ~17k tokens (lab3 calibration:
                            # real tokens ~1.13x the make_transcript estimate)


class Session:
    # One synthetic duck "game worker" in one of two prompt regimes:
    #   churn  — shipped-duck behaviour: after EVERY request the oldest history
    #            block is dropped, so the cached prefix stays short.
    #   stable — append-only; the image rides only on the newest turn; trimming
    #            is one big cut (half the history) and only past STABLE_CAP.
    # Both regimes start from the SAME layout (system + intro + SEED_TURNS
    # history turns) so prompt sizes are comparable.
    def __init__(self, idx, seed, regime="stable", images_per_req=1):
        self.rng = random.Random(seed)
        self.idx = idx
        self.regime = regime
        self.images_per_req = images_per_req if IMAGES_ENABLED["on"] else 0
        self.turns = []
        self.base_context = make_transcript(self.rng, 600)
        for _ in range(SEED_TURNS):
            user_msg = _strip_images(self._new_user_turn())
            reply = {"role": "assistant",
                     "content": ("Plan: " + make_transcript(self.rng, 320))[:1500]}
            self.turns.append((user_msg, reply))
        self.last_prompt_tokens = None
        self._pending_user = None
        self.trims = 0
        self.requests = 0

    def _new_user_turn(self):
        text = ("[turn] board updated; analyze the change and choose the next "
                "actions.\n" + make_transcript(self.rng, 700))
        content = [{"type": "text", "text": text}]
        if self.images_per_req > 0 and IMAGES_ENABLED["on"]:
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + board_png_b64(self.rng)}})
        return {"role": "user", "content": content}

    def build_messages(self):
        msgs = [{"role": "system", "content": SYSTEM_TEXT},
                {"role": "user", "content": [{"type": "text", "text":
                    "Game intro and rules so far:\n" + self.base_context}]},
                {"role": "assistant", "content": "Understood. World model initialized."}]
        total = len(self.turns)
        for i, (user_msg, assistant_msg) in enumerate(self.turns):
            if total - i > max(self.images_per_req - 1, 0):
                user_msg = _strip_images(user_msg)
            msgs.append(user_msg)
            msgs.append(assistant_msg)
        self._pending_user = self._new_user_turn()
        msgs.append(self._pending_user)
        return msgs

    def record(self, assistant_text, usage):
        reply = (assistant_text or "").strip()[:1500] or "(thinking only)"
        self.turns.append((self._pending_user, {"role": "assistant", "content": reply}))
        self.requests += 1
        tokens = usage.get("prompt_tokens")
        self.last_prompt_tokens = tokens
        if self.regime == "churn":
            if len(self.turns) > 2:
                self.turns.pop(0)          # ALWAYS drop the oldest block (prefix shifts)
                self.trims += 1
            while tokens and tokens > CHURN_CAP_TOKENS and len(self.turns) > 2:
                self.turns.pop(0)
                self.trims += 1
                tokens -= 1100
        else:
            if tokens and tokens > STABLE_CAP_TOKENS and len(self.turns) > 4:
                del self.turns[:len(self.turns) // 2]   # hysteresis: one big cut
                self.trims += 1


'''

LIB_LOAD_PHASE = r'''def agg_of(phase):
    if not phase:
        return None
    metric = phase.get("gen_tok_s_aggregate_metric")
    return metric if metric is not None else phase.get("gen_tok_s_aggregate_usage")


def run_load_phase(name, conc, warmup_s, measure_s, regime="stable"):
    if elapsed_min() > LAB_HARD_CAP_MIN:
        print(f"flashnext-gate: SKIP {name} — past hard cap ({elapsed_min():.0f} min)", flush=True)
        RESULTS["phases"][name] = {"skipped": "hard-cap", "regime": regime}
        save_results()
        return None
    if not server_alive(15):
        print(f"flashnext-gate: SKIP {name} — server not alive", flush=True)
        RESULTS["phases"][name] = {"skipped": "server-dead", "regime": regime}
        save_results()
        return None
    print(f"\nflashnext-gate: === {name} (regime={regime}, conc={conc}, warmup={warmup_s}s, "
          f"measure={measure_s}s, elapsed={elapsed_min():.1f} min, server={CURRENT_SERVER['tag']}) ===",
          flush=True)
    reset_status = reset_prefix_cache()
    events, errors, sessions = [], [], []
    lock = threading.Lock()
    stop = threading.Event()
    gpu_samples = []
    seed_base = zlib.crc32(name.encode("utf-8")) & 0xFFFF   # deterministic (hash() is salted)

    def sampler():
        while not stop.is_set():
            sample = gpu_sample()
            if sample:
                gpu_samples.append(sample)
            stop.wait(30)

    def worker(i):
        session = Session(i, seed=seed_base * 100 + i, regime=regime)
        with lock:
            sessions.append(session)
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
                    session.images_per_req = 0
                    IMAGES_ENABLED["on"] = False
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
            print(f"flashnext-gate: SERVER DIED during {name}", flush=True)
            break
    t1 = time.time()
    metrics_after, metric_lines = scrape_metrics()
    stop.set()
    for thread in threads:
        thread.join(timeout=2)
    if not server_died:
        drain_inflight()

    def delta(key):
        return metrics_after.get(key, 0.0) - metrics_before.get(key, 0.0)

    window = [e for e in events if t0 <= e["t_end"] <= t1]
    minutes = max((t1 - t0) / 60.0, 0.01)
    usage_gen = sum(e["completion_tokens"] for e in window)
    gen_delta = delta("vllm:generation_tokens")
    pc_q, pc_h = delta("vllm:prefix_cache_queries"), delta("vllm:prefix_cache_hits")
    latencies = sorted(e["latency_s"] for e in window)
    prompts = [e["prompt_tokens"] for e in window]
    result = {
        "conc": conc,
        "regime": regime,
        "server_tag": CURRENT_SERVER["tag"],
        "warmup_s": warmup_s,
        "measure_min": round(minutes, 2),
        "prefix_cache_reset_before_warmup": reset_status,
        "requests_in_window": len(window),
        "requests_total": len(events),
        "errors": len(errors),
        "error_samples": errors[:5],
        "gen_tok_min_session_metric": round(gen_delta / minutes / conc, 1) if gen_delta > 0 else None,
        "gen_tok_min_session_usage": round(usage_gen / minutes / conc, 1),
        "gen_tok_s_aggregate_metric": round(gen_delta / (minutes * 60.0), 1) if gen_delta > 0 else None,
        "gen_tok_s_aggregate_usage": round(usage_gen / (minutes * 60.0), 1),
        "prompt_tok_s_window": round(delta("vllm:prompt_tokens") / (minutes * 60.0), 1),
        "prefix_cache_queries_window": pc_q,
        "prefix_cache_hits_window": pc_h,
        "prefix_cache_hit_rate_window": round(pc_h / pc_q, 4) if pc_q > 0 else None,
        "preemptions_window": delta("vllm:num_preemptions"),
        "requests_running_at_end": metrics_after.get("vllm:num_requests_running"),
        "requests_waiting_at_end": metrics_after.get("vllm:num_requests_waiting"),
        "kv_cache_usage_at_end": (metrics_after.get("vllm:kv_cache_usage_perc")
                                  if metrics_after.get("vllm:kv_cache_usage_perc") is not None
                                  else metrics_after.get("vllm:gpu_cache_usage_perc")),
        "mean_latency_s": round(statistics.fmean(latencies), 1) if latencies else None,
        "p50_latency_s": round(statistics.median(latencies), 1) if latencies else None,
        "prompt_tokens_mean": round(statistics.fmean(prompts)) if prompts else None,
        "prompt_tokens_min": min(prompts) if prompts else None,
        "prompt_tokens_max": max(prompts) if prompts else None,
        "completion_tokens_mean": round(statistics.fmean(
            [e["completion_tokens"] for e in window])) if window else None,
        "session_trims_mean": round(statistics.fmean([s.trims for s in sessions]), 2) if sessions else None,
        "session_requests_mean": round(statistics.fmean([s.requests for s in sessions]), 2) if sessions else None,
        "images_enabled": IMAGES_ENABLED["on"],
        "acceptance": acceptance_delta(metrics_before, metrics_after),
        "acceptance_log_lines": log_acceptance_lines(),
        "metric_lines_sample": metric_lines[:12],
        "gpu_samples_tail": gpu_samples[-6:],
        "server_died": server_died,
        "server_alive_at_end": server_alive(),
    }
    RESULTS["phases"][name] = result
    save_results()
    print(f"flashnext-gate: {name}: {agg_of(result)} tok/s agg | "
          f"{per_session_of(result)} gen-tok/min/session | "
          f"prefix hit {result['prefix_cache_hit_rate_window']} | "
          f"kv {result['kv_cache_usage_at_end']} | "
          f"{len(window)} reqs / {len(errors)} errs | prompt~{result['prompt_tokens_mean']} | "
          f"p50 lat {result['p50_latency_s']}s | trims/session {result['session_trims_mean']}", flush=True)
    return result


'''

LIB_BATTERY = r'''# ---- tool battery (gate: qwen3_xml parse >= 0.95) + greedy sanity battery ----
BATTERY_N = 20
BATTERY_MAX_TOKENS = 224
TOOLB_N = 20


def run_tool_battery(tag):
    """20 prompts, all carrying the python tool: 10 with tool_choice forced,
    10 auto with an explicit instruction to call it. Parse attempts = the 10
    forced + every auto response that emitted (or tried to emit) a tool call;
    a forced response with no parsed call counts as a failure."""
    name = f"tool_battery_{tag}"
    if not server_alive(15):
        RESULTS["phases"][name] = {"skipped": "server-dead"}
        save_results()
        return None
    print(f"flashnext-gate: === {name} ({TOOLB_N} prompts, elapsed {elapsed_min():.1f} min) ===",
          flush=True)
    t0 = time.time()
    outs = []
    for k in range(TOOLB_N):
        rng = random.Random(31000 + k)
        forced = (k % 2 == 0)
        ask = ("Use the python tool NOW: call it with code that prints "
               f"grid[{k % 5}][{k % 7}] and nothing else.")
        payload = {
            "model": QWEN_SERVED_MODEL_NAME,
            "messages": [
                {"role": "system", "content":
                    "You are an ARC-AGI-3 analyst. Use the python tool to act."},
                {"role": "user", "content": "Board state:\n" + make_transcript(rng, 900)
                    + "\n\n" + ask},
            ],
            "tools": PYTHON_TOOL,
            "temperature": 1.0, "top_p": 0.95, "top_k": 20,
            # generous cap: reasoning must not eat the budget before the call
            "max_tokens": 3000,
            "chat_template_kwargs": {"enable_thinking": True},
        }
        if forced:
            payload["tool_choice"] = {"type": "function", "function": {"name": "python"}}
        rec = {"k": k, "forced": forced}
        try:
            resp = http_json(VLLM_API + "/chat/completions", payload, timeout=900)
            message = resp["choices"][0]["message"]
            calls = message.get("tool_calls") or []
            content = message.get("content") or ""
            attempted = forced or bool(calls) or ("<tool_call>" in content)
            parsed = False
            if calls:
                fn = calls[0].get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                    parsed = fn.get("name") == "python" and isinstance(args.get("code"), str)
                except Exception:
                    parsed = False
            rec.update({"attempted": attempted, "parsed": parsed,
                        "n_tool_calls": len(calls),
                        "finish_reason": resp["choices"][0].get("finish_reason"),
                        "content_head": content[:160] if not parsed else ""})
        except Exception as exc:
            rec.update({"attempted": forced, "parsed": False, "error": repr(exc)[:200]})
        outs.append(rec)
    attempts = sum(1 for o in outs if o.get("attempted"))
    parsed_n = sum(1 for o in outs if o.get("parsed"))
    rate = round(parsed_n / attempts, 4) if attempts else None
    result = {"tag": tag, "n": len(outs), "attempts": attempts, "parsed": parsed_n,
              "parse_rate": rate, "elapsed_s": round(time.time() - t0, 1),
              "outputs": outs}
    RESULTS["phases"][name] = result
    RESULTS["verdicts"]["tool_parse_rate"] = rate
    save_results()
    print(f"flashnext-gate: {name}: parse {parsed_n}/{attempts} = {rate} "
          f"(gate >= {GATE_TOOL_PARSE}) in {result['elapsed_s']} s", flush=True)
    return result


def battery_prompts():
    prompts = []
    for k in range(BATTERY_N):
        rng = random.Random(4242 + k)
        text = make_transcript(rng, 2200)
        tool_mode = (k % 4 == 3)     # 5 of 20 carry the python tool (parser path)
        ask = ("use the python tool to print grid[0][0] before you act."
               if tool_mode else "state the next three actions with one-line justifications.")
        content = [{"type": "text", "text": "Game transcript so far:\n" + text
                    + "\n\nAnalyze the latest board change and " + ask}]
        if IMAGES_ENABLED["on"]:
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + board_png_b64(rng)}})
        msgs = [{"role": "system", "content": SYSTEM_TEXT},
                {"role": "user", "content": content}]
        prompts.append({"k": k, "messages": msgs, "tools": tool_mode})
    return prompts


def run_battery(tag, max_tokens=BATTERY_MAX_TOKENS):
    name = f"battery_{tag}"
    if elapsed_min() > LAB_HARD_CAP_MIN:
        RESULTS["phases"][name] = {"skipped": "hard-cap"}
        save_results()
        return None
    if not server_alive(15):
        RESULTS["phases"][name] = {"skipped": "server-dead"}
        save_results()
        return None
    print(f"flashnext-gate: === {name} (greedy, {BATTERY_N} prompts, sequential, "
          f"max_tokens {max_tokens}, elapsed {elapsed_min():.1f} min) ===", flush=True)
    t0 = time.time()
    outs = []
    for p in battery_prompts():
        payload = {"model": QWEN_SERVED_MODEL_NAME, "messages": p["messages"],
                   "temperature": 0.0, "seed": 0, "max_tokens": max_tokens,
                   "chat_template_kwargs": {"enable_thinking": True}}
        if p["tools"]:
            payload["tools"] = PYTHON_TOOL
        try:
            resp = http_json(VLLM_API + "/chat/completions", payload, timeout=600)
            choice = resp["choices"][0]
            msg = choice["message"]
            calls = [[(c.get("function") or {}).get("name"), (c.get("function") or {}).get("arguments")]
                     for c in (msg.get("tool_calls") or [])]
            text = ("<REASON>" + (msg.get("reasoning_content") or msg.get("reasoning") or "")
                    + "<CONTENT>" + (msg.get("content") or "")
                    + "<TOOLS>" + json.dumps(calls, sort_keys=True))
            usage = resp.get("usage") or {}
            outs.append({"k": p["k"], "tools": p["tools"],
                         "sha16": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
                         "text_head": text[:200],
                         "completion_tokens": usage.get("completion_tokens"),
                         "finish_reason": choice.get("finish_reason"),
                         "n_tool_calls": len(calls)})
        except Exception as exc:
            outs.append({"k": p["k"], "tools": p["tools"], "error": repr(exc)[:200]})
    result = {"tag": tag, "server_tag": CURRENT_SERVER["tag"], "n": len(outs),
              "errors": sum(1 for o in outs if "error" in o),
              "elapsed_s": round(time.time() - t0, 1),
              "outputs": outs}
    RESULTS["phases"][name] = result
    save_results()
    print(f"flashnext-gate: {name}: {result['n'] - result['errors']}/{result['n']} ok in "
          f"{result['elapsed_s']} s; shas {[o.get('sha16') for o in outs][:6]}...", flush=True)
    return result


def image_smoke(tag):
    """One chat completion with a board image — is multimodal input wired?"""
    name = f"image_smoke_{tag}"
    rng = random.Random(777)
    payload = {"model": QWEN_SERVED_MODEL_NAME,
               "messages": [{"role": "user", "content": [
                   {"type": "text", "text": "Describe this game board in one sentence."},
                   {"type": "image_url", "image_url": {
                       "url": "data:image/png;base64," + board_png_b64(rng)}}]}],
               "temperature": 1.0, "top_p": 0.95, "top_k": 20, "max_tokens": 256,
               "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": True}}
    try:
        resp = http_json(VLLM_API + "/chat/completions", payload, timeout=600)
        msg = resp["choices"][0]["message"]
        usage = resp.get("usage") or {}
        result = {"ok": True, "prompt_tokens": usage.get("prompt_tokens"),
                  "completion_tokens": usage.get("completion_tokens"),
                  "content_head": (msg.get("content") or "")[:200]}
    except Exception as exc:
        result = {"ok": False, "error": repr(exc)[:400]}
        IMAGES_ENABLED["on"] = False
        print("flashnext-gate: image smoke FAILED — disabling images in the load generator",
              flush=True)
    RESULTS["phases"][name] = result
    RESULTS["verdicts"]["multimodal_wired"] = result["ok"]
    save_results()
    print(f"flashnext-gate: {name}: {result}", flush=True)
    return result


print(f"flashnext-gate: library ready, elapsed {elapsed_min():.1f} min")
save_results()
'''


CELL_BOOT = r'''# ===================== Phase 2 — BOOT (their argv verbatim; pre-registered retry rung) =====================
BOOT_TAG = None
try:
    if not RUNTIME_OK:
        raise RuntimeError("assemble failed — boot skipped")
    for _tag, _flags, _arch in BOOT_LADDER:
        try:
            ARCH_OVERRIDE["value"] = _arch
            print(f"flashnext-gate: rung [{_tag}] TORCH_CUDA_ARCH_LIST={_arch!r}", flush=True)
            start_server(_flags, tag=_tag)
            BOOT_TAG = _tag
            break
        except Exception as _exc:
            traceback.print_exc()
            RESULTS["boots"].setdefault(_tag, {})["error"] = repr(_exc)[:600]
            print(f"flashnext-gate: boot [{_tag}] FAILED — next rung of the ladder", flush=True)
            save_results()
            stop_server(f"cleanup after failed {_tag}")
    if BOOT_TAG is None:
        RESULTS["verdicts"]["boot"] = "BOOT FAILED on all rungs (device-detect arch, 12.0a, reduced)"
    else:
        _b = RESULTS["boots"][BOOT_TAG]
        _el = _b.get("engine_lines") or {}
        RESULTS["verdicts"]["boot"] = (
            f"BOOT OK on rung '{BOOT_TAG}'"
            + (" (DEGRADED-CONFIG: their exact config did not serve)" if BOOT_TAG == "reduced" else "")
            + f" in {_b.get('boot_s')} s — attention backend {_el.get('attention_backend_name')}, "
            + f"gpu after boot {_b.get('gpu_after')}")
except Exception as _exc:
    traceback.print_exc()
    RESULTS["verdicts"].setdefault("boot", "BOOT PHASE ERROR: " + repr(_exc)[:300])
print("flashnext-gate: BOOT VERDICT:", RESULTS["verdicts"].get("boot"), flush=True)
save_results()
'''


CELL_SMOKE = r'''# ===================== Phase 3 — SMOKE: models, parser, image, tool battery, greedy battery =====================
try:
    if BOOT_TAG is None:
        raise RuntimeError("no server — smoke skipped")
    RESULTS["phases"]["models_endpoint"] = {"served_ids": served_ids()}
    print("flashnext-gate: /v1/models ->", RESULTS["phases"]["models_endpoint"], flush=True)
    save_results()
    try:
        parser_roundtrip(BOOT_TAG)
    except Exception:
        traceback.print_exc()
    try:
        image_smoke(BOOT_TAG)
    except Exception:
        traceback.print_exc()
    try:
        run_tool_battery(BOOT_TAG)
    except Exception:
        traceback.print_exc()
    try:
        run_battery(BOOT_TAG)
    except Exception:
        traceback.print_exc()
except Exception as _exc:
    traceback.print_exc()
    RESULTS["verdicts"].setdefault("smoke", "SMOKE SKIPPED: " + repr(_exc)[:200])
print("flashnext-gate: smoke done at", round(elapsed_min(), 1), "min", flush=True)
save_results()
'''


CELL_LOADS = r'''# ===================== Phase 4 — duck-shaped load: conc-1 / conc-8 / conc-28 stable (+churn if time) =====================
if BOOT_TAG is not None:
    for _name, _conc, _warm, _meas, _regime in [
            ("probe_conc1", 1, 30, 120, "stable"),
            ("stable_conc8", 8, 60, 300, "stable"),
            ("stable_conc28", 28, 90, 480, "stable")]:
        try:
            run_load_phase(_name, _conc, _warm, _meas, regime=_regime)
        except Exception:
            traceback.print_exc()
            RESULTS["phases"].setdefault(_name, {"error": "see traceback", "regime": _regime})
            save_results()
    if elapsed_min() < CHURN_GATE_MIN:
        try:
            run_load_phase("churn_conc28", 28, 60, 360, regime="churn")
        except Exception:
            traceback.print_exc()
            RESULTS["phases"].setdefault("churn_conc28", {"error": "see traceback", "regime": "churn"})
            save_results()
    else:
        RESULTS["phases"]["churn_conc28"] = {"skipped": "time-gate"}
else:
    for _name in ("probe_conc1", "stable_conc8", "stable_conc28", "churn_conc28"):
        RESULTS["phases"][_name] = {"skipped": "no-server"}
print("flashnext-gate: loads done at", round(elapsed_min(), 1), "min", flush=True)
save_results()
'''


CELL_FINAL = r'''# ===================== Phase 5 — pre-registered GATE + comparison vs the 27B baseline =====================
def _cell(value, width=12):
    return ("-" if value is None else str(value)).rjust(width)


_boot_ok = BOOT_TAG is not None
_conc28 = RESULTS["phases"].get("stable_conc28") or {}
_conc8 = RESULTS["phases"].get("stable_conc8") or {}
_conc1 = RESULTS["phases"].get("probe_conc1") or {}
_churn = RESULTS["phases"].get("churn_conc28") or {}
_agg28 = agg_of(_conc28)
_parse = RESULTS["verdicts"].get("tool_parse_rate")

print("\n" + "=" * 100)
print("arc3-flashnext-gate — sonpham Flash-Next NVFP4 on ONE Kaggle RTX Pro 6000")
print("=" * 100)
print("GPU/driver :", RESULTS["meta"].get("gpu_name"), "| driver",
      RESULTS["meta"].get("driver_version"), "| python", RESULTS["meta"].get("python"))
print("runtime    :", (RESULTS["meta"].get("assemble") or {}).get("runtime_import"),
      "(pinned vLLM dev — NOT SGLang)")
print("model      :", RESULTS["meta"].get("served_model_name"), "@", "@@MODEL_REVISION@@")
print("BOOT       :", RESULTS["verdicts"].get("boot"))
_b = RESULTS["boots"].get(BOOT_TAG or "gcp_exact") or {}
for _k in ("model_load", "kv_cache_size", "max_concurrency", "ple_lower", "offload"):
    for _ln in ((_b.get("engine_lines") or {}).get(_k) or {}).get("lines", [])[:1]:
        print(f"   {_k}: {_ln[:900]}")
print("multimodal :", RESULTS["verdicts"].get("multimodal_wired"),
      "| parser round-trip:", (RESULTS["phases"].get(f"parser_roundtrip_{BOOT_TAG}") or {}).get("verdict"),
      "| tool parse rate:", _parse)

print("-" * 100)
print("LOAD TABLE — Flash-Next NVFP4 (this run) vs 27B-FP8 (lab3 08-26, same GPU pool)")
print(f"{'phase':<22}{'flash tok/s':>12}{'27B tok/s':>11}{'ratio':>7}"
      f"{'flash t/m/sess':>15}{'27B t/m/sess':>13}{'prefix hit':>11}{'prompt~':>9}{'reqs/err':>10}")
for _name, _phase, _bkey in (("probe_conc1", _conc1, "conc1"),
                             ("stable_conc8", _conc8, "conc8"),
                             ("stable_conc28", _conc28, "conc28"),
                             ("churn_conc28", _churn, "conc28")):
    if _phase.get("skipped") or _phase.get("error"):
        print(f"{_name:<22}{'(' + str(_phase.get('skipped') or _phase.get('error'))[:40] + ')':>12}")
        continue
    _fa = agg_of(_phase)
    _ba = BASELINE_27B[_bkey]["agg_tok_s"]
    _ratio = round(_fa / _ba, 2) if (_fa and _ba) else None
    print(f"{_name:<22}{_cell(_fa, 12)}{_cell(_ba, 11)}{_cell(_ratio, 7)}"
          f"{_cell(per_session_of(_phase), 15)}{_cell(BASELINE_27B[_bkey]['tok_min_session'], 13)}"
          f"{_cell(_phase.get('prefix_cache_hit_rate_window'), 11)}"
          f"{_cell(_phase.get('prompt_tokens_mean'), 9)}"
          f"{_cell(str(_phase.get('requests_in_window')) + '/' + str(_phase.get('errors')), 10)}")

print("-" * 100)
_speed_ok = _agg28 is not None and _agg28 >= GATE_CONC28_TOK_S
_parse_ok = _parse is not None and _parse >= GATE_TOOL_PARSE
_gate = _boot_ok and _speed_ok and _parse_ok
RESULTS["verdicts"]["gate_inputs"] = {
    "boot_ok": _boot_ok, "boot_rung": BOOT_TAG,
    "stable_conc28_agg_tok_s": _agg28, "gate_conc28_tok_s": GATE_CONC28_TOK_S,
    "speed_ok": _speed_ok,
    "tool_parse_rate": _parse, "gate_tool_parse": GATE_TOOL_PARSE, "parse_ok": _parse_ok,
    "spec_decode_note": "MTP disabled in their package (expected zero draft tokens): "
        + json.dumps((_conc28.get("acceptance") or {})),
}
RESULTS["verdicts"]["gate"] = (
    ("GATE PASS" if _gate else "GATE FAIL")
    + f" — boot={'OK/' + str(BOOT_TAG) if _boot_ok else 'FAILED'}, "
    + f"stable conc-28 {_agg28} tok/s (rule >= {GATE_CONC28_TOK_S}; 27B = "
    + f"{BASELINE_27B['conc28']['agg_tok_s']}), tool parse {_parse} (rule >= {GATE_TOOL_PARSE})")
print("GATE RULE  : boots on 1 GPU AND stable conc-28 >= "
      f"{GATE_CONC28_TOK_S} tok/s agg AND tool parse >= {GATE_TOOL_PARSE}")
print("GATE       :", RESULTS["verdicts"]["gate"])
if BOOT_TAG == "reduced":
    print("NOTE       : boot succeeded only on the DEGRADED rung (util 0.92 / 16k / 12 seqs) — "
          "their exact GCP config did not serve on this GPU.")
print("-" * 100)
RESULTS["meta"]["finished_utc"] = datetime.utcnow().isoformat() + "Z"
RESULTS["meta"]["total_elapsed_min"] = round(elapsed_min(), 1)
host_snapshot("end")
save_results()
print(f"flashnext-gate: DONE in {elapsed_min():.1f} min — results JSON: {RESULTS_PATH} "
      f"({RESULTS_PATH.stat().st_size} bytes)")
print("=" * 100, flush=True)
try:
    stop_server("kernel end")
except Exception:
    pass
'''


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
def _code_cell(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


def _md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": source.splitlines(keepends=True)}


def _replace_between(text: str, start_marker: str, end_marker: str, new: str) -> str:
    i = text.index(start_marker)
    j = text.index(end_marker, i)
    return text[:i] + new + text[j:]


def _lab_library() -> str:
    """arc3-serving-lab's library with the serve-chain blocks REPLACED and the
    duck-shaped generator + measurement core kept verbatim (lab4 recipe)."""
    lab1 = runpy.run_path(str(LAB1_BUILD))
    lib = lab1["CELL_LAB_LIB"]

    lib = _replace_between(lib, "# ============================ SERVING LAB LIBRARY",
                           "import base64\n", LIB_HEADER)
    lib = lib.replace("import base64\n",
                      "import base64\nimport hashlib\nimport signal\nimport zlib\n", 1)
    constants = (LIB_CONSTANTS
                 .replace("@@BASELINE_27B@@", json.dumps(BASELINE_27B, indent=4))
                 .replace("@@GATE_CONC28_TOK_S@@", repr(GATE_CONC28_TOK_S))
                 .replace("@@GATE_TOOL_PARSE@@", repr(GATE_TOOL_PARSE)))
    lib = _replace_between(lib, 'VLLM_HOST = "127.0.0.1"', "def elapsed_min():", constants)
    # elapsed_min/save_results are defined identically in the imports cell — drop the dupes
    lib = _replace_between(lib, "def elapsed_min():", "def http_json(", "")
    lib = _replace_between(lib, "# ---- server lifecycle", "# ---- duck-shaped request synthesis",
                           LIB_SERVER)
    lib = _replace_between(lib, "class Session:", "def chat_request(session", LIB_SESSION)
    lib = _replace_between(lib, "def run_load_phase(", "# ---- parser round-trip", LIB_LOAD_PHASE)
    lib = _replace_between(lib, "# ---- long-context soak", 'print(f"serving-lab: library ready',
                           LIB_BATTERY)
    lib = lib.replace('print(f"serving-lab: library ready, elapsed {elapsed_min():.1f} min")\n'
                      "save_results()\n", "", 1)
    # their gameplay sampling is temperature 1.0 (LOCAL_ANALYZER_TEMPERATURE) —
    # applies to chat_request and parser_roundtrip; the greedy battery stays 0.0
    lib = lib.replace('"temperature": 0.6,', '"temperature": 1.0,')
    lib = lib.replace('"had_reasoning": bool(message.get("reasoning_content"))',
                      '"had_reasoning": bool(message.get("reasoning_content") or message.get("reasoning"))', 1)
    lib = lib.replace('WANTED_PREFIXES = ("vllm:spec_decode", "vllm:generation_tokens",\n'
                      '                   "vllm:prompt_tokens", "vllm:prefix_cache",\n'
                      '                   "vllm:num_preemptions", "vllm:num_requests")',
                      'WANTED_PREFIXES = ("vllm:spec_decode", "vllm:generation_tokens",\n'
                      '                   "vllm:prompt_tokens", "vllm:prefix_cache",\n'
                      '                   "vllm:num_preemptions", "vllm:num_requests",\n'
                      '                   "vllm:kv_cache_usage", "vllm:gpu_cache_usage")', 1)
    lib = lib.replace("# ---- Prometheus scrape (metric names verified at vLLM v0.19.0 tag) ----",
                      "# ---- Prometheus scrape (same counter names on their pinned vLLM dev) ----", 1)
    lib = lib.replace("serving-lab:", "flashnext-gate:")
    # invariants: everything lab1-specific is gone, gate pieces are in
    for gone in ("MTP3_FLAGS", "MTP2_FLAGS", "MODAL_H100", "long_context_soak", "Q1_RULE",
                 "NST2_GATE_MIN", "serving_lab_results.json", '"kernel": "arc3-serving-lab"',
                 "BASE_SERVE_FLAGS", '"temperature": 0.6'):
        assert gone not in lib, f"lab1 leftover in library: {gone}"
    for keep in ("def make_transcript", "def board_png_b64", "def chat_request",
                 "def parser_roundtrip", "def scrape_metrics", "def acceptance_delta",
                 "def drain_inflight", "PYTHON_TOOL", "def run_load_phase", "def run_battery",
                 "def run_tool_battery", "def image_smoke", "def start_server", "def stop_server",
                 "def reset_prefix_cache", "def capture_engine_lines", "def agg_of",
                 "def per_session_of", 'regime="stable"', '"churn"',
                 "prefix_cache_hit_rate_window", '"qwen3_xml"', "BOOT_LADDER",
                 "IMAGES_ENABLED", "BASELINE_27B", "GATE_CONC28_TOK_S"):
        assert keep in lib, f"missing in flashnext-gate library: {keep}"
    return lib


def build_notebook() -> dict:
    assemble_cell = (CELL_ASSEMBLE
                     .replace("@@RUNTIME_SHA@@", EXPECTED_RUNTIME_SHA256)
                     .replace("@@ZSTD_SHA@@", EXPECTED_ZSTD_SHA256)
                     .replace("@@VERSIONS_LINE@@", EXPECTED_VERSIONS_LINE)
                     .replace("@@RUNTIME_ARCHIVE@@", RUNTIME_ARCHIVE)
                     .replace("@@MODEL_ID@@", MODEL_ID)
                     .replace("@@MODEL_REVISION@@", MODEL_REVISION)
                     .replace("@@EXPECTED_SAFETENSOR_FILES@@", str(EXPECTED_SAFETENSOR_FILES))
                     .replace("@@EXPECTED_PAYLOAD_MIN_BYTES@@", str(EXPECTED_PAYLOAD_MIN_BYTES)))
    final_cell = CELL_FINAL.replace("@@MODEL_REVISION@@", MODEL_REVISION)
    assert "@@" not in assemble_cell and "@@" not in final_cell
    cells = [
        _md_cell(MD_HEADER),
        _code_cell(CELL_IMPORTS),        # phase 0: driver first + fail-fast GPU assert
        _code_cell(assemble_cell),       # phase 1: runtime + model view
        _code_cell(_lab_library()),      # generator + measurement (lab1/lab4) adapted
        _code_cell(CELL_BOOT),           # phase 2
        _code_cell(CELL_SMOKE),          # phase 3
        _code_cell(CELL_LOADS),          # phase 4
        _code_cell(final_cell),          # phase 5
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def main() -> None:
    notebook = build_notebook()
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        try:
            ast.parse("".join(cell["source"]))
        except SyntaxError as exc:
            raise SystemExit(f"cell {i} failed ast.parse: {exc}") from exc

    joined = "\n".join("".join(c["source"]) for c in notebook["cells"])
    # Ordering invariants
    assert (joined.index("DRIVER + HOST + MOUNTS FIRST") < joined.index("WRONG-GPU")
            < joined.index("Phase 1 — ASSEMBLE"))
    assert (joined.index("Phase 1 — ASSEMBLE") < joined.index("SERVING LAB LIBRARY (flashnext-gate)")
            < joined.index("Phase 2 — BOOT") < joined.index("Phase 3 — SMOKE")
            < joined.index("Phase 4 —") < joined.index("Phase 5 —"))
    # Their serve-chain invariants (kaggle_flashnext_setup.py, copied verbatim)
    for needed in ('"VLLM_PLE_CPU_OFFLOAD": "1"', '"VLLM_PLE_OFFLOAD_READY_TIMEOUT": "1800"',
                   '"VLLM_ENABLE_CUDA_COMPATIBILITY": "0"',
                   '"PYTORCH_ALLOC_CONF": "expandable_segments:False"',
                   '"--tensor-parallel-size", "1"', '"--gpu-memory-utilization", "0.96"',
                   '"--max-num-seqs", "22"', '"--max-num-batched-tokens", "6144"',
                   '"--max-model-len", "32768"', '"--kv-cache-dtype", "auto"',
                   '"--distributed-executor-backend", "mp"', '"--no-enable-flashinfer-autotune"',
                   '"--tool-call-parser", "qwen3_xml"', '"--enable-prefix-caching"',
                   EXPECTED_RUNTIME_SHA256, EXPECTED_ZSTD_SHA256, EXPECTED_VERSIONS_LINE,
                   "flashnext_gate_results.json", "start_new_session=True",
                   "vllm:prefix_cache_hits", "vllm:spec_decode_num_accepted_tokens",
                   "serving-part-000", "source-bundle/zstd", "FLASHNEXT_GCP_MODEL_INFO.json",
                   "model-plefp8-", "ple-bf16-conversion.json", "WRONG-GPU"):
        assert needed in joined, f"missing from kernel: {needed}"
    # No games, no submission machinery, no online installs.
    for banned in ("submission.parquet", "GameAgent", "scorecard", "arc_agi_3_wheels",
                   "pip install", "sglang", "@@"):
        assert banned not in joined, f"banned token in gate kernel: {banned}"

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
            "sonphamorg/arc3-flashnext-serving-part-a-v1",
            "sonphamorg/arc3-flashnext-serving-part-b-v1",
            "sonphamorg/arc3-flashnext-serving-part-c-v1",
            "sonphamorg/arc3-flashnext-gcp-runtime-exact-v1",
        ],
        "kernel_sources": [],
        # THE RTX Pro 6000 GATE (08-22 push lesson): the competition source is
        # what admits the kernel to the scored GPU pool.
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        # their preconverted Kaggle MODEL is private (403) — datasets are the path
        "model_sources": [],
    }, indent=2) + "\n")

    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    print("built", KERNEL_SLUG, "code-cell sha256", hashlib.sha256(code.encode()).hexdigest())
    print("cells:", len(notebook["cells"]), "| notebook:", nb_path, "|", nb_path.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
