#!/usr/bin/env python3
"""build_serving_lab4.py — vLLM 0.24 boot + prefix-regime + MTP probe (NO games).

One private RTX Pro 6000 commit, <= 2.5 h, no games, no submission. Three
questions, each phase-wrapped so a failed boot never kills the kernel and
partial results always land in /kaggle/working/serving_lab4_results.json:

Q1 BOOT — does vLLM 0.24.0 boot on the Kaggle image with the pack3 recipe
   (docs/research-2026-08-29/R5-pack3-vllm024-boot-recipe.md §6.2): the
   jcole75/arc3-qwen36-runtime-wheels v3 wheelhouse (197 wheels, offline pip
   --target), CUDA-13.3 nvcc under site-packages/nvidia/cu13,
   VLLM_USE_FLASHINFER_SAMPLER=0, VLLM_USE_DEEP_GEMM=0, --mamba-cache-mode
   align, prefix caching ON — serving the foysalemonshanto repack (Kaggle MODEL
   mount, includes mtp.safetensors; the index must reference it).
   Printed: nvidia-smi driver FIRST, the engine-config log line
   (speculative_config=..., attention backend), any flashinfer.jit frames.

Q2 PREFIX REGIME — conc-28 duck-shaped load (15-25k-token prompts with a
   base64 board PNG, thinking on, 1-3k gen tokens) in TWO regimes:
     churn  — every request drops its oldest block: the prefix past the
              system prompt changes on every call (today's shipped duck)
     stable — a long fixed prefix, append-only (the Pack-1 hysteresis regime)
   Reported per regime: gen tok/s aggregate, gen tok/min per session, and
   the /metrics prefix-cache hit rate over the measured window.

Q3 MTP — restart with --speculative-config '{"method":"mtp",
   "num_speculative_tokens":2}' WITH prefix caching; tool-call round-trip
   (python tool, qwen3_coder parser); a 20-prompt GREEDY correctness battery
   compared byte-for-byte against the non-MTP server (a base-vs-base repeat
   gives the noise floor); vllm:spec_decode_* acceptance; the conc-28 stable
   load again. Boot ladder: MTP+prefix -> MTP no-prefix -> base recovery.

Load generator + measurement + metrics code are REUSED from arc3-serving-lab
(the pattern lab3 re-ran on this exact GPU on 08-26), adapted by block
replacement — see _lab_library(). Everything that touches the serve chain is
new (0.24 install, CUDA-13 env, 0.24 argv, MTP ladder).

vLLM 0.24.0 names verified against the v0.24.0 tag on 2026-08-29:
  flags  : vllm/engine/arg_utils.py — mamba_cache_mode, mm_encoder_tp_mode,
           speculative_config, limit_mm_per_prompt, enable_prefix_caching
           (BooleanOptionalAction => --no-enable-prefix-caching), max_num_seqs,
           gpu_memory_utilization; vllm/config/cache.py — mamba_cache_mode in
           {"none","all","align"}.
  metrics: vllm/v1/metrics/loggers.py — vllm:prefix_cache_queries,
           vllm:prefix_cache_hits, vllm:generation_tokens, vllm:prompt_tokens,
           vllm:num_requests_running/_waiting, vllm:num_preemptions;
           vllm/v1/spec_decode/metrics.py — vllm:spec_decode_num_drafts,
           _num_draft_tokens, _num_accepted_tokens, _num_accepted_tokens_per_pos;
           log line "Avg Draft acceptance rate: %.1f%%".

Usage:
  .venv/bin/python submission/_serving_lab4/build_serving_lab4.py
  .venv/bin/python submission/_serving_lab4/validate_serving_lab4.py
  # DO NOT push without Ahmed's go: this build is local-only.
"""
import ast
import hashlib
import json
import runpy
import textwrap
from pathlib import Path

HERE = Path(__file__).parent
LAB1_BUILD = HERE.parent / "_serving_lab" / "build_serving_lab.py"
DUCK38_BUILD = HERE.parent / "_duck38_v12" / "build_duck38_v12.py"
KERNEL_SLUG = "arc3-serving-lab4"

# Wheelhouse identity (R5 §1.1 / §1.2; WHEELHOUSE_MANIFEST.json read locally
# from the Kaggle CLI copy on 2026-08-29). Cheap checks instead of hashing
# 197 wheels: count + total bytes + a few named wheels.
WH_WHEEL_COUNT = 197
WH_TOTAL_BYTES = 6486476943
WH_KEY_WHEELS = [
    "vllm-0.24.0-cp38-abi3-manylinux_2_28_x86_64.whl",
    "torch-2.11.0+cu129-cp312-cp312-manylinux_2_28_x86_64.whl",
    "flashinfer_python-0.6.12-py3-none-any.whl",
    "flashinfer_cubin-0.6.12-py3-none-any.whl",
    "nvidia_cuda_nvcc-13.3.73-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
    "nvidia_cuda_runtime-13.3.29-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
    "nvidia_cuda_crt-13.3.73-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
    "xgrammar-0.2.3-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
]
# jacquesbuis pins the manifest sha (R5 §1.1 shows only prefix/suffix) — a
# WARNING-level identity check, never fatal.
WH_MANIFEST_SHA_PREFIX = "b765fe47"
WH_MANIFEST_SHA_SUFFIX = "22d2"


MD_HEADER = """\
# arc3-serving-lab4 — vLLM 0.24 boot · prefix regimes · MTP (NO games, NO submission)

Private RTX Pro 6000 commit, budget <= 2.5 h. Three questions, every phase
failure-wrapped; results rewritten after every phase to
`/kaggle/working/serving_lab4_results.json`.

| Phase | What | Budget |
|---|---|---|
| 0 | nvidia-smi **driver first**, GPU assert (dies in minutes on a P100 rehome) | 1 min |
| inputs | wheelhouse identity (manifest wheel_count == 197, key wheels), model mount, **index references mtp.safetensors**, disk preflight | 1 min |
| install | offline pip --target from `jcole75/arc3-qwen36-runtime-wheels` v3 (their flags verbatim); version assert; CUDA-13.3 nvcc gate | 5-8 min |
| **Q1** | boot 0.24.0: pack3 argv + `--mamba-cache-mode align` + prefix caching ON, `VLLM_USE_FLASHINFER_SAMPLER=0`, `VLLM_USE_DEEP_GEMM=0`; print engine-config line (speculative_config, attention backend), flashinfer.jit frames; weight attestation | ~15-20 min |
| **Q2** | parser round-trip; greedy battery x2 (noise floor); conc-1 probe; conc-28 **churn** (90 s + 8 min); conc-28 **stable** (90 s + 8 min) — gen tok/s, tok/min/session, prefix hit rate | ~32 min |
| **Q3** | restart MTP nst=2 **with** prefix caching (ladder: -> no-prefix -> base recovery); spec-decode probe; parser; greedy battery vs base (mismatch count); conc-1 probe; conc-28 stable; churn if time | ~40 min |
| final | decision table + pre-registered verdicts | 1 min |

**Pre-registered decision rules**

* **MTP worth adopting** only if `greedy mismatches == 0` AND
  `stable conc-28 gen tok/s (MTP) >= 1.3x (no MTP)` AND `acceptance >= 0.5`.
* **Prefix-stable regime confirmed** if its window prefix hit rate `>= 0.5`
  AND `gen tok/s >= 1.5x churn`.
* Q1 is binary: `/v1/models` serves `Qwen/Qwen3.8-27B-FP8` from the 0.24 engine
  with `speculative_config=None` on the first boot.

Reference for scale (lab3, 08-26, vLLM 0.19, same GPU): conc-28 636 gen
tok/min/session, 297 tok/s aggregate, prompt ~21.5k; window prefix hit rate
was NOT measured then (cumulative counters implied ~0.08 at conc 28).
"""


CELL_IMPORTS = r'''# ================= serving-lab4 — imports, results sink, DRIVER + GPU FIRST =================
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import time
import traceback
from datetime import datetime
from pathlib import Path

NOTEBOOK_START_EPOCH = time.time()
WORKING_DIR = Path("/kaggle/working")
WORKING_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = WORKING_DIR / "serving_lab4_results.json"
RESULTS = {
    "meta": {"kernel": "arc3-serving-lab4",
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


def _smi_one(field):
    r = subprocess.run(["nvidia-smi", "--query-gpu=" + field, "--format=csv,noheader"],
                       capture_output=True, text=True)
    out = (r.stdout or "").strip()
    return out.splitlines()[0].strip() if (r.returncode == 0 and out) else None


def _smi_query(fields):
    # ONE bad field fails the whole batch query, so fall back field-by-field
    # (driver renamed clocks_throttle_reasons.* -> clocks_event_reasons.*).
    names = [f.strip() for f in fields.split(",") if f.strip()]
    r = subprocess.run(["nvidia-smi", "--query-gpu=" + ",".join(names),
                        "--format=csv,noheader"], capture_output=True, text=True)
    out = (r.stdout or "").strip()
    if r.returncode == 0 and out:
        vals = [v.strip() for v in out.splitlines()[0].split(",")]
        if len(vals) == len(names):
            return dict(zip(names, vals))
    result = {"_batch_query_failed": (r.stderr or "")[:200]}
    for name in names:
        alt = name.replace("clocks_throttle_reasons", "clocks_event_reasons")
        value = _smi_one(name)
        if value is None and alt != name:
            value = _smi_one(alt)
            if value is not None:
                name = alt
        result[name] = value
    return result


# ---- DRIVER VERSION FIRST (R5 §6.1 risk 2: CUDA-13-linked JIT kernels need an
# R580-class driver; torch+cu129 needs R575). Printed before anything else.
_first = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
     "--format=csv,noheader"], capture_output=True, text=True)
GPU_LINE = (_first.stdout or "").strip()
print("serving-lab4: nvidia-smi name,driver,memory =", repr(GPU_LINE), "rc=", _first.returncode, flush=True)
_smi_full = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
print("\n".join((_smi_full.stdout or "").splitlines()[:12]), flush=True)
_parts = [p.strip() for p in GPU_LINE.split(",")] if GPU_LINE else []
GPU_NAME = _parts[0] if _parts else ""
DRIVER_VERSION = _parts[1] if len(_parts) > 1 else ""
try:
    DRIVER_MAJOR = int(DRIVER_VERSION.split(".")[0])
except Exception:
    DRIVER_MAJOR = None
RESULTS["meta"]["gpu_name"] = GPU_NAME
RESULTS["meta"]["driver_version"] = DRIVER_VERSION
RESULTS["meta"]["driver_major"] = DRIVER_MAJOR
if DRIVER_MAJOR is not None and DRIVER_MAJOR < 580:
    print(f"serving-lab4: WARNING driver {DRIVER_VERSION} < 580 — any FlashInfer kernel "
          "JIT-linked against libcudart.so.13 may fail at load; with the sampler off "
          "and FLASH_ATTN auto-selected nothing should need the JIT", flush=True)
    RESULTS["meta"]["driver_warning"] = "driver < 580: CUDA-13-linked JIT kernels at risk"
RESULTS["meta"]["gpu_static"] = _smi_query(
    "name,driver_version,vbios_version,memory.total,clocks.max.sm,clocks.max.mem,"
    "power.max_limit,power.limit,pcie.link.gen.max,pcie.link.width.max,compute_mode")
RESULTS["meta"]["gpu_now"] = _smi_query(
    "clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu,memory.used,"
    "clocks_throttle_reasons.active,clocks_throttle_reasons.sw_power_cap,"
    "clocks_throttle_reasons.hw_slowdown,clocks_throttle_reasons.sw_thermal_slowdown")
try:
    _models = [ln.split(":", 1)[1].strip() for ln in Path("/proc/cpuinfo").read_text(errors="replace").splitlines()
               if ln.lower().startswith("model name")]
    RESULTS["meta"]["cpu_model"] = _models[0] if _models else None
    RESULTS["meta"]["cpu_logical_count"] = len(_models) or os.cpu_count()
    _mem = dict((ln.split(":", 1)[0], ln.split(":", 1)[1].strip())
                for ln in Path("/proc/meminfo").read_text().splitlines() if ":" in ln)
    RESULTS["meta"]["mem_total"] = _mem.get("MemTotal")
except Exception as _exc:
    RESULTS["meta"]["host_probe_error"] = repr(_exc)[:200]
RESULTS["meta"]["python"] = sys.version.split()[0]
RESULTS["meta"]["image_env"] = {k: os.environ.get(k) for k in
                                ("TORCH_CUDA_ARCH_LIST", "VLLM_ATTENTION_BACKEND", "CUDA_HOME",
                                 "LD_LIBRARY_PATH", "KAGGLE_KERNEL_RUN_TYPE")}
save_results()

# ---- FAIL-FAST GPU ASSERT: this lab only means anything on the RTX Pro 6000
# pool. Die IMMEDIATELY on a P100/T4 rehome so the slot costs minutes.
if _first.returncode != 0 or not GPU_NAME:
    raise RuntimeError("WRONG-GPU: nvidia-smi failed — no usable GPU. Aborting fast.")
if "6000" not in GPU_NAME.upper() or "RTX" not in GPU_NAME.upper():
    raise RuntimeError(f"WRONG-GPU: expected an RTX Pro 6000, got {GPU_NAME!r}. Aborting fast.")
print(f"serving-lab4: GPU assert PASS ({GPU_NAME}, driver {DRIVER_VERSION}), "
      f"python {RESULTS['meta']['python']}", flush=True)
'''


CELL_INPUTS = r'''# ================= INPUTS — wheelhouse identity (cheap), model mount, MTP index =================
# Wheelhouse identity numbers are substituted by the builder (WH_* constants).
WH_CANDIDATES = [Path("/kaggle/input/arc3-qwen36-runtime-wheels"),
                 Path("/kaggle/input/datasets/jcole75/arc3-qwen36-runtime-wheels")]
WHEELHOUSE = next((p for p in WH_CANDIDATES if (p / "WHEELHOUSE_MANIFEST.json").is_file()), None)
if WHEELHOUSE is None:
    _tree = subprocess.run(["find", "/kaggle/input", "-maxdepth", "3", "-mindepth", "1"],
                           capture_output=True, text=True)
    print("serving-lab4: /kaggle/input tree:\n" + (_tree.stdout or "")[:4000], flush=True)
    raise FileNotFoundError("wheelhouse jcole75/arc3-qwen36-runtime-wheels not mounted at "
                            + " or ".join(str(p) for p in WH_CANDIDATES))
WH_WHEELS = WHEELHOUSE / "wheels"
WH_REQUIREMENTS = WHEELHOUSE / "requirements-runtime.txt"
_manifest_raw = (WHEELHOUSE / "WHEELHOUSE_MANIFEST.json").read_bytes()
_manifest = json.loads(_manifest_raw)
_manifest_sha = __import__("hashlib").sha256(_manifest_raw).hexdigest()
_wh_info = {
    "path": str(WHEELHOUSE),
    "wheel_count": _manifest.get("wheel_count"),
    "total_bytes": _manifest.get("total_bytes"),
    "cuda_wheel_target": _manifest.get("cuda_wheel_target"),
    "cuda_jit_toolchain": _manifest.get("cuda_jit_toolchain"),
    "created_utc": _manifest.get("created_utc"),
    "files_listed": len(_manifest.get("files") or []),
    "manifest_sha256": _manifest_sha,
    "requirements_head": WH_REQUIREMENTS.read_text(errors="replace").splitlines()[:6]
    if WH_REQUIREMENTS.is_file() else None,
}
print("serving-lab4: wheelhouse", json.dumps(_wh_info, indent=1, default=str), flush=True)
assert _wh_info["wheel_count"] == @@WH_WHEEL_COUNT@@, f"wheelhouse wheel_count {_wh_info['wheel_count']} != @@WH_WHEEL_COUNT@@"
assert _wh_info["total_bytes"] == @@WH_TOTAL_BYTES@@, f"wheelhouse total_bytes {_wh_info['total_bytes']} != @@WH_TOTAL_BYTES@@"
assert _wh_info["cuda_wheel_target"] == "cu129", _wh_info["cuda_wheel_target"]
assert WH_WHEELS.is_dir(), f"missing {WH_WHEELS}"
assert WH_REQUIREMENTS.is_file(), f"missing {WH_REQUIREMENTS}"
assert "vllm==0.24.0" in WH_REQUIREMENTS.read_text(), "requirements-runtime.txt does not pin vllm==0.24.0"
_missing_wheels = [w for w in @@WH_KEY_WHEELS@@ if not (WH_WHEELS / w).is_file()]
_wh_info["key_wheels_missing"] = _missing_wheels
assert not _missing_wheels, f"key wheels missing from wheelhouse: {_missing_wheels}"
if not (_manifest_sha.startswith("@@WH_SHA_PREFIX@@") and _manifest_sha.endswith("@@WH_SHA_SUFFIX@@")):
    print(f"serving-lab4: WARNING manifest sha256 {_manifest_sha} does not match the "
          "jacquesbuis-pinned prefix/suffix (R5 §1.1) — dataset version may differ from v3",
          flush=True)
    _wh_info["manifest_sha_warning"] = True
RESULTS["meta"]["wheelhouse"] = _wh_info

# ---- model: Kaggle MODEL mount (not a dataset — R5 §6.1 risk 9: hard-code) ----
QWEN_SERVED_MODEL_NAME = "Qwen/Qwen3.8-27B-FP8"
QWEN_MODEL_PATH = Path("/kaggle/input/models/foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1")
if not QWEN_MODEL_PATH.is_dir():
    _alts = sorted(Path("/kaggle/input/models/foysalemonshanto/qwen3-8-27b-fp8-repacked-v1").rglob("config.json")) \
        if Path("/kaggle/input/models/foysalemonshanto/qwen3-8-27b-fp8-repacked-v1").exists() else []
    if _alts:
        QWEN_MODEL_PATH = _alts[0].parent
        print("serving-lab4: model mount resolved via rglob ->", QWEN_MODEL_PATH, flush=True)
    else:
        raise FileNotFoundError(f"Qwen3.8 Kaggle Model not attached at {QWEN_MODEL_PATH}")
_required_qwen_files = ["config.json", "model.safetensors.index.json", "tokenizer.json",
                        "tokenizer_config.json", "outside.safetensors", "mtp.safetensors",
                        "chat_template.jinja"]
_missing_qwen = [n for n in _required_qwen_files if not (QWEN_MODEL_PATH / n).is_file()]
assert not _missing_qwen, f"Qwen3.8 mount incomplete; missing {_missing_qwen}"
_layer_shards = sorted(QWEN_MODEL_PATH.glob("model-layers-*.safetensors"))
_all_shards = sorted(QWEN_MODEL_PATH.glob("*.safetensors"))
print(f"serving-lab4: model {QWEN_MODEL_PATH}: {len(_layer_shards)} layer shards, "
      f"{len(_all_shards)} safetensors", flush=True)

# ---- THE MTP INDEX CHECK (R5 §6.1 risk 9): if the repack's index does not map
# the mtp.* tensors to mtp.safetensors, vLLM loads fine and silently has NO
# draft head — Q3 would measure nothing. Print it.
_index = json.loads((QWEN_MODEL_PATH / "model.safetensors.index.json").read_text())
_wmap = _index.get("weight_map") or {}
_mtp_file_entries = {k: v for k, v in _wmap.items() if v == "mtp.safetensors"}
_mtp_named = {k: v for k, v in _wmap.items() if "mtp" in k.lower()}
_mtp_files = sorted(set(_mtp_named.values()))
_cfg = json.loads((QWEN_MODEL_PATH / "config.json").read_text())
_text_cfg = _cfg.get("text_config") or _cfg
MTP_INDEX_INFO = {
    "index_files_referenced": len(set(_wmap.values())),
    "tensors_in_mtp_safetensors": len(_mtp_file_entries),
    "tensors_named_mtp": len(_mtp_named),
    "files_holding_mtp_named_tensors": _mtp_files,
    "sample_mtp_keys": sorted(_mtp_file_entries)[:8],
    "mtp_safetensors_bytes": (QWEN_MODEL_PATH / "mtp.safetensors").stat().st_size,
    "config_mtp_num_hidden_layers": _text_cfg.get("mtp_num_hidden_layers"),
    "config_mtp_use_dedicated_embeddings": _text_cfg.get("mtp_use_dedicated_embeddings"),
    "architectures": _cfg.get("architectures"),
}
print("serving-lab4: MTP index check", json.dumps(MTP_INDEX_INFO, indent=1), flush=True)
RESULTS["meta"]["mtp_index"] = MTP_INDEX_INFO
if not _mtp_file_entries:
    print("serving-lab4: WARNING — index does NOT reference mtp.safetensors; MTP will "
          "have no draft head (Q3 becomes a no-op measurement)", flush=True)
    RESULTS["verdicts"]["mtp_index_warning"] = "index does not reference mtp.safetensors"

# ---- disk preflight (their numbers) + cache root ----
NEED_FREE = 13_250_838_162 + 2 * 1024 ** 3
_free = shutil.disk_usage(str(WORKING_DIR)).free
print(f"serving-lab4: /kaggle/working free {_free / 1e9:.1f} GB (need {NEED_FREE / 1e9:.1f} GB)", flush=True)
RESULTS["meta"]["working_free_bytes"] = _free
if _free < NEED_FREE:
    raise RuntimeError(f"insufficient /kaggle/working capacity: {_free} < {NEED_FREE}")
CACHE_ROOT = Path("/kaggle/temp/duck-vllm-cache")
try:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
except Exception:
    CACHE_ROOT = Path("/tmp/duck-vllm-cache")
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
RESULTS["meta"]["cache_root"] = str(CACHE_ROOT)
RESULTS["meta"]["model_path"] = str(QWEN_MODEL_PATH)
RESULTS["meta"]["served_model_name"] = QWEN_SERVED_MODEL_NAME
save_results()
'''


CELL_INSTALL = r'''# ================= INSTALL — vLLM 0.24.0 from the jcole75 wheelhouse (their pip flags verbatim) =================
SITE_PACKAGES = WORKING_DIR / "vllm-site-packages"
CUDA_HOME_PATH = SITE_PACKAGES / "nvidia" / "cu13"
CUDA_BIN_PATH = CUDA_HOME_PATH / "bin"
CUDA_LIB_PATH = CUDA_HOME_PATH / "lib"
SITE_BIN_PATH = SITE_PACKAGES / "bin"
INSTALL_STAMP = SITE_PACKAGES / ".arc3-qwen36-runtime-wheels"
STAMP_TEXT = ("dataset=jcole75/arc3-qwen36-runtime-wheels@3 vllm==0.24.0 torch==2.11.0+cu129 "
              "transformers==5.13.0 flashinfer==0.6.12 model=Qwen/Qwen3.8-27B-FP8\n")
EXPECTED_VERSIONS = {"vllm": "0.24.0", "torch": "2.11.0+cu129",
                     "transformers": "5.13.0", "flashinfer-python": "0.6.12"}
RUNTIME_OK = False
INSTALL_INFO = {}


def vllm_env():
    """The pack3 launch environment (R5 §2.2), verbatim semantics: CUDA-13.3
    toolchain from the wheelhouse for any FlashInfer JIT, FlashInfer sampler +
    DeepGEMM OFF, caches on /kaggle/temp. The image's pre-sm75
    TORCH_CUDA_ARCH_LIST (our lab2 v1 death) and any VLLM_ATTENTION_BACKEND are
    NOT inherited."""
    env = os.environ.copy()
    popped = {k: env.pop(k) for k in ("VLLM_ATTENTION_BACKEND", "TORCH_CUDA_ARCH_LIST") if k in env}
    INSTALL_INFO["env_popped"] = popped
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{SITE_PACKAGES}{os.pathsep}{existing}" if existing else str(SITE_PACKAGES)
    env.update({"CUDA_HOME": str(CUDA_HOME_PATH), "CUDA_PATH": str(CUDA_HOME_PATH),
                "FLASHINFER_NVCC": str(CUDA_BIN_PATH / "nvcc"),
                "FLASHINFER_EXTRA_LDFLAGS": f"-L{CUDA_LIB_PATH} -L/usr/local/nvidia/lib64"})
    for key, vals in {"PATH": (CUDA_BIN_PATH, SITE_BIN_PATH),
                      "LD_LIBRARY_PATH": (CUDA_LIB_PATH, Path("/usr/local/nvidia/lib64")),
                      "LIBRARY_PATH": (CUDA_LIB_PATH, Path("/usr/local/nvidia/lib64")),
                      "CPATH": (CUDA_HOME_PATH / "include",)}.items():
        env[key] = os.pathsep.join([*(str(v) for v in vals), *([env[key]] if env.get(key) else [])])
    env.update({"USE_TF": "0", "TRANSFORMERS_NO_TF": "1", "TRANSFORMERS_NO_TORCHVISION": "1",
                "PYTHONDONTWRITEBYTECODE": "1", "VLLM_NO_USAGE_STATS": "1",
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_USE_DEEP_GEMM": "0",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "XDG_CACHE_HOME": str(CACHE_ROOT),
                "TORCH_EXTENSIONS_DIR": str(CACHE_ROOT / "torch-extensions"),
                "TRITON_CACHE_DIR": str(CACHE_ROOT / "triton"),
                "FLASHINFER_WORKSPACE_BASE": str(CACHE_ROOT / "flashinfer")})
    return env


try:
    _t0 = time.time()
    if INSTALL_STAMP.is_file() and INSTALL_STAMP.read_text(encoding="utf-8") == STAMP_TEXT:
        print("serving-lab4: cached 0.24 install present at", SITE_PACKAGES, flush=True)
        INSTALL_INFO["cached"] = True
    else:
        shutil.rmtree(SITE_PACKAGES, ignore_errors=True)
        SITE_PACKAGES.mkdir(parents=True, exist_ok=True)
        _pip_cmd = [sys.executable, "-m", "pip", "install", "--no-index",
                    "--find-links", str(WH_WHEELS), "--requirement", str(WH_REQUIREMENTS),
                    "--target", str(SITE_PACKAGES), "--upgrade", "--ignore-installed",
                    "--only-binary", ":all:", "--no-compile", "--no-cache-dir",
                    "--disable-pip-version-check", "--no-warn-conflicts"]
        _pip_log = WORKING_DIR / "pip-install-0240.log"
        print("serving-lab4: pip:", " ".join(_pip_cmd), flush=True)
        with _pip_log.open("w", encoding="utf-8") as _fh:
            _pip = subprocess.run(_pip_cmd, stdout=_fh, stderr=subprocess.STDOUT, text=True)
        _pip_tail = _pip_log.read_text(errors="replace").splitlines()[-25:]
        print("\n".join(_pip_tail), flush=True)
        INSTALL_INFO["pip_rc"] = _pip.returncode
        INSTALL_INFO["pip_tail"] = _pip_tail[-8:]
        if _pip.returncode != 0:
            raise RuntimeError(f"pip install rc={_pip.returncode}; tail above")
        INSTALL_STAMP.write_text(STAMP_TEXT, encoding="utf-8")
        INSTALL_INFO["cached"] = False
    INSTALL_INFO["install_s"] = round(time.time() - _t0, 1)
    INSTALL_INFO["site_packages_bytes"] = sum(
        p.stat().st_size for p in SITE_PACKAGES.rglob("*") if p.is_file())

    # version assert via importlib.metadata only (torch is NOT imported in the notebook process)
    _probe = subprocess.run(
        [sys.executable, "-c",
         "import importlib.metadata as m, json; names=('vllm','torch','transformers',"
         "'flashinfer-python','flashinfer-cubin','xgrammar','triton','nvidia-cuda-nvcc');"
         "out={}\n"
         "for n in names:\n"
         "    try: out[n]=m.version(n)\n"
         "    except Exception as e: out[n]='ERR '+repr(e)[:60]\n"
         "print(json.dumps(out))"],
        env=vllm_env(), capture_output=True, text=True)
    _versions = json.loads((_probe.stdout or "{}").strip().splitlines()[-1]) if _probe.stdout.strip() else {}
    INSTALL_INFO["versions"] = _versions
    print("serving-lab4: installed versions", _versions, flush=True)
    _bad = {k: (_versions.get(k), v) for k, v in EXPECTED_VERSIONS.items() if _versions.get(k) != v}
    if _bad:
        raise RuntimeError(f"version assert failed (installed, expected): {_bad}")

    # their CUDA-13.3 toolchain gate (R5 §2.1)
    _required = [CUDA_BIN_PATH / "nvcc", CUDA_HOME_PATH / "include" / "cuda_runtime.h",
                 CUDA_HOME_PATH / "include" / "cuda.h", CUDA_LIB_PATH / "libcudart.so.13",
                 Path(sysconfig.get_path("include")) / "Python.h"]
    _missing_tc = [str(p) for p in _required if not p.is_file()]
    INSTALL_INFO["toolchain_missing"] = _missing_tc
    _env = vllm_env()
    INSTALL_INFO["which"] = {x: shutil.which(x, path=_env["PATH"]) for x in ("nvcc", "ninja", "c++")}
    _nvcc = subprocess.run([str(CUDA_BIN_PATH / "nvcc"), "--version"], env=_env,
                           capture_output=True, text=True) if (CUDA_BIN_PATH / "nvcc").is_file() else None
    INSTALL_INFO["nvcc_version"] = ((_nvcc.stdout or "").strip().splitlines()[-2:] if _nvcc else None)
    INSTALL_INFO["nvcc_release_13_3"] = bool(_nvcc and "release 13.3" in (_nvcc.stdout or ""))
    print("serving-lab4: toolchain", json.dumps({k: INSTALL_INFO[k] for k in
          ("toolchain_missing", "which", "nvcc_version", "nvcc_release_13_3")}, indent=1), flush=True)
    if _missing_tc or not INSTALL_INFO["nvcc_release_13_3"]:
        # Not fatal by design: with the sampler off and FLASH_ATTN auto-selected
        # nothing should JIT; record it and let the boot decide.
        print("serving-lab4: WARNING CUDA-13.3 toolchain gate did not fully pass "
              "(recorded; boot proceeds)", flush=True)
        INSTALL_INFO["toolchain_warning"] = True
    RUNTIME_OK = True
    print(f"serving-lab4: install OK in {INSTALL_INFO['install_s']} s, "
          f"elapsed {elapsed_min():.1f} min", flush=True)
except Exception as _exc:
    traceback.print_exc()
    INSTALL_INFO["error"] = repr(_exc)[:800]
    RESULTS["verdicts"]["q1_boot"] = "INSTALL-FAILED: " + repr(_exc)[:300]
    print("serving-lab4: INSTALL FAILED — every later phase will be skipped", flush=True)
RESULTS["meta"]["install"] = INSTALL_INFO
save_results()
'''


# ---------------------------------------------------------------------------
# Library blocks that REPLACE their lab1 counterparts (block replacement keeps
# the duck-shaped generator + measurement core verbatim).
# ---------------------------------------------------------------------------
LIB_HEADER = r'''# ============================ SERVING LAB LIBRARY (lab4) ============================
# No games are played in this kernel. Load generator + measurement core are the
# arc3-serving-lab / lab3 code (08-22 / 08-26 on this GPU) adapted for:
#   - vLLM 0.24 pack3 argv + CUDA-13 env (start_server), MTP boot ladder
#   - TWO prompt regimes (Session.regime): churn (shipped) vs stable (Pack-1)
#   - prefix-cache hit rate per measured window (vllm:prefix_cache_* deltas)
#   - a greedy correctness battery (MTP must be output-identical under greedy)
'''

LIB_CONSTANTS = r'''VLLM_HOST = "127.0.0.1"
VLLM_PORT = 1234
VLLM_ROOT = f"http://{VLLM_HOST}:{VLLM_PORT}"
VLLM_API = VLLM_ROOT + "/v1"
VLLM_MAX_MODEL_LEN = 65536

LAB_HARD_CAP_MIN = 135.0     # no NEW phase starts after this (kernel budget 150 min)
MTP_OFF_GATE_MIN = 100.0     # MTP no-prefix arm after a corrupt prefix-on arm only before this
MTP_CHURN_GATE_MIN = 110.0   # optional churn-under-MTP arm only before this

# ---- vLLM 0.24 argv: the pack3 recipe (R5 §3.1 / §6.2) verbatim, minus the
# prefix-caching and spec-decode flags, which are parameterised per boot.
# NOT copied from pack3: reasoning_effort "xhigh" (policy, and our 08-21
# dead-decode cluster). KV cache stays default bf16 (no --kv-cache-dtype).
# --limit-mm-per-prompt image:1 IS kept (duck ships MULTIMODAL_CONTEXT=
# current_grid = one image per request) and the generator sends one image.
BASE_SERVE_FLAGS = [
    "--model", str(QWEN_MODEL_PATH),
    "--served-model-name", QWEN_SERVED_MODEL_NAME,
    "--host", VLLM_HOST,
    "--port", str(VLLM_PORT),
    "--tensor-parallel-size", "1",
    "--dtype", "bfloat16",
    "--max-num-seqs", "32",
    "--gpu-memory-utilization", "0.9",
    "--mm-encoder-tp-mode", "data",
    "--limit-mm-per-prompt", '{"image": 1, "video": 0}',
    "--enable-auto-tool-choice",
    "--tool-call-parser", "qwen3_coder",
    "--generation-config", "vllm",
    "--default-chat-template-kwargs", '{"preserve_thinking": true}',
    "--reasoning-parser", "qwen3",
    "--max-model-len", str(VLLM_MAX_MODEL_LEN),
]
# mamba-cache-mode align is what makes prefix caching valid for the GDN
# layers (R5 §6.1 risk 8); with prefix caching OFF the mode must be "none"
# (vllm/config/cache.py), so the OFF rung carries no mamba flag.
PREFIX_ON_FLAGS = ["--enable-prefix-caching", "--mamba-cache-mode", "align"]
PREFIX_OFF_FLAGS = ["--no-enable-prefix-caching"]
MTP2_SPEC = json.dumps({"method": "mtp", "num_speculative_tokens": 2})
MTP2_FLAGS = ["--speculative-config", MTP2_SPEC]

# Q3 boot ladder: never let a failed MTP boot end the kernel.
MTP_LADDER = [
    ("mtp2_prefix_on", PREFIX_ON_FLAGS + MTP2_FLAGS),
    ("mtp2_prefix_off", PREFIX_OFF_FLAGS + MTP2_FLAGS),
    ("base024_recover", PREFIX_ON_FLAGS),
]

CURRENT_SERVER = {"proc": None, "log": str(WORKING_DIR / "vllm-base024.log"),
                  "tag": "none", "flags": []}

RESULTS["meta"].update({
    "max_model_len": VLLM_MAX_MODEL_LEN,
    "base_flags": BASE_SERVE_FLAGS,
    "prefix_on_flags": PREFIX_ON_FLAGS,
    "prefix_off_flags": PREFIX_OFF_FLAGS,
    "mtp2_flags": MTP2_FLAGS,
    "kv_cache_dtype": "default (bf16) — fp8 KV deliberately NOT used",
    "env_switches": {"VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_USE_DEEP_GEMM": "0"},
    "metric_names_verified_v0_24_0": [
        "vllm:prefix_cache_queries", "vllm:prefix_cache_hits", "vllm:generation_tokens",
        "vllm:spec_decode_num_drafts", "vllm:spec_decode_num_draft_tokens",
        "vllm:spec_decode_num_accepted_tokens", "vllm:spec_decode_num_accepted_tokens_per_pos"],
    "decision_rules": {
        "mtp_adopt": "greedy_mismatches == 0 AND stable_conc28_tok_s(MTP) >= 1.3x (no MTP) AND acceptance >= 0.5",
        "prefix_stable_confirmed": "stable window prefix hit rate >= 0.5 AND stable gen tok/s >= 1.5x churn",
        "q1_boot": "0.24 engine serves Qwen/Qwen3.8-27B-FP8 with speculative_config=None on the first boot",
    },
})


'''

LIB_SERVER = r'''# ---- server lifecycle (0.24 pack3 env, own process group) --------------------
ENGINE_LINE_PATTERNS = {
    "engine_config": "Initializing a V1 LLM engine",
    "non_default_args": "non-default args",
    "attention_backend": "attention backend",
    "kv_cache_size": "GPU KV cache size",
    "max_concurrency": "Maximum concurrency for",
    "kv_cache_memory": "Available KV cache memory",
    "model_load": "Model loading took",
    "spec_decode_lines": "spec_decode",
    "flashinfer_jit": "flashinfer/jit",
    "flashinfer_jit_dotted": "flashinfer.jit",
    "cuda_error": "CUDA error",
    "illegal_memory": "illegal memory access",
    "engine_dead": "EngineDeadError",
    "traceback": "Traceback (most recent call last)",
}


def capture_engine_lines(log_path, max_hits=3):
    lines = tail_log_lines(log_path, max_bytes=4_000_000)
    found = {}
    for key, needle in ENGINE_LINE_PATTERNS.items():
        hits = [ln.strip()[:1500] for ln in lines if needle in ln]
        if hits:
            found[key] = {"count": len(hits), "lines": hits[:max_hits]}
    spec = None
    backend = None
    for ln in lines:
        if "speculative_config=" in ln and spec is None:
            spec = ln[ln.index("speculative_config="):][:400]
        m = re.search(r"Using (\S+) attention backend", ln)
        if m and backend is None:
            backend = m.group(1)
    found["speculative_config_str"] = spec
    found["attention_backend_name"] = backend
    found["flashinfer_jit_frames"] = (found.get("flashinfer_jit", {}).get("count", 0)
                                      + found.get("flashinfer_jit_dotted", {}).get("count", 0))
    return found


def print_engine_lines(tag, found):
    print(f"serving-lab4: [{tag}] engine lines:", flush=True)
    for key in ("engine_config", "attention_backend", "kv_cache_size", "max_concurrency",
                "kv_cache_memory", "model_load", "non_default_args"):
        for ln in (found.get(key) or {}).get("lines", [])[:1]:
            print(f"  {key}: {ln[:1200]}", flush=True)
    print(f"  speculative_config: {found.get('speculative_config_str')}", flush=True)
    print(f"  attention backend : {found.get('attention_backend_name')}", flush=True)
    print(f"  flashinfer.jit frames: {found.get('flashinfer_jit_frames')}", flush=True)
    for key in ("flashinfer_jit", "flashinfer_jit_dotted", "cuda_error", "illegal_memory", "engine_dead"):
        for ln in (found.get(key) or {}).get("lines", [])[:2]:
            print(f"  !! {key}: {ln[:600]}", flush=True)


def stop_server(reason):
    print(f"serving-lab4: stopping vLLM ({reason})", flush=True)
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
    print(f"serving-lab4: server stopped, gpu={gpu_sample()}", flush=True)


def served_ids():
    try:
        return [m.get("id") for m in http_json(VLLM_API + "/models", timeout=5).get("data", [])]
    except Exception:
        return None


def start_server(extra_flags, tag, timeout_s=1500):
    """Boot vLLM 0.24 with BASE_SERVE_FLAGS + extra_flags in the pack3 env.
    Raises on death/timeout with the FULL log tail printed (lab2 lesson) and
    the boot recorded under RESULTS['boots'][tag] either way."""
    flags = [*BASE_SERVE_FLAGS, *extra_flags]
    log_path = WORKING_DIR / f"vllm-{tag}.log"
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", *flags]
    env = vllm_env()
    print(f"serving-lab4: starting vLLM [{tag}] (elapsed {elapsed_min():.1f} min):",
          " ".join(cmd), flush=True)
    handle = log_path.open("w", encoding="utf-8")
    t0 = time.time()
    proc = subprocess.Popen(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT,
                            text=True, start_new_session=True)
    CURRENT_SERVER.update({"proc": proc, "log": str(log_path), "tag": tag, "flags": flags})
    boot = {"tag": tag, "flags": flags, "started_utc": datetime.utcnow().isoformat() + "Z"}
    RESULTS["boots"][tag] = boot
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            lines = tail_log_lines(log_path)
            print(f"serving-lab4: [{tag}] BOOT FAILURE rc={proc.returncode} — log tail:", flush=True)
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
                         "engine_lines": found})
            print(f"serving-lab4: vLLM ready [{tag}] in {boot['boot_s']} s, served ids {ids}", flush=True)
            print_engine_lines(tag, found)
            if ids != [QWEN_SERVED_MODEL_NAME]:
                print(f"serving-lab4: WARNING served ids {ids} != [{QWEN_SERVED_MODEL_NAME}]", flush=True)
            save_results()
            return flags
        time.sleep(5)
    lines = tail_log_lines(log_path)
    print(f"serving-lab4: [{tag}] BOOT TIMEOUT — log tail:", flush=True)
    print("\n".join(lines[-150:]), flush=True)
    boot.update({"ok": False, "rc": None, "timeout": True, "boot_s": round(time.time() - t0, 1),
                 "log_tail": lines[-60:], "engine_lines": capture_engine_lines(log_path)})
    save_results()
    raise TimeoutError(f"vLLM [{tag}] not ready in {timeout_s}s")


def reset_prefix_cache():
    # vLLM API server: POST /reset_prefix_cache (frees cached blocks; only
    # effective when no request is running). Best effort — recorded, never fatal.
    try:
        req = urllib.request.Request(VLLM_ROOT + "/reset_prefix_cache", data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status
    except Exception as exc:
        return ("failed: " + repr(exc))[:120]


'''

LIB_SESSION = r'''CHURN_CAP_TOKENS = 24500    # shipped duck: pop the oldest block past this
STABLE_CAP_TOKENS = 30000   # Pack-1 hysteresis: one big cut, rarely (never inside a normal window)
SEED_TURNS = 12             # pre-seeded history: FIRST request ~17k tokens (lab3 calibration:
                            # real tokens ~1.13x the make_transcript estimate), stable regime
                            # ends a 8-min window near ~25k, churn holds ~17-18k


class Session:
    # One synthetic duck "game worker" in one of two prompt regimes:
    #   churn  — today's shipped behaviour: after EVERY request the oldest
    #            history block is dropped, so everything after the system
    #            prompt + short game intro shifts and the cached prefix is
    #            only ~2.5k tokens.
    #   stable — Pack-1 hysteresis: append-only; the image rides only on the
    #            newest turn; trimming is one big cut (half the history) and
    #            only past STABLE_CAP_TOKENS.
    # Both regimes start from the SAME layout (system + intro + SEED_TURNS
    # history turns) so prompt sizes are comparable.
    def __init__(self, idx, seed, regime="churn", images_per_req=1):
        self.rng = random.Random(seed)
        self.idx = idx
        self.regime = regime
        self.images_per_req = images_per_req
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
        content = [{"type": "text", "text": text},
                   {"type": "image_url", "image_url": {
                       "url": "data:image/png;base64," + board_png_b64(self.rng)}}]
        return {"role": "user", "content": content}

    def build_messages(self):
        msgs = [{"role": "system", "content": SYSTEM_TEXT},
                {"role": "user", "content": [{"type": "text", "text":
                    "Game intro and rules so far:\n" + self.base_context}]},
                {"role": "assistant", "content": "Understood. World model initialized."}]
        total = len(self.turns)
        for i, (user_msg, assistant_msg) in enumerate(self.turns):
            if total - i > self.images_per_req - 1:
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

LIB_LOAD_PHASE = r'''def run_load_phase(name, conc, warmup_s, measure_s, regime="churn"):
    if elapsed_min() > LAB_HARD_CAP_MIN:
        print(f"serving-lab4: SKIP {name} — past hard cap ({elapsed_min():.0f} min)", flush=True)
        RESULTS["phases"][name] = {"skipped": "hard-cap", "regime": regime}
        save_results()
        return None
    if not server_alive(15):
        print(f"serving-lab4: SKIP {name} — server not alive", flush=True)
        RESULTS["phases"][name] = {"skipped": "server-dead", "regime": regime}
        save_results()
        return None
    print(f"\nserving-lab4: === {name} (regime={regime}, conc={conc}, warmup={warmup_s}s, "
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
            print(f"serving-lab4: SERVER DIED during {name}", flush=True)
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
    cum_q = metrics_after.get("vllm:prefix_cache_queries", 0.0)
    cum_h = metrics_after.get("vllm:prefix_cache_hits", 0.0)
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
        "prefix_cache_hit_rate_cumulative": round(cum_h / cum_q, 4) if cum_q > 0 else None,
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
        "acceptance": acceptance_delta(metrics_before, metrics_after),
        "acceptance_log_lines": log_acceptance_lines(),
        "metric_lines_sample": metric_lines[:12],
        "gpu_samples_tail": gpu_samples[-6:],
        "server_died": server_died,
        "server_alive_at_end": server_alive(),
    }
    RESULTS["phases"][name] = result
    save_results()
    print(f"serving-lab4: {name}: {per_session_of(result)} gen-tok/min/session | "
          f"{result['gen_tok_s_aggregate_metric'] or result['gen_tok_s_aggregate_usage']} tok/s agg | "
          f"prefix hit {result['prefix_cache_hit_rate_window']} | "
          f"{len(window)} reqs / {len(errors)} errs | prompt~{result['prompt_tokens_mean']} | "
          f"p50 lat {result['p50_latency_s']}s | trims/session {result['session_trims_mean']}", flush=True)
    if result["acceptance"]["num_draft_tokens"] > 0:
        print(f"serving-lab4: {name}: acceptance_rate="
              f"{result['acceptance']['acceptance_rate']:.3f} "
              f"mean_accepted_per_draft={result['acceptance']['mean_accepted_per_draft']:.2f}", flush=True)
    return result


'''

LIB_BATTERY = r'''# ---- greedy correctness battery (MTP must be output-identical under greedy) ----
BATTERY_N = 20
BATTERY_MAX_TOKENS = 224


def battery_prompts():
    prompts = []
    for k in range(BATTERY_N):
        rng = random.Random(4242 + k)
        text = make_transcript(rng, 2200)
        tool_mode = (k % 4 == 3)     # 5 of 20 carry the python tool (parser path)
        ask = ("use the python tool to print grid[0][0] before you act."
               if tool_mode else "state the next three actions with one-line justifications.")
        msgs = [{"role": "system", "content": SYSTEM_TEXT},
                {"role": "user", "content": [
                    {"type": "text", "text": "Game transcript so far:\n" + text
                        + "\n\nAnalyze the latest board change and " + ask},
                    {"type": "image_url", "image_url": {
                        "url": "data:image/png;base64," + board_png_b64(rng)}}]}]
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
    print(f"serving-lab4: === {name} (greedy, {BATTERY_N} prompts, sequential, "
          f"max_tokens {max_tokens}, elapsed {elapsed_min():.1f} min) ===", flush=True)
    before, _ = scrape_metrics()
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
            outs.append({"k": p["k"], "tools": p["tools"], "text": text,
                         "sha16": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
                         "completion_tokens": usage.get("completion_tokens"),
                         "finish_reason": choice.get("finish_reason"),
                         "n_tool_calls": len(calls)})
        except Exception as exc:
            outs.append({"k": p["k"], "tools": p["tools"], "error": repr(exc)[:200]})
    after, _ = scrape_metrics()
    result = {"tag": tag, "server_tag": CURRENT_SERVER["tag"], "n": len(outs),
              "errors": sum(1 for o in outs if "error" in o),
              "elapsed_s": round(time.time() - t0, 1),
              "acceptance": acceptance_delta(before, after),
              "outputs": outs}
    RESULTS["phases"][name] = result
    save_results()
    print(f"serving-lab4: {name}: {result['n'] - result['errors']}/{result['n']} ok in "
          f"{result['elapsed_s']} s; shas {[o.get('sha16') for o in outs][:6]}...", flush=True)
    return result


def battery_mismatches(a, b):
    if not a or not b or "outputs" not in a or "outputs" not in b:
        return None
    da = {o["k"]: o for o in a["outputs"]}
    db = {o["k"]: o for o in b["outputs"]}
    compared, mism = 0, []
    for k in sorted(da):
        oa, ob = da[k], db.get(k)
        if ob is None or "error" in oa or "error" in ob:
            continue
        compared += 1
        if oa["text"] != ob["text"]:
            i = next((i for i, (x, y) in enumerate(zip(oa["text"], ob["text"])) if x != y),
                     min(len(oa["text"]), len(ob["text"])))
            mism.append({"k": k, "tools": oa.get("tools"), "first_divergence_char": i,
                         "len_a": len(oa["text"]), "len_b": len(ob["text"]),
                         "a_head": oa["text"][max(0, i - 40):i + 60],
                         "b_head": ob["text"][max(0, i - 40):i + 60]})
    return {"a": a.get("tag"), "b": b.get("tag"), "compared": compared,
            "mismatches": len(mism), "detail": mism[:20]}


print(f"serving-lab4: library ready, elapsed {elapsed_min():.1f} min")
save_results()
'''


# ---------------------------------------------------------------------------
# Phase cells
# ---------------------------------------------------------------------------
CELL_BOOT_BASE = r'''# ===================== Q1 — BOOT vLLM 0.24 (pack3 recipe, prefix caching ON, no MTP) =====================
BASE_TAG = "base024"
try:
    if not RUNTIME_OK:
        raise RuntimeError("runtime install failed — boot skipped")
    start_server(PREFIX_ON_FLAGS, tag=BASE_TAG)
    _b = RESULTS["boots"][BASE_TAG]
    _el = _b.get("engine_lines") or {}
    _spec_none = (_el.get("speculative_config_str") or "").startswith("speculative_config=None")
    _ids_ok = _b.get("served_ids") == [QWEN_SERVED_MODEL_NAME]
    RESULTS["verdicts"]["q1_boot"] = (
        f"BOOT OK in {_b.get('boot_s')} s — attention backend {_el.get('attention_backend_name')}, "
        f"speculative_config {'None' if _spec_none else (_el.get('speculative_config_str') or 'NOT FOUND IN LOG')}, "
        f"flashinfer.jit frames {_el.get('flashinfer_jit_frames')}, served ids ok={_ids_ok}")
    # vLLM's own identity, read from the served process' interpreter (metadata only)
    _ver = subprocess.run([sys.executable, "-c",
                           "import importlib.metadata as m; print(m.version('vllm'), m.version('torch'))"],
                          env=vllm_env(), capture_output=True, text=True)
    RESULTS["meta"]["vllm_torch_versions_served"] = (_ver.stdout or _ver.stderr or "").strip()[:120]
except Exception as _exc:
    traceback.print_exc()
    RESULTS["verdicts"].setdefault("q1_boot", "BOOT FAILED: " + repr(_exc)[:300])
    print("serving-lab4: Q1 BOOT FAILED — recorded; later phases will skip on server-dead", flush=True)
print("serving-lab4: Q1 VERDICT:", RESULTS["verdicts"].get("q1_boot"), flush=True)
save_results()
'''


def _wrapped_attest(attest_cell: str) -> str:
    body = textwrap.indent(attest_cell, "    ")
    return (
        "# ============ weight attestation (doctrine v2, verbatim from duck38 v12; failure-wrapped) ============\n"
        "try:\n"
        "    if not server_alive(15):\n"
        "        raise RuntimeError('server not alive — attestation skipped')\n"
        + body +
        "\n    RESULTS['phases']['attest'] = {'ok': True}\n"
        "except Exception as _exc:\n"
        "    traceback.print_exc()\n"
        "    RESULTS['phases']['attest'] = {'ok': False, 'error': repr(_exc)[:400]}\n"
        "    print('serving-lab4: ATTEST FAILED — recorded, kernel continues', flush=True)\n"
        "save_results()\n"
    )


CELL_Q2 = r'''# ===================== Q2 — parser, greedy battery x2, conc-1 probe, conc-28 CHURN vs STABLE =====================
try:
    parser_roundtrip(BASE_TAG)
except Exception:
    traceback.print_exc()
try:
    run_battery(BASE_TAG)                 # reference greedy outputs for Q3
    run_battery(BASE_TAG + "_rep")        # base-vs-base repeat = noise floor of the instrument
    RESULTS["verdicts"]["greedy_floor_base_vs_base"] = battery_mismatches(
        RESULTS["phases"].get(f"battery_{BASE_TAG}"), RESULTS["phases"].get(f"battery_{BASE_TAG}_rep"))
    print("serving-lab4: greedy noise floor (base vs base):",
          {k: v for k, v in (RESULTS["verdicts"]["greedy_floor_base_vs_base"] or {}).items() if k != "detail"},
          flush=True)
except Exception:
    traceback.print_exc()
save_results()

for _name, _conc, _warm, _meas, _regime in [
        ("probe_conc1_" + BASE_TAG, 1, 30, 120, "stable"),
        ("churn_conc28_" + BASE_TAG, 28, 90, 480, "churn"),
        ("stable_conc28_" + BASE_TAG, 28, 90, 480, "stable")]:
    try:
        run_load_phase(_name, _conc, _warm, _meas, regime=_regime)
    except Exception:
        traceback.print_exc()
        RESULTS["phases"].setdefault(_name, {"error": "see traceback", "regime": _regime})
        save_results()

_churn = RESULTS["phases"].get("churn_conc28_" + BASE_TAG) or {}
_stable = RESULTS["phases"].get("stable_conc28_" + BASE_TAG) or {}
_c_tps, _s_tps = agg_of(_churn), agg_of(_stable)
_s_hit = _stable.get("prefix_cache_hit_rate_window")
if _c_tps and _s_tps:
    _ratio = _s_tps / _c_tps
    _ok = (_s_hit is not None and _s_hit >= 0.5) and _ratio >= 1.5
    RESULTS["verdicts"]["q2_prefix_stable"] = (
        f"{'CONFIRMED' if _ok else 'NOT CONFIRMED'} — stable {_s_tps} tok/s vs churn {_c_tps} tok/s "
        f"({_ratio:.2f}x; rule >= 1.5x), stable hit rate {_s_hit} (rule >= 0.5), "
        f"churn hit rate {_churn.get('prefix_cache_hit_rate_window')}")
else:
    RESULTS["verdicts"]["q2_prefix_stable"] = "NO-DATA (a conc-28 regime phase produced no rate)"
print("serving-lab4: Q2 VERDICT:", RESULTS["verdicts"]["q2_prefix_stable"], flush=True)
print("serving-lab4: Q2 done at", round(elapsed_min(), 1), "min", flush=True)
save_results()
'''


CELL_Q3 = r'''# ===================== Q3 — MTP nst=2 WITH prefix caching (ladder: -> no-prefix -> base recovery) =====================
MTP_TAG = None
try:
    if elapsed_min() > LAB_HARD_CAP_MIN:
        raise RuntimeError(f"hard cap reached ({elapsed_min():.0f} min) — MTP arm skipped")
    if not RUNTIME_OK:
        raise RuntimeError("runtime install failed — MTP arm skipped")
    stop_server("switch to MTP nst=2 (prefix caching ON)")
    for _tag, _extra in MTP_LADDER:
        try:
            start_server(_extra, tag=_tag)
            MTP_TAG = _tag
            break
        except Exception as _exc:
            traceback.print_exc()
            RESULTS["boots"].setdefault(_tag, {})["error"] = repr(_exc)[:600]
            print(f"serving-lab4: boot [{_tag}] FAILED — next rung of the ladder", flush=True)
            save_results()
            stop_server(f"cleanup after failed {_tag}")
    RESULTS["verdicts"]["q3_boot_rung"] = MTP_TAG
    if MTP_TAG is None:
        RESULTS["verdicts"]["q3_mtp"] = "MTP-FATAL (BOOT): every rung failed, including base recovery"
    elif MTP_TAG == "base024_recover":
        RESULTS["verdicts"]["q3_mtp"] = ("MTP-FATAL (BOOT): MTP failed both with and without prefix "
                                         "caching; base 0.24 recovered")
except Exception as _exc:
    traceback.print_exc()
    RESULTS["verdicts"]["q3_mtp"] = "Q3 SKIPPED: " + repr(_exc)[:300]
save_results()


def mtp_quality(tag):
    """Cheap quality gate on one MTP server (~6 min): spec-decode probe,
    parser round-trip, greedy battery vs base. Returns (corrupt, mismatches)."""
    arm = {}
    _before, _ = scrape_metrics()
    _probe_session = Session(999, seed=999, regime="stable")
    chat_request(_probe_session, 256, timeout=900)
    _after, _lines = scrape_metrics()
    _probe = acceptance_delta(_before, _after)
    arm["spec_decode_active"] = (_probe["num_draft_tokens"] or 0) > 0
    arm["probe"] = _probe
    arm["spec_metric_lines"] = [ln for ln in _lines if "spec_decode" in ln][:8]
    RESULTS["phases"][f"{tag}_probe"] = arm
    if arm["spec_decode_active"]:
        print(f"serving-lab4: [{tag}] MTP ACTIVE — probe acceptance_rate="
              f"{(_probe['acceptance_rate'] if _probe['acceptance_rate'] is not None else float('nan')):.3f}",
              flush=True)
    else:
        print(f"serving-lab4: [{tag}] WARNING — MTP INERT: no spec_decode draft tokens after the "
              "probe request (draft head missing? flag dropped? metric names?)", flush=True)
    save_results()
    parser_roundtrip(tag)
    run_battery(tag)
    mism = battery_mismatches(RESULTS["phases"].get(f"battery_{BASE_TAG}"),
                              RESULTS["phases"].get(f"battery_{tag}"))
    RESULTS["verdicts"][f"greedy_mismatch_{tag}_vs_base"] = mism
    print(f"serving-lab4: [{tag}] greedy mismatches vs base:",
          {k: v for k, v in (mism or {}).items() if k != "detail"}, flush=True)
    parser = RESULTS["phases"].get(f"parser_roundtrip_{tag}") or {}
    dead = not server_alive(10) and not vllm_procs()
    corrupt = ((mism or {}).get("mismatches", 0) > 0) or parser.get("verdict") == "FAIL" or dead
    RESULTS["verdicts"][f"q3_quality_{tag}"] = {
        "corrupt": corrupt, "mismatches": (mism or {}).get("mismatches"),
        "parser": parser.get("verdict"), "server_dead": dead,
        "spec_decode_active": arm["spec_decode_active"]}
    save_results()
    return corrupt, mism


def mtp_loads(tag, full=True):
    """Throughput on one MTP server: conc-1 probe, conc-28 stable (+churn if time)."""
    run_load_phase(f"probe_conc1_{tag}", 1, 30, 120, regime="stable")
    run_load_phase(f"stable_conc28_{tag}", 28, 90, 480, regime="stable")
    if full and elapsed_min() < MTP_CHURN_GATE_MIN:
        run_load_phase(f"churn_conc28_{tag}", 28, 60, 360, regime="churn")
    if not server_alive(10) and not vllm_procs():
        RESULTS["verdicts"][f"q3_crash_{tag}"] = f"MTP-FATAL (LOAD): server died during the {tag} arm"
        print(f"serving-lab4: MTP-FATAL (LOAD) on {tag}", flush=True)
    save_results()


try:
    if MTP_TAG in ("mtp2_prefix_on", "mtp2_prefix_off"):
        _corrupt, _mism = mtp_quality(MTP_TAG)
        _load_tag = MTP_TAG
        # R5 §6.1 risk 6: if MTP+prefix caching looks corrupt (the GDN landmine,
        # fix PR #47861 unverified in 0.24), switch to the no-prefix rung BEFORE
        # spending the load budget — time-gated. Quality first, loads second.
        if MTP_TAG == "mtp2_prefix_on" and _corrupt and elapsed_min() < MTP_OFF_GATE_MIN:
            print(f"serving-lab4: MTP+prefix arm looks CORRUPT "
                  f"({RESULTS['verdicts'].get('q3_quality_mtp2_prefix_on')}) — "
                  "switching to --no-enable-prefix-caching before the load phases", flush=True)
            stop_server("MTP prefix-on arm corrupt -> prefix-off rung")
            try:
                start_server(PREFIX_OFF_FLAGS + MTP2_FLAGS, tag="mtp2_prefix_off")
                RESULTS["verdicts"]["q3_boot_rung_secondary"] = "mtp2_prefix_off"
                mtp_quality("mtp2_prefix_off")
                _load_tag = "mtp2_prefix_off"
            except Exception as _exc:
                traceback.print_exc()
                RESULTS["verdicts"]["q3_boot_rung_secondary"] = "mtp2_prefix_off FAILED: " + repr(_exc)[:200]
                _load_tag = None
        elif MTP_TAG == "mtp2_prefix_on" and _corrupt:
            print("serving-lab4: MTP+prefix arm looks CORRUPT but past the prefix-off gate — "
                  "loads still run on it (throughput is informative even if outputs are not)", flush=True)
        if _load_tag and server_alive(10):
            mtp_loads(_load_tag, full=(_load_tag == MTP_TAG))
except Exception:
    traceback.print_exc()
    RESULTS["verdicts"].setdefault("q3_mtp", "Q3 ARM ERROR (see traceback)")
print("serving-lab4: Q3 done at", round(elapsed_min(), 1), "min", flush=True)
save_results()
'''


CELL_FINAL = r'''# ===================== FINAL — decision table + pre-registered verdicts =====================
def _cell(value, width=12):
    return ("-" if value is None else str(value)).rjust(width)


def _acc(phase):
    rate = ((phase or {}).get("acceptance") or {}).get("acceptance_rate")
    return f"{rate:.3f}" if isinstance(rate, float) else None


_base_boot = RESULTS["boots"].get(BASE_TAG) or {}
_el = _base_boot.get("engine_lines") or {}
print("\n" + "=" * 100)
print("serving-lab4 — vLLM 0.24 on the Kaggle RTX Pro 6000")
print("=" * 100)
print("GPU/driver :", RESULTS["meta"].get("gpu_name"), "| driver", RESULTS["meta"].get("driver_version"),
      "| python", RESULTS["meta"].get("python"))
print("versions   :", (RESULTS["meta"].get("install") or {}).get("versions"))
print("Q1 BOOT    :", RESULTS["verdicts"].get("q1_boot"))
for _k in ("engine_config", "attention_backend"):
    for _ln in (_el.get(_k) or {}).get("lines", [])[:1]:
        print(f"   {_k}: {_ln[:900]}")
print("   speculative_config:", _el.get("speculative_config_str"))
print("   flashinfer.jit frames:", _el.get("flashinfer_jit_frames"))
_att = RESULTS["phases"].get("attest") or {}
print("   attest:", "OK" if _att.get("ok") else _att)
print("   MTP index:", {k: RESULTS["meta"].get("mtp_index", {}).get(k) for k in
                        ("tensors_in_mtp_safetensors", "files_holding_mtp_named_tensors",
                         "config_mtp_num_hidden_layers")})

print("-" * 100)
print("Q2/Q3 LOAD TABLE — conc-28 unless noted (window = measured minutes after warmup)")
print(f"{'phase':<34}{'tok/s agg':>11}{'tok/min/sess':>14}{'prefix hit':>12}{'prompt~':>9}"
      f"{'p50 lat':>9}{'accept':>8}{'reqs/err':>10}")
_phase_names = [n for n in RESULTS["phases"] if n.startswith(("churn_", "stable_", "probe_conc1"))]
for _n in _phase_names:
    _p = RESULTS["phases"][_n]
    if _p.get("skipped") or _p.get("error"):
        print(f"{_n:<34}{'(' + str(_p.get('skipped') or _p.get('error')) + ')':>11}")
        continue
    print(f"{_n:<34}{_cell(agg_of(_p), 11)}{_cell(per_session_of(_p), 14)}"
          f"{_cell(_p.get('prefix_cache_hit_rate_window'), 12)}{_cell(_p.get('prompt_tokens_mean'), 9)}"
          f"{_cell(_p.get('p50_latency_s'), 9)}{_cell(_acc(_p), 8)}"
          f"{_cell(str(_p.get('requests_in_window')) + '/' + str(_p.get('errors')), 10)}")

print("-" * 100)
print("Q2 RULE    : stable confirmed if window prefix hit rate >= 0.5 AND gen tok/s >= 1.5x churn")
print("Q2 VERDICT :", RESULTS["verdicts"].get("q2_prefix_stable"))

# ---- Q3 verdict (pre-registered) ----
_rung = RESULTS["verdicts"].get("q3_boot_rung")
_mtp_tag = _rung if _rung in ("mtp2_prefix_on", "mtp2_prefix_off") else None
_floor = RESULTS["verdicts"].get("greedy_floor_base_vs_base") or {}
print("Q3 RULE    : adopt MTP only if greedy mismatches == 0 AND stable conc-28 tok/s(MTP) >= 1.3x base "
      "AND acceptance >= 0.5")
print("greedy floor (base vs base):", {k: v for k, v in _floor.items() if k != "detail"} or "n/a")
if _mtp_tag is None:
    print("Q3 VERDICT :", RESULTS["verdicts"].get("q3_mtp"))
else:
    _parts = []
    for _t in ("mtp2_prefix_on", "mtp2_prefix_off"):
        _st = RESULTS["phases"].get(f"stable_conc28_{_t}")
        if not RESULTS["boots"].get(_t, {}).get("ok"):
            continue
        _mism = (RESULTS["verdicts"].get(f"greedy_mismatch_{_t}_vs_base") or {})
        _m = _mism.get("mismatches")
        _s_mtp, _s_base = agg_of(_st), agg_of(RESULTS["phases"].get("stable_conc28_" + BASE_TAG))
        _ratio = (_s_mtp / _s_base) if (_s_mtp and _s_base) else None
        _acc_rate = ((_st or {}).get("acceptance") or {}).get("acceptance_rate")
        if _acc_rate is None:
            _acc_rate = ((RESULTS["phases"].get(f"battery_{_t}") or {}).get("acceptance") or {}).get("acceptance_rate")
        _probe = (RESULTS["phases"].get(f"{_t}_probe") or {})
        _crash = RESULTS["verdicts"].get(f"q3_crash_{_t}")
        _p1_mtp = agg_of(RESULTS["phases"].get(f"probe_conc1_{_t}"))
        _p1_base = agg_of(RESULTS["phases"].get("probe_conc1_" + BASE_TAG))
        _adopt = (_m == 0 and _ratio is not None and _ratio >= 1.3
                  and _acc_rate is not None and _acc_rate >= 0.5 and not _crash)
        _txt = (f"{_t}: {'ADOPT' if _adopt else 'DO NOT ADOPT'} — mismatches {_m} "
                f"(compared {_mism.get('compared')}), stable conc-28 {_s_mtp} vs {_s_base} tok/s "
                f"({_ratio:.2f}x)" if _ratio else
                f"{_t}: {'ADOPT' if _adopt else 'DO NOT ADOPT'} — mismatches {_m}, stable conc-28 NO-DATA")
        _txt += (f", acceptance {_acc_rate:.3f}" if isinstance(_acc_rate, float) else ", acceptance n/a")
        _txt += f", spec_decode_active={_probe.get('spec_decode_active')}"
        _txt += (f", conc-1 {_p1_mtp} vs {_p1_base} tok/s ({_p1_mtp / _p1_base:.2f}x)"
                 if (_p1_mtp and _p1_base) else "")
        _txt += (f", {_crash}" if _crash else "")
        _parts.append(_txt)
        RESULTS["verdicts"][f"q3_adopt_{_t}"] = _adopt
    RESULTS["verdicts"]["q3_mtp"] = " || ".join(_parts) if _parts else RESULTS["verdicts"].get("q3_mtp")
    print("Q3 VERDICT :", RESULTS["verdicts"]["q3_mtp"])
    if _floor.get("mismatches"):
        print("   NOTE: base-vs-base greedy floor is NON-ZERO — the mismatch instrument cannot "
              "attribute differences to MTP alone")

print("-" * 100)
print("boots:", {k: {"ok": v.get("ok"), "boot_s": v.get("boot_s"),
                    "backend": (v.get("engine_lines") or {}).get("attention_backend_name"),
                    "spec": ((v.get("engine_lines") or {}).get("speculative_config_str") or "")[:80]}
                for k, v in RESULTS["boots"].items()})
RESULTS["meta"]["finished_utc"] = datetime.utcnow().isoformat() + "Z"
RESULTS["meta"]["total_elapsed_min"] = round(elapsed_min(), 1)
save_results()
print("=" * 100)
print(f"serving-lab4: DONE in {elapsed_min():.1f} min — results JSON: {RESULTS_PATH} "
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
    duck-shaped generator + measurement core kept verbatim."""
    lab1 = runpy.run_path(str(LAB1_BUILD))
    lib = lab1["CELL_LAB_LIB"]

    # header comment
    lib = _replace_between(lib, "# ============================ SERVING LAB LIBRARY",
                           "import base64\n", LIB_HEADER)
    lib = lib.replace("import base64\n", "import base64\nimport hashlib\nimport signal\nimport zlib\n", 1)
    # constants + RESULTS block -> lab4 constants (RESULTS lives in the imports cell)
    lib = _replace_between(lib, 'VLLM_HOST = "127.0.0.1"', "def elapsed_min():", LIB_CONSTANTS)
    # elapsed_min/save_results are defined identically in the imports cell — drop the dupes
    lib = _replace_between(lib, "def elapsed_min():", "def http_json(", "")
    # server lifecycle -> 0.24 pack3 env + process group + engine-line capture
    lib = _replace_between(lib, "# ---- server lifecycle", "# ---- duck-shaped request synthesis", LIB_SERVER)
    # Session -> regime-aware
    lib = _replace_between(lib, "class Session:", "def chat_request(session", LIB_SESSION)
    # load phase -> prefix-cache deltas, regime, deterministic seeds
    lib = _replace_between(lib, "def run_load_phase(", "# ---- parser round-trip", LIB_LOAD_PHASE)
    # long-context soak (lab1 Phase C) is not part of lab4 -> replace with the battery
    lib = _replace_between(lib, "# ---- long-context soak", 'print(f"serving-lab: library ready', LIB_BATTERY)
    lib = lib.replace('print(f"serving-lab: library ready, elapsed {elapsed_min():.1f} min")\nsave_results()\n', "", 1)
    # small adaptations inside kept blocks
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
                      "# ---- Prometheus scrape (metric names re-verified at the vLLM v0.24.0 tag) ----", 1)
    lib = lib.replace("# vLLM v0.19.0 vllm/v1/spec_decode/metrics.py logs",
                      "# vLLM v0.24.0 vllm/v1/spec_decode/metrics.py logs", 1)
    # per_session_of + an aggregate helper used by the verdict cells
    lib = lib.replace(
        "def run_load_phase(",
        "def agg_of(phase):\n"
        "    if not phase:\n"
        "        return None\n"
        '    metric = phase.get("gen_tok_s_aggregate_metric")\n'
        '    return metric if metric is not None else phase.get("gen_tok_s_aggregate_usage")\n\n\n'
        "def run_load_phase(", 1)
    lib = lib.replace("serving-lab:", "serving-lab4:")
    # invariants: everything lab1-specific is gone, lab4 pieces are in
    for gone in ("MTP3_FLAGS", "MODAL_H100", "long_context_soak", "Q1_RULE", "NST2_GATE_MIN",
                 "serving_lab_results.json", '"kernel": "arc3-serving-lab"'):
        assert gone not in lib, f"lab1 leftover in library: {gone}"
    for keep in ("def make_transcript", "def board_png_b64", "def chat_request", "def parser_roundtrip",
                 "def scrape_metrics", "def acceptance_delta", "def drain_inflight", "PYTHON_TOOL",
                 "def run_load_phase", "def run_battery", "def battery_mismatches", "def start_server",
                 "def stop_server", "def reset_prefix_cache", "def capture_engine_lines",
                 "regime=", "prefix_cache_hit_rate_window", "--mamba-cache-mode",
                 "MTP_LADDER", "def agg_of"):
        assert keep in lib, f"missing in lab4 library: {keep}"
    return lib


def build_notebook() -> dict:
    attest_cell = runpy.run_path(str(DUCK38_BUILD))["ATTEST_CELL"]
    inputs_cell = (CELL_INPUTS
                   .replace("@@WH_WHEEL_COUNT@@", str(WH_WHEEL_COUNT))
                   .replace("@@WH_TOTAL_BYTES@@", str(WH_TOTAL_BYTES))
                   .replace("@@WH_KEY_WHEELS@@", repr(WH_KEY_WHEELS))
                   .replace("@@WH_SHA_PREFIX@@", WH_MANIFEST_SHA_PREFIX)
                   .replace("@@WH_SHA_SUFFIX@@", WH_MANIFEST_SHA_SUFFIX))
    assert "@@" not in inputs_cell
    cells = [
        _md_cell(MD_HEADER),
        _code_cell(CELL_IMPORTS),        # driver first + fail-fast GPU assert
        _code_cell(inputs_cell),         # wheelhouse identity, model mount, MTP index
        _code_cell(CELL_INSTALL),        # offline pip --target, version assert, nvcc gate, vllm_env()
        _code_cell(_lab_library()),      # generator + measurement (lab1/lab3) adapted
        _code_cell(CELL_BOOT_BASE),      # Q1
        _code_cell(_wrapped_attest(attest_cell)),
        _code_cell(CELL_Q2),             # Q2
        _code_cell(CELL_Q3),             # Q3
        _code_cell(CELL_FINAL),
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
    assert joined.index("DRIVER + GPU FIRST") < joined.index("WRONG-GPU") < joined.index("INPUTS —")
    assert joined.index("INPUTS —") < joined.index("INSTALL —") < joined.index("SERVING LAB LIBRARY (lab4)")
    assert joined.index("SERVING LAB LIBRARY (lab4)") < joined.index("Q1 — BOOT") < joined.index("attest: OK")
    assert joined.index("attest: OK") < joined.index("Q2 — parser") < joined.index("Q3 — MTP") < joined.index("FINAL —")
    # Recipe invariants (R5 §6.2)
    for needed in ('"VLLM_USE_FLASHINFER_SAMPLER": "0"', '"VLLM_USE_DEEP_GEMM": "0"',
                   '"--mamba-cache-mode", "align"', '"--enable-prefix-caching"', '"--no-enable-prefix-caching"',
                   '"method": "mtp"', '"num_speculative_tokens": 2', 'SITE_PACKAGES / "nvidia" / "cu13"',
                   "requirements-runtime.txt", "WHEELHOUSE_MANIFEST.json", "mtp.safetensors",
                   "--only-binary", "--no-cache-dir", "start_new_session=True",
                   "vllm:spec_decode_num_accepted_tokens", "vllm:prefix_cache_hits",
                   "reset_prefix_cache", "TORCH_CUDA_ARCH_LIST", "driver_version"):
        assert needed in joined, f"missing from kernel: {needed}"
    assert '"--kv-cache-dtype"' not in joined, "KV must stay bf16"
    assert "reasoning_effort" not in joined.replace("reasoning_effort \"xhigh\" (policy", ""), "no xhigh policy"
    # No games, no submission machinery.
    for banned in ("submission.parquet", "GameAgent", "scorecard", "arc_agi_3_wheels"):
        assert banned not in joined, f"banned token in lab kernel: {banned}"

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
        "dataset_sources": ["jcole75/arc3-qwen36-runtime-wheels"],
        "kernel_sources": [],
        # THE RTX Pro 6000 GATE (08-22 push lesson): the competition source is
        # what admits the kernel to the scored GPU pool.
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "model_sources": ["foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1"],
    }, indent=2) + "\n")

    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    print("built", KERNEL_SLUG, "code-cell sha256", hashlib.sha256(code.encode()).hexdigest())
    print("cells:", len(notebook["cells"]), "| notebook:", nb_path, "|", nb_path.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
