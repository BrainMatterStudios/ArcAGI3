# =============================================================================
# Off-Kaggle replica of the BEST PUBLIC ARC-AGI-3 notebook's serving regime
# (keithtyser V14, "duck-qwen3-8-flash-next-nvfp4-mtp") on Modal, for
# telemetry experiments while the Kaggle GPU quota is exhausted.
#
# What it reproduces (every value pinned by offkaggle/KEITH_REGIME.md and
# checked by offkaggle/test_flashnext_serve.py):
#   * model: RadixArk/Qwen3.8-Flash-Next-NVFP4 @ 7b71922 (the exact HF
#     revision behind Keith's Kaggle model), served as
#     'Qwen/Qwen3.8-Flash-Next-NVFP4' with the model's own chat_template.jinja.
#   * runtime: the SAME docker image Keith ships as layer blobs —
#     vllm/vllm-openai:qwen38-flash-next pulled BY DIGEST (vLLM
#     0.1.dev20073+g8e685d198, CUDA 13.0, py3.12) — with his 3.9 KB PLE-FP8
#     loader patch applied at image build by his own applicator script
#     (refuses unless the stock file hash matches; verifies the result).
#   * the exact `vllm serve` argv (profile kv5-bf16-mtp3-c8-cg32: 8 seqs,
#     5 GiB KV, 8192 batched tokens, cudagraph 32, MTP 3, prefix caching OFF,
#     async scheduling, chunked prefill, bf16, modelopt_fp4, mp executor).
#   * the exact server process environment (PLE CPU offload, no expandable
#     segments, spawn workers, OMP 1, offline HF, arch 12.0, ...).
#   * the GPU: Modal "RTX-PRO-6000" (the Kaggle eval GPU; NVFP4 needs
#     Blackwell — H100 cannot run modelopt_fp4).
#
# Deviations are listed in KEITH_REGIME.md (host RAM 128 GiB vs 189 GB,
# compile caches persisted on the Volume, Modal driver 580.95 vs 580.159).
#
# Auth/cost model is the same as the other offkaggle rigs: a stdlib bearer-
# token reverse proxy (secret 'arc3-vllm-token', key TOKEN) with GET
# /v1/models and GET /health exempt; scale-to-zero after 15 min idle; a 4 h
# in-container hard lifetime cap; max_containers=1.
#
# Entrypoints (modal CLI at ~/Library/Python/3.14/bin/modal):
#     modal run    offkaggle/modal_flashnext_serve.py::warm    # CPU: 135 GB -> Volume
#     modal run    offkaggle/modal_flashnext_serve.py::probe   # CPU: image identity
#     modal deploy offkaggle/modal_flashnext_serve.py          # prints the URL
#     modal run    offkaggle/modal_flashnext_serve.py::smoke --url https://... [--token ...]
# =============================================================================
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

try:  # modal is only needed to deploy/run; the tests import this module
    import modal  # noqa: F401  # on a host without the modal client.
except ModuleNotFoundError:  # pragma: no cover - exercised on bare hosts
    modal = None

APP_NAME = "arc3-flashnext"
VOLUME_NAME = "arc3-hf-cache"          # shared with the other offkaggle rigs
SECRET_NAME = "arc3-vllm-token"        # modal secret create arc3-vllm-token TOKEN=<random>

# --- GPU -------------------------------------------------------------------
# Modal GPU string verified on https://modal.com/docs/guide/gpu (2026-09-02):
# "T4 L4 A10 L40S A100 A100-40GB A100-80GB RTX-PRO-6000 H100 H200 B200 B300".
# Pricing page: "Nvidia RTX PRO 6000 $0.000842 / sec" (~$3.03/h).
GPU_KIND = "RTX-PRO-6000"
# Fallback GPU classes if no RTX PRO 6000 worker can be scheduled (09-03: a container was
# preempted and the replacement waited on capacity). B200 is also Blackwell (native NVFP4);
# a run on the fallback is recorded in /arc3/identity (nvidia-smi rows) and is a deviation.
GPU_FALLBACK: list[str] = []   # 09-03: a B200 fallback silently confounded a single-knob arm; fidelity first
N_GPU = 1

# --- model (KEITH_REGIME.md §1) --------------------------------------------
HF_MODEL_REPO = "RadixArk/Qwen3.8-Flash-Next-NVFP4"
HF_MODEL_REVISION = "7b719225242aacd3dbd3f9407468c2ee9a9d2594"
MODEL_CONFIG_SHA256 = "e765305daba0951974308f4d32c075b52a6a45974730d273f2216718a994d624"
SERVED_MODEL_NAME = "Qwen/Qwen3.8-Flash-Next-NVFP4"
MODEL_TOTAL_BYTES = 135_253_622_894

# --- runtime (KEITH_REGIME.md §2) ------------------------------------------
VLLM_VERSION = "0.1.dev20073+g8e685d198"
VLLM_IMAGE = "vllm/vllm-openai:qwen38-flash-next"
VLLM_IMAGE_INDEX_DIGEST = "sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8"
VLLM_IMAGE_AMD64_DIGEST = "sha256:0aea30240f3e3d9ffae8526643950e170eb5fa07fc427016a9dd90892afa2aa3"
VLLM_IMAGE_REF = f"vllm/vllm-openai@{VLLM_IMAGE_AMD64_DIGEST}"   # pulled by digest
IMAGE_PYTHON = "/usr/bin/python3"
SITE_PACKAGES = "/usr/local/lib/python3.12/dist-packages"
CUDA_HOME = "/usr/local/cuda-13.0"

PLE_PATCH_TARGET = "vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py"
PLE_STOCK_SHA256 = "a71144c1d36e06f22a2da1b1ada900076597fe5e824a911e7ada86249a0993e7"
PLE_PATCHED_SHA256 = "a8a064744efc3c99eefff649f50395d77e1c36085266034d17b51722f68176ed"
PLE_APPLICATOR_SHA256 = "a9f88c019d04d7c8804ca6532e1c689d2ea97cf085cfec37dd4a779cb2c2c3fd"
PLE_PATCH_DIFF_SHA256 = "e3b83f9a65e436d21d3e757ddc8dd2699f3c59de9f73967107b3b20c101d0a9f"
PATCH_DIR_LOCAL = Path(__file__).resolve().parent / "flashnext_patches"
PATCH_DIR_IMAGE = "/opt/arc3/vllm-patches"
PLE_APPLICATOR = "apply_radixark_nvfp4_ple_fp8_patch.py"

# --- the serve command (KEITH_REGIME.md §3) ---------------------------------
VLLM_HOST = "127.0.0.1"
VLLM_PORT = 1234
VLLM_MAX_MODEL_LEN = 32768
# The notebook's cell-3 profile (recorded for the log; the resolved argv below
# is what serving_setup.py derives from it).
PUBLIC25_VLLM_PROFILE_NAME = "kv5-bf16-mtp3-c8-cg32"
PUBLIC25_VLLM_PROFILE_ENV = {
    "TAAF_VLLM_ENABLE_PREFIX_CACHING": "0",
    "TAAF_VLLM_KV_CACHE_DTYPE": "auto",
    "TAAF_VLLM_KV_CACHE_MEMORY_BYTES": "5368709120",
    "TAAF_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE": "32",
    "TAAF_VLLM_MAX_NUM_BATCHED_TOKENS": "8192",
    "TAAF_VLLM_MAX_NUM_SEQS": "8",
    "TAAF_VLLM_MTP_TOKENS": "3",
    "TAAF_VLLM_OMP_THREADS": "1",
}
SPECULATIVE_CONFIG = json.dumps(
    {"method": "mtp", "num_speculative_tokens": 3}, separators=(",", ":"))


def vllm_cmd(model_dir: str) -> list[str]:
    """Keith's serving_setup.server_command(), token for token
    (vllm-server-identity.json argv); only <model_dir> is host-specific."""
    return [
        IMAGE_PYTHON, "-m", "vllm.entrypoints.cli.main", "serve", model_dir,
        "--served-model-name", SERVED_MODEL_NAME,
        "--host", VLLM_HOST,
        "--port", str(VLLM_PORT),
        "--load-format", "safetensors",
        "--dtype", "bfloat16",
        "--quantization", "modelopt_fp4",
        "--tensor-parallel-size", "1",
        "--distributed-executor-backend", "mp",
        "--kv-cache-memory-bytes", PUBLIC25_VLLM_PROFILE_ENV["TAAF_VLLM_KV_CACHE_MEMORY_BYTES"],
        "--max-model-len", str(VLLM_MAX_MODEL_LEN),
        "--max-num-seqs", PUBLIC25_VLLM_PROFILE_ENV["TAAF_VLLM_MAX_NUM_SEQS"],
        "--max-num-batched-tokens", PUBLIC25_VLLM_PROFILE_ENV["TAAF_VLLM_MAX_NUM_BATCHED_TOKENS"],
        "--async-scheduling",
        "--enable-chunked-prefill",
        "--max-cudagraph-capture-size", PUBLIC25_VLLM_PROFILE_ENV["TAAF_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE"],
        "--no-enable-prefix-caching",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_coder",
        "--reasoning-parser", "qwen3",
        "--chat-template", f"{model_dir}/chat_template.jinja",
        "--speculative-config", SPECULATIVE_CONFIG,
        "--no-enable-log-requests",
        "--disable-uvicorn-access-log",
        "--uvicorn-log-level", "info",
    ]


# --- the server process environment (KEITH_REGIME.md §4) --------------------
CACHE_DIR = "/cache"                                   # the Volume mount
HF_HOME = f"{CACHE_DIR}/huggingface"
CACHE_ROOT = f"{CACHE_DIR}/flashnext/cache"            # Keith: /tmp/qwen38-flash-next-vllm-cache
COMPILE_CACHE_ROOT = f"{CACHE_DIR}/flashnext/compile-cache"  # Keith: /tmp/...-compile-cache (fresh per run)
TMP_ROOT = "/tmp/qwen38-flash-next-vllm-tmp"           # identical path
SERVER_LOG = "/tmp/vllm-openai-server.log"
SERVER_READY_TIMEOUT = 1500                            # serving_setup.py:141

SERVER_ENV_EXACT = {
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUDA_VISIBLE_DEVICES": "0",
    "CUDA_HOME": CUDA_HOME,
    "CUDACXX": f"{CUDA_HOME}/bin/nvcc",
    "TMPDIR": TMP_ROOT,
    "VLLM_PLE_CPU_OFFLOAD": "1",
    "VLLM_PLE_OFFLOAD_READY_TIMEOUT": str(SERVER_READY_TIMEOUT),
    "VLLM_ENABLE_CUDA_COMPATIBILITY": "0",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "VLLM_NO_USAGE_STATS": "1",
    "DO_NOT_TRACK": "1",
    "VLLM_RADIXARK_QWEN38_NVFP4_PLE_FP8": "1",
    "VLLM_RADIXARK_QWEN38_NVFP4_CONFIG_SHA256": MODEL_CONFIG_SHA256,
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "TORCH_CUDA_ARCH_LIST": "12.0",
    "TOKENIZERS_PARALLELISM": "false",
    "OMP_NUM_THREADS": PUBLIC25_VLLM_PROFILE_ENV["TAAF_VLLM_OMP_THREADS"],
    "MKL_NUM_THREADS": PUBLIC25_VLLM_PROFILE_ENV["TAAF_VLLM_OMP_THREADS"],
}
SERVER_ENV_REMOVED = ["PYTORCH_ALLOC_CONF"]
SERVER_ENV_CACHE = {
    "HF_HOME": HF_HOME,
    "XDG_CACHE_HOME": CACHE_ROOT,
    "TORCH_HOME": f"{CACHE_ROOT}/torch",
    "TORCHINDUCTOR_CACHE_DIR": f"{COMPILE_CACHE_ROOT}/torchinductor",
    "TRITON_CACHE_DIR": f"{COMPILE_CACHE_ROOT}/triton",
    "CUDA_CACHE_PATH": f"{COMPILE_CACHE_ROOT}/cuda",
    "FLASHINFER_WORKSPACE_BASE": f"{COMPILE_CACHE_ROOT}/flashinfer",
}


def server_env(base: dict | None = None) -> dict[str, str]:
    """serving_setup.runtime_environment(): copy of the parent env, allocator
    alias removed, PYTHONPATH/PATH/LD_LIBRARY_PATH prepended in Keith's order
    (native image paths; is_dir()-filtered like his), then the exact keys."""
    env = dict(os.environ if base is None else base)
    for key in SERVER_ENV_REMOVED:
        env.pop(key, None)
    site = Path(SITE_PACKAGES)
    cuda_home = Path(CUDA_HOME)
    cutlass_nested = site / "nvidia_cutlass_dsl" / "dsl_packages"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(cutlass_nested), str(site)]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env["PATH"] = os.pathsep.join(
        [str(cuda_home / "bin"), "/usr/local/bin", env.get("PATH", "")])
    driver = _find_real_cuda_driver()
    library_dirs = [
        cuda_home / "targets" / "x86_64-linux" / "lib",
        cuda_home / "lib64",
        site / "torch" / "lib",
        *sorted(p for p in (site / "nvidia").glob("*/lib") if p.is_dir()),
        *([driver.parent] if driver else []),
        Path("/usr/local/nvidia/lib64"),
        Path("/usr/lib/x86_64-linux-gnu"),
    ]
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(p) for p in library_dirs if p.is_dir()]
        + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []))
    for key, path in SERVER_ENV_CACHE.items():
        Path(path).mkdir(parents=True, exist_ok=True)
        env[key] = path
    Path(TMP_ROOT).mkdir(parents=True, exist_ok=True)
    env.update(SERVER_ENV_EXACT)
    return env


def _find_real_cuda_driver() -> Path | None:
    """Where libcuda.so.1 resolves (Keith: ldconfig -p; Kaggle injects it at
    /usr/local/nvidia/lib64, Modal at /usr/lib/x86_64-linux-gnu)."""
    try:
        out = subprocess.run(["ldconfig", "-p"], capture_output=True,
                             text=True, check=True, timeout=30).stdout
    except Exception:  # noqa: BLE001
        return None
    for line in out.splitlines():
        if "libcuda.so.1" in line and "=>" in line:
            p = Path(line.split("=>", 1)[1].strip())
            if p.exists():
                return p.resolve()
    return None


# --- plumbing ---------------------------------------------------------------
PROXY_PORT = 8080                # the port Modal exposes
STARTUP_TIMEOUT_S = 45 * 60      # cold start: 126 GiB weights off the volume + PLE offload + compile
VLLM_READY_TIMEOUT_S = 40 * 60
PROXY_UPSTREAM_TIMEOUT_S = 3600

# --- cost guards ---------------------------------------------------------
IDLE_TIMEOUT_S = int(os.environ.get("ARC3_IDLE_TIMEOUT_S", "900"))            # 15 min
MAX_LIFETIME_S = int(os.environ.get("ARC3_MAX_LIFETIME_S", str(6 * 3600)))    # 6 h (two 2.2 h arms + cold start)
MAX_CONTAINERS = 1
MEMORY_MIB = 98304               # 96 GiB request (Keith's gate: >= 64 GiB available; 128 GiB starved scheduling on 09-03)
MIN_HOST_AVAILABLE_BYTES = 64 * 1024**3   # serving_setup.py:191
CPU_CORES = 8.0

AUTH_EXEMPT = {
    ("GET", "/v1/models"),
    ("GET", "/health"),
}
IDENTITY_PATH = "/arc3/identity"   # bearer-protected: startup log evidence
IDENTITY_PATTERNS = [
    r"version 0\.1\.dev",
    r"non-default args",
    r"Detected ModelOpt NVFP4",
    r"NvFp4 MoE backend",
    r"PleOffload: spawning",
    r"PLE offload matched",
    r"Model loading took",
    r"reserved .* for KV Cache",
    r"Maximum concurrency for",
    r"cudagraph_capture_sizes",
    r"Graph capturing finished",
    r"init engine",
    r"Starting vLLM server",
]
_identity: dict = {}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _snapshot_dir(local_only: bool) -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=HF_MODEL_REPO, revision=HF_MODEL_REVISION,
                             local_files_only=local_only, max_workers=8)


def _verify_model_dir(model_dir: str) -> None:
    cfg = Path(model_dir) / "config.json"
    tpl = Path(model_dir) / "chat_template.jinja"
    if not cfg.is_file() or not tpl.is_file():
        raise RuntimeError(f"snapshot at {model_dir} lacks config.json/chat_template.jinja")
    got = sha256_file(cfg)
    if got != MODEL_CONFIG_SHA256:
        raise RuntimeError(f"config.json sha256 {got} != pinned {MODEL_CONFIG_SHA256}")
    n = sum(1 for p in Path(model_dir).iterdir() if p.name.endswith(".safetensors"))
    print(f"[{APP_NAME}] model dir OK: {model_dir} ({n} safetensors shards, "
          f"config sha {got[:12]}...)", flush=True)


def _meminfo_available() -> int:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return -1


def _host_inventory() -> dict:
    inv = {"mem_available_bytes": _meminfo_available(),
           "cpu_count": os.cpu_count()}
    try:
        st = os.statvfs("/dev/shm")
        inv["shm_free_bytes"] = st.f_bavail * st.f_frsize
    except OSError:
        inv["shm_free_bytes"] = -1
    for cmd in (["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version,compute_cap",
                 "--format=csv,noheader"],):
        try:
            inv["gpu_rows"] = subprocess.run(cmd, capture_output=True, text=True,
                                             timeout=30).stdout.strip().splitlines()
        except Exception as e:  # noqa: BLE001
            inv["gpu_rows"] = [f"nvidia-smi unavailable: {e!r}"]
    return inv


def _runtime_identity() -> dict:
    """What Keith's setup verifies about the runtime, read from the image."""
    ple = Path(SITE_PACKAGES) / PLE_PATCH_TARGET
    ident = {
        "python": sys.executable,
        "image_python": IMAGE_PYTHON,
        "ple_layer_sha256": sha256_file(ple) if ple.is_file() else None,
        "ple_layer_patched": (sha256_file(ple) == PLE_PATCHED_SHA256) if ple.is_file() else False,
        "nvcc": shutil.which("nvcc", path=f"{CUDA_HOME}/bin"),
        "cutlass_pth": (Path(SITE_PACKAGES) / "nvidia_cutlass_dsl_packages.pth").is_file(),
    }
    code = ("import json,vllm,torch,transformers,flashinfer,triton;"
            "print(json.dumps({'vllm':vllm.__version__,'torch':torch.__version__,"
            "'transformers':transformers.__version__,'flashinfer':flashinfer.__version__,"
            "'triton':triton.__version__,'cuda':torch.version.cuda}))")
    try:
        out = subprocess.run([IMAGE_PYTHON, "-c", code], capture_output=True, text=True,
                             timeout=600, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        ident["versions"] = json.loads(out.stdout.strip().splitlines()[-1]) if out.returncode == 0 else None
        if out.returncode != 0:
            ident["versions_error"] = out.stderr[-2000:]
    except Exception as e:  # noqa: BLE001
        ident["versions_error"] = repr(e)
    return ident


def _tee_and_wait(process: "subprocess.Popen", log_path: str) -> threading.Thread:
    def _pump() -> None:
        with open(log_path, "a", encoding="utf-8") as fh:
            for line in process.stdout:  # type: ignore[union-attr]
                sys.stdout.write(line)
                fh.write(line)
                fh.flush()
    t = threading.Thread(target=_pump, daemon=True)
    t.start()
    return t


def _wait_for_vllm(timeout_s: float, process: "subprocess.Popen") -> float:
    """serving_setup.wait_for_server(): poll /v1/models until it serves
    EXACTLY [SERVED_MODEL_NAME]; die if the process dies first."""
    url = f"http://{VLLM_HOST}:{VLLM_PORT}/v1/models"
    t0 = time.monotonic()
    deadline = t0 + timeout_s
    last_notice = -1
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited rc={process.returncode} before ready")
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                ids = [m.get("id") for m in json.loads(r.read()).get("data", [])]
            if ids != [SERVED_MODEL_NAME]:
                raise RuntimeError(f"vLLM served the wrong model identity: {ids}")
            ready = time.monotonic() - t0
            print(f"[{APP_NAME}] vLLM ready in {ready:.0f}s, serving {ids}", flush=True)
            return ready
        except (urllib.error.URLError, OSError, TimeoutError, ValueError):
            elapsed = int(time.monotonic() - t0)
            if elapsed // 60 != last_notice:
                last_notice = elapsed // 60
                print(f"[{APP_NAME}] VLLM_WAIT elapsed_s={elapsed}", flush=True)
            time.sleep(5)
    raise TimeoutError(f"vLLM not ready after {timeout_s}s")


def _grep_log(log_path: str) -> list[str]:
    try:
        lines = Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    pats = [re.compile(p) for p in IDENTITY_PATTERNS]
    return [ln for ln in lines if any(p.search(ln) for p in pats)][:60]


def _start_lifetime_watchdog() -> None:
    cap = int(os.environ.get("ARC3_MAX_LIFETIME_S", str(MAX_LIFETIME_S)))

    def _reaper() -> None:
        time.sleep(cap)
        print(f"[{APP_NAME}] MAX LIFETIME {cap}s reached — exiting container "
              "(cost guard; Modal cold-starts a fresh one on the next request)", flush=True)
        os._exit(0)

    threading.Thread(target=_reaper, daemon=True).start()


def _run_auth_proxy() -> None:
    """Bearer-token reverse proxy in front of vLLM (stdlib only; the harness
    never streams). Adds GET /arc3/identity = the startup-log evidence."""
    import http.server

    token = os.environ.get("TOKEN", "").strip()

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quiet; Modal captures stdout
            pass

        def _reply(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _handle(self) -> None:
            path_only = self.path.split("?", 1)[0]
            if (self.command, path_only) not in AUTH_EXEMPT:
                if not token:  # fail CLOSED if the secret is missing
                    self._reply(500, b'{"error":"server has no TOKEN secret"}')
                    return
                if self.headers.get("Authorization", "") != f"Bearer {token}":
                    self._reply(401, b'{"error":"unauthorized"}')
                    return
            if self.command == "GET" and path_only == IDENTITY_PATH:
                body = dict(_identity)
                body["log_evidence"] = _grep_log(SERVER_LOG)
                self._reply(200, json.dumps(body, indent=1).encode())
                return
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else None
            req = urllib.request.Request(
                f"http://{VLLM_HOST}:{VLLM_PORT}{self.path}", data=body, method=self.command)
            for h in ("Content-Type", "Accept"):
                if self.headers.get(h):
                    req.add_header(h, self.headers[h])
            try:
                with urllib.request.urlopen(req, timeout=PROXY_UPSTREAM_TIMEOUT_S) as r:
                    data = r.read()
                    self._reply(r.status, data, r.headers.get("Content-Type", "application/json"))
            except urllib.error.HTTPError as e:  # forward vLLM errors verbatim
                self._reply(e.code, e.read() or b"{}")
            except Exception as e:  # noqa: BLE001
                self._reply(502, json.dumps({"error": f"upstream vLLM unreachable: {e!r}"}).encode())

        do_GET = _handle    # noqa: N815
        do_POST = _handle   # noqa: N815

    srv = http.server.ThreadingHTTPServer(("0.0.0.0", PROXY_PORT), Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[{APP_NAME}] auth proxy on :{PROXY_PORT} (exempt: {sorted(AUTH_EXEMPT)}; "
          f"identity at GET {IDENTITY_PATH})", flush=True)


# Container-level env (record only; the vLLM process env is built by server_env()).
_CONTAINER_ENV = {
    **PUBLIC25_VLLM_PROFILE_ENV,
    "PUBLIC25_VLLM_PROFILE_NAME": PUBLIC25_VLLM_PROFILE_NAME,
    "HF_HOME": HF_HOME,
    "ARC3_MAX_LIFETIME_S": str(MAX_LIFETIME_S),
}


if modal is not None:
    app = modal.App(APP_NAME)
    hf_cache_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

    # THE runtime: Keith's docker image, by amd64 manifest digest, with his PLE
    # patch applied by his applicator (stock-hash gated, result-hash verified).
    # `.entrypoint([])` clears the image's `vllm serve` ENTRYPOINT so Modal can
    # run our function. Modal (builder >= 2025.06) installs NOTHING into the
    # image — its client is mounted at runtime — but it needs a `python` on
    # PATH to detect the version; the image only ships `python3`, hence the
    # symlink (no second interpreter: `add_python` is deliberately NOT used).
    runtime_image = (
        modal.Image.from_registry(
            VLLM_IMAGE_REF,
            setup_dockerfile_commands=[
                'RUN command -v python >/dev/null 2>&1 || ln -s "$(command -v python3)" /usr/local/bin/python',
            ])
        .entrypoint([])
        .add_local_dir(str(PATCH_DIR_LOCAL), PATCH_DIR_IMAGE, copy=True)
        .run_commands(
            f"sha256sum {SITE_PACKAGES}/{PLE_PATCH_TARGET}",
            f"{IMAGE_PYTHON} -B {PATCH_DIR_IMAGE}/{PLE_APPLICATOR} {SITE_PACKAGES}",
            f"sha256sum {SITE_PACKAGES}/{PLE_PATCH_TARGET} | grep -q {PLE_PATCHED_SHA256}",
            f"find {SITE_PACKAGES}/vllm/models/qwen3_8_flash_next -name '__pycache__' -prune -exec rm -rf {{}} +",
        )
        .env(_CONTAINER_ENV)
    )

    # A slim CPU image for the download only — keeps the runtime image pristine
    # (no hf_transfer/hf_xet added to Keith's site-packages).
    warm_image = (
        modal.Image.debian_slim(python_version="3.12")
        .uv_pip_install("huggingface_hub[hf_transfer,hf_xet]")
        .env({"HF_HOME": HF_HOME, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    )

    @app.function(image=warm_image, volumes={CACHE_DIR: hf_cache_vol},
                  cpu=4.0, memory=16384, timeout=6 * 3600)
    def warm() -> str:
        """CPU-only: snapshot the pinned revision (~135 GB) into the Volume
        once, so no GPU minute is ever spent downloading."""
        t0 = time.time()
        path = _snapshot_dir(local_only=False)
        _verify_model_dir(path)
        total = sum(p.stat().st_size for p in Path(path).iterdir() if p.is_file() or p.is_symlink())
        print(f"[{APP_NAME}] snapshot ready at {path}: {total:,} bytes "
              f"(pinned {MODEL_TOTAL_BYTES:,}) in {time.time() - t0:.0f}s; committing volume",
              flush=True)
        hf_cache_vol.commit()
        print(f"[{APP_NAME}] volume committed ({time.time() - t0:.0f}s total)", flush=True)
        return path

    @app.function(image=runtime_image, volumes={CACHE_DIR: hf_cache_vol},
                  cpu=CPU_CORES, memory=MEMORY_MIB, timeout=1800)
    def probe() -> dict:
        """CPU-only identity check of the built runtime image (no GPU): vLLM/
        torch/flashinfer versions, the patched PLE hash, nvcc, host RAM and
        /dev/shm at the serve-time resource request, snapshot presence."""
        out = {"runtime": _runtime_identity(), "host": _host_inventory(),
               "image_env": {k: os.environ.get(k) for k in
                             ("PATH", "LD_LIBRARY_PATH", "CUDA_HOME", "PYTHONPATH")}}
        try:
            path = _snapshot_dir(local_only=True)
            _verify_model_dir(path)
            out["snapshot"] = path
        except Exception as e:  # noqa: BLE001
            out["snapshot"] = f"MISSING ({e!r}) — run ::warm first"
        out["vllm_cmd"] = vllm_cmd("<MODEL_DIR>")
        env = server_env()
        out["server_env_exact_applied"] = {k: env.get(k) for k in SERVER_ENV_EXACT}
        out["server_env_paths"] = {k: env.get(k) for k in ("PYTHONPATH", "PATH", "LD_LIBRARY_PATH")}
        print(json.dumps(out, indent=1), flush=True)
        return out

    @app.function(
        image=runtime_image,
        gpu=[f"{GPU_KIND}:{N_GPU}", *[f"{g}:{N_GPU}" for g in GPU_FALLBACK]],
        cpu=CPU_CORES,
        memory=MEMORY_MIB,
        volumes={CACHE_DIR: hf_cache_vol},
        secrets=[modal.Secret.from_name(SECRET_NAME)],
        scaledown_window=IDLE_TIMEOUT_S,   # scale to zero after 15 min idle
        max_containers=MAX_CONTAINERS,     # never a surprise second GPU
        timeout=STARTUP_TIMEOUT_S,
    )
    @modal.concurrent(max_inputs=64)       # one replica takes the whole wave
    @modal.web_server(port=PROXY_PORT, startup_timeout=STARTUP_TIMEOUT_S)
    def serve() -> None:
        t_start = time.time()
        inv = _host_inventory()
        print(f"[{APP_NAME}] host inventory: {json.dumps(inv)}", flush=True)
        if inv["mem_available_bytes"] < MIN_HOST_AVAILABLE_BYTES:
            raise RuntimeError(
                f"Host memory too small for FP8 PLE CPU offload: "
                f"{inv['mem_available_bytes']} < {MIN_HOST_AVAILABLE_BYTES}")
        ident = _runtime_identity()
        print(f"[{APP_NAME}] runtime identity: {json.dumps(ident)}", flush=True)
        if not ident["ple_layer_patched"]:
            raise RuntimeError(f"PLE patch not in place: {ident['ple_layer_sha256']}")
        if (ident.get("versions") or {}).get("vllm") != VLLM_VERSION:
            raise RuntimeError(f"vLLM version drift: {ident.get('versions')} != {VLLM_VERSION}")
        try:
            model_dir = _snapshot_dir(local_only=True)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError("model snapshot not in the volume — run "
                               f"`modal run offkaggle/modal_flashnext_serve.py::warm` first ({e!r})")
        _verify_model_dir(model_dir)

        cmd = vllm_cmd(model_dir)
        env = server_env()
        Path(SERVER_LOG).unlink(missing_ok=True)
        print(f"[{APP_NAME}] VLLM_START_COMMAND {json.dumps(cmd)}", flush=True)
        print(f"[{APP_NAME}] PUBLIC25_VLLM_PROFILE name={PUBLIC25_VLLM_PROFILE_NAME} "
              f"env={json.dumps(PUBLIC25_VLLM_PROFILE_ENV, sort_keys=True)}", flush=True)
        process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        _tee_and_wait(process, SERVER_LOG)
        ready_s = _wait_for_vllm(VLLM_READY_TIMEOUT_S, process)
        _identity.update({
            "app": APP_NAME, "gpu": GPU_KIND, "served_model_name": SERVED_MODEL_NAME,
            "hf_repo": HF_MODEL_REPO, "hf_revision": HF_MODEL_REVISION,
            "vllm_version_pinned": VLLM_VERSION, "image": VLLM_IMAGE,
            "image_amd64_digest": VLLM_IMAGE_AMD64_DIGEST,
            "argv": cmd, "profile": PUBLIC25_VLLM_PROFILE_NAME,
            "profile_env": PUBLIC25_VLLM_PROFILE_ENV,
            "runtime": ident, "host": inv,
            "ready_seconds": ready_s, "cold_start_seconds": time.time() - t_start,
            "started_epoch": t_start,
        })
        for ln in _grep_log(SERVER_LOG):
            if "Maximum concurrency" in ln:
                print(f"[{APP_NAME}] {ln.strip()}", flush=True)
        try:
            hf_cache_vol.commit()   # persist compile caches for the next cold start
        except Exception as e:  # noqa: BLE001
            print(f"[{APP_NAME}] volume commit skipped: {e!r}", flush=True)
        _start_lifetime_watchdog()
        _run_auth_proxy()           # the port opens only once vLLM is ready

    @app.local_entrypoint()
    def smoke(url: str = "", token: str = "", max_tokens: int = 512) -> None:
        """GET /v1/models without and with the token, a 401 with a wrong token,
        the server's startup evidence (max-concurrency line) via /arc3/identity,
        then ONE thinking-enabled chat completion at the harness's sampling
        settings, reporting tokens/s. First hit cold-starts the GPU (~10-20 min)."""
        base = (url or serve.get_web_url()).rstrip("/")
        token = token or os.environ.get("ARC3_VLLM_TOKEN", "")
        if not token:
            tok_file = Path.home() / ".config" / "arc3" / "vllm_token"
            if tok_file.is_file():
                token = tok_file.read_text().strip()
        if not token:
            raise SystemExit("pass --token, set ARC3_VLLM_TOKEN, or create ~/.config/arc3/vllm_token")
        auth = {"Authorization": f"Bearer {token}"}

        def get(path: str, headers: dict | None = None, timeout: int = STARTUP_TIMEOUT_S):
            req = urllib.request.Request(f"{base}{path}", headers=headers or {})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, json.loads(r.read())

        print(f"[smoke] GET {base}/v1/models (no auth — exempt; cold start may take 10-20 min)")
        t0 = time.time()
        status, models = get("/v1/models")
        ids = [m.get("id") for m in models.get("data", [])]
        assert status == 200 and ids == [SERVED_MODEL_NAME], f"served {ids}, want [{SERVED_MODEL_NAME!r}]"
        print(f"[smoke] models OK (no auth): {ids}; max_model_len="
              f"{models['data'][0].get('max_model_len')}; {time.time() - t0:.0f}s")
        status, models = get("/v1/models", auth, 120)
        assert status == 200, status
        print("[smoke] models OK (with token)")

        payload = {
            "model": SERVED_MODEL_NAME,
            "messages": [{"role": "user", "content":
                          "Think briefly, then answer in one short sentence: "
                          "what is the smallest prime greater than 100?"}],
            "temperature": 0.6, "top_p": 0.95, "top_k": 20,      # LOCAL_ANALYZER_* (KEITH_REGIME §5)
            "max_tokens": int(max_tokens),
            "chat_template_kwargs": {"enable_thinking": True},
        }
        body = json.dumps(payload).encode()
        bad = urllib.request.Request(f"{base}/v1/chat/completions", data=body, method="POST",
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer wrong-token"})
        try:
            urllib.request.urlopen(bad, timeout=60)
            raise SystemExit("SMOKE FAIL: bad token was ACCEPTED — auth is off")
        except urllib.error.HTTPError as e:
            assert e.code == 401, f"bad token got {e.code}, expected 401"
            print("[smoke] bad token rejected with 401: auth is ON")

        status, ident = get(IDENTITY_PATH, auth, 120)
        conc = [ln for ln in ident.get("log_evidence", []) if "Maximum concurrency" in ln]
        print(f"[smoke] server identity: vllm={ (ident.get('runtime') or {}).get('versions') } "
              f"ready_s={ident.get('ready_seconds')} cold_start_s={ident.get('cold_start_seconds')} "
              f"gpu={ (ident.get('host') or {}).get('gpu_rows') }")
        print(f"[smoke] max-concurrency line: {conc[0].strip() if conc else 'NOT FOUND in log'}")
        for ln in ident.get("log_evidence", []):
            if any(k in ln for k in ("non-default args", "reserved", "NvFp4 MoE", "PLE offload matched",
                                     "Model loading took", "cudagraph_capture_sizes")):
                print(f"[smoke]   {ln.strip()[:300]}")

        good = urllib.request.Request(f"{base}/v1/chat/completions", data=body, method="POST",
                                      headers={"Content-Type": "application/json", **auth})
        t0 = time.time()
        with urllib.request.urlopen(good, timeout=900) as r:
            resp = json.loads(r.read())
        dt = time.time() - t0
        msg = resp["choices"][0]["message"]
        usage = resp.get("usage") or {}
        ctoks = int(usage.get("completion_tokens") or 0)
        reasoning = (msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
        print(f"[smoke] completion: {ctoks} completion tokens in {dt:.1f}s = {ctoks / dt:.1f} tok/s "
              f"(prompt {usage.get('prompt_tokens')}); finish={resp['choices'][0].get('finish_reason')}")
        print(f"[smoke] reasoning[{len(reasoning)} chars]: {reasoning[:200]!r}")
        print(f"[smoke] content: {(msg.get('content') or '').strip()[:300]!r}")
        assert ctoks > 0, "no completion tokens"
        assert reasoning, "thinking was requested but no reasoning_content came back"
        print("[smoke] PASS")
