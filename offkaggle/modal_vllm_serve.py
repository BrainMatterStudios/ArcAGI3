# =============================================================================
# Off-Kaggle certification rig — exact-fidelity vLLM server on Modal.
#
# Replicates the SCORED eval serving config byte-for-byte on the semantic
# surface. Source of truth: `git show HEAD:scratchpad/taaf_scored_ref/
# setup_commands.json` (the scored kernel's setup command). Every constant
# below that mirrors that blob is pinned by offkaggle/test_offkaggle.py, which
# parses the git blob and diffs it against this module — drift fails the test.
#
# What the scored config does (and this file reproduces):
#   * model: local HF snapshot of vrfai/Qwen3.6-27B-FP8, served under the
#     alias 'vrfai/Qwen3.6-27B-FP8' (Kaggle mirror dataset:
#     driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot). HF repo id VERIFIED LIVE
#     2026-08-10: GET https://huggingface.co/api/models/vrfai/Qwen3.6-27B-FP8
#     -> 200, public, fp8/w8a8 compressed-tensors, base Qwen/Qwen3.6-27B.
#   * pins: vllm==0.19.0 torch==2.10.0 flashinfer==0.6.6 (the wheelhouse
#     STAMP_TEXT). PyPI ships flashinfer as 'flashinfer-python'.
#   * serve flags: --enable-auto-tool-choice --tool-call-parser qwen3_coder
#     --generation-config vllm --enable-prefix-caching
#     --default-chat-template-kwargs '{"preserve_thinking": true}'
#     --reasoning-parser qwen3 --max-model-len 65536 --tensor-parallel-size 1
#   * entrypoint: python -m vllm.entrypoints.openai.api_server (NOT `vllm
#     serve` — same module the scored kernel launches).
#
# Auth: a bearer token from the Modal secret 'arc3-vllm-token' guards the
# endpoint via a tiny stdlib reverse proxy in front of vLLM. Create it once:
#     modal secret create arc3-vllm-token TOKEN=<random>
# GET /v1/models and GET /health are DELIBERATELY exempt from auth:
# pc_driver._pc_serving_probe (submission/_ab_patch_closure/pc_driver.py:288)
# hits /models with a bare urllib request carrying NO Authorization header,
# and that file must stay byte-identical to the rig. Everything else
# (chat/completions included) requires `Authorization: Bearer <TOKEN>` — which
# the duck harness sends natively when LOCAL_ANALYZER_API_KEY is set
# (tool_agent.py:959-973 -> openai_compat.build_headers).
#
# Cost guards: scale-to-zero after IDLE_TIMEOUT_S with no requests (Modal
# scaledown_window), plus an in-container hard lifetime cap MAX_LIFETIME_S
# enforced by a watchdog thread (Modal has no per-container lifetime knob for
# web servers; the watchdog exits the process, Modal cold-starts on the next
# request). max_containers=1 so a burst can never spin up a second H100.
#
# Deploy / run (requires `pip install modal` + `modal setup`; see README.md):
#     modal run    offkaggle/modal_vllm_serve.py::warm    # CPU-only snapshot warm
#     modal deploy offkaggle/modal_vllm_serve.py          # prints the web URL
#     modal run    offkaggle/modal_vllm_serve.py::smoke --url https://...
# =============================================================================
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

try:  # modal is only needed to deploy/run; test_offkaggle.py imports this
    import modal  # noqa: F401  # module on a host without the modal client.
except ModuleNotFoundError:  # pragma: no cover - exercised on bare hosts
    modal = None

APP_NAME = "arc3-vllm"
VOLUME_NAME = "arc3-hf-cache"
SECRET_NAME = "arc3-vllm-token"  # modal secret create arc3-vllm-token TOKEN=<random>

# --- GPU CHOICE -------------------------------------------------------------
# The scored eval runs on a Kaggle RTX Pro 6000 (Blackwell, 96 GB, native FP8
# tensor cores). The model is FP8 W8A8 compressed-tensors, so the replica GPU
# MUST have hardware FP8 or vLLM falls back to different kernels and the
# numerics drift. On Modal the cheapest GPU with native FP8 and enough memory
# is the H100 (Hopper, 80 GB): A100 is Ampere (NO FP8 units), L40S is Ada with
# FP8 but only 48 GB (~30 GB weights + 65536-ctx KV for 28 clones does not
# fit), H200/B200 cost more for no fidelity gain. Known residual gap, noted in
# the pre-registration: 80 GB vs 96 GB means a smaller KV cache (more
# preemption under load) and SM90-vs-SM120 kernel selection — throughput and
# bitwise numerics are NOT identical, the serving semantics are.
GPU_KIND = "H100"
N_GPU = 1

# --- the scored serving contract (test-pinned against the HEAD blob) --------
VLLM_VERSION = "0.19.0"
TORCH_VERSION = "2.10.0"
FLASHINFER_VERSION = "0.6.6"
HF_MODEL_REPO = "vrfai/Qwen3.6-27B-FP8"
SERVED_MODEL_NAME = "vrfai/Qwen3.6-27B-FP8"
VLLM_MAX_MODEL_LEN = 65536
VLLM_TENSOR_PARALLEL_SIZE = 1
# The exact non-plumbing serve flags from the scored setup command, in the
# scored order. Host/port/model-path are plumbing and differ; these are the
# semantics.
VLLM_EVAL_FLAGS = [
    "--enable-auto-tool-choice",
    "--tool-call-parser", "qwen3_coder",
    "--generation-config", "vllm",
    "--enable-prefix-caching",
    "--default-chat-template-kwargs", '{"preserve_thinking": true}',
    "--reasoning-parser", "qwen3",
    "--max-model-len", str(VLLM_MAX_MODEL_LEN),
]

# --- plumbing ---------------------------------------------------------------
VLLM_HOST = "127.0.0.1"   # vLLM binds loopback; only the auth proxy is public
VLLM_PORT = 8000
PROXY_PORT = 8080          # the port Modal exposes
CACHE_DIR = "/cache"       # the modal.Volume mount (HF snapshot ~30 GB)
STARTUP_TIMEOUT_S = 45 * 60   # cold start = 30 GB download + FP8 load
VLLM_READY_TIMEOUT_S = 40 * 60
PROXY_UPSTREAM_TIMEOUT_S = 3600  # per-request read timeout proxy -> vLLM

# --- cost guards (deploy-time env overrides; defaults are the contract) -----
IDLE_TIMEOUT_S = int(os.environ.get("ARC3_IDLE_TIMEOUT_S", "600"))       # 10 min
MAX_LIFETIME_S = int(os.environ.get("ARC3_MAX_LIFETIME_S", str(6 * 3600)))  # 6 h

# Auth exemptions: (method, exact path). GET /v1/models MUST stay exempt —
# see the header comment (pc_driver's serving probe sends no auth header).
AUTH_EXEMPT = {
    ("GET", "/v1/models"),
    ("GET", "/health"),
}


def vllm_cmd(model_path: str) -> list[str]:
    """The exact scored launch: module entrypoint + local snapshot path +
    served alias + the pinned semantic flags."""
    return [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--served-model-name", SERVED_MODEL_NAME,
        "--host", VLLM_HOST,
        "--port", str(VLLM_PORT),
        "--tensor-parallel-size", str(VLLM_TENSOR_PARALLEL_SIZE),
        *VLLM_EVAL_FLAGS,
    ]


def _download_snapshot() -> str:
    """Snapshot into the volume-backed HF cache (idempotent; repeat launches
    hit the cache and skip the ~30 GB download)."""
    from huggingface_hub import snapshot_download

    t0 = time.time()
    path = snapshot_download(repo_id=HF_MODEL_REPO)
    print(f"[arc3-vllm] snapshot ready at {path} ({round(time.time() - t0, 1)}s)",
          flush=True)
    return path


def _wait_for_vllm(timeout_s: float, process: "subprocess.Popen") -> None:
    url = f"http://{VLLM_HOST}:{VLLM_PORT}/v1/models"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"vLLM server exited with rc={process.returncode} before ready")
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                ids = [m.get("id") for m in json.loads(r.read()).get("data", [])]
            if SERVED_MODEL_NAME in ids:
                print(f"[arc3-vllm] vLLM ready, serving {ids}", flush=True)
                return
            raise RuntimeError(f"vLLM serves {ids}, expected {SERVED_MODEL_NAME!r}")
        except (urllib.error.URLError, OSError, TimeoutError):
            time.sleep(5)
    raise TimeoutError(f"vLLM not ready after {timeout_s}s")


def _start_lifetime_watchdog() -> None:
    """Hard lifetime cap: after MAX_LIFETIME_S the container exits no matter
    what, so a wedged session cannot silently burn credits."""
    cap = int(os.environ.get("ARC3_MAX_LIFETIME_S", str(MAX_LIFETIME_S)))

    def _reaper() -> None:
        time.sleep(cap)
        print(f"[arc3-vllm] MAX LIFETIME {cap}s reached — exiting container "
              "(cost guard; Modal cold-starts a fresh one on the next request)",
              flush=True)
        os._exit(0)

    threading.Thread(target=_reaper, daemon=True).start()


def _run_auth_proxy() -> None:
    """Bearer-token reverse proxy in front of vLLM, stdlib only.

    The harness never streams (build_chat_payload pins "stream": False), so a
    plain buffered forward is faithful. Threaded server: 28 concurrent
    long-poll analyzer calls is the working regime.
    """
    import http.server

    token = os.environ.get("TOKEN", "").strip()

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quiet; Modal captures stdout
            pass

        def _reply(self, code: int, body: bytes,
                   ctype: str = "application/json") -> None:
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
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else None
            req = urllib.request.Request(
                f"http://{VLLM_HOST}:{VLLM_PORT}{self.path}",
                data=body, method=self.command)
            for h in ("Content-Type", "Accept"):
                if self.headers.get(h):
                    req.add_header(h, self.headers[h])
            try:
                with urllib.request.urlopen(
                        req, timeout=PROXY_UPSTREAM_TIMEOUT_S) as r:
                    data = r.read()
                    self._reply(r.status, data,
                                r.headers.get("Content-Type", "application/json"))
            except urllib.error.HTTPError as e:  # forward vLLM errors verbatim
                self._reply(e.code, e.read() or b"{}")
            except Exception as e:  # noqa: BLE001
                self._reply(502, json.dumps(
                    {"error": f"upstream vLLM unreachable: {e!r}"}).encode())

        do_GET = _handle    # noqa: N815
        do_POST = _handle   # noqa: N815

    srv = http.server.ThreadingHTTPServer(("0.0.0.0", PROXY_PORT), Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[arc3-vllm] auth proxy on :{PROXY_PORT} "
          f"(exempt: {sorted(AUTH_EXEMPT)})", flush=True)


# Env the scored setup exports around vLLM (vllm_env() in the blob), plus HF
# cache redirection into the volume.
_CONTAINER_ENV = {
    "USE_TF": "0",
    "TRANSFORMERS_NO_TF": "1",
    "TRANSFORMERS_NO_TORCHVISION": "1",
    "VLLM_NO_USAGE_STATS": "1",
    "HF_HOME": f"{CACHE_DIR}/huggingface",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "ARC3_MAX_LIFETIME_S": str(MAX_LIFETIME_S),
}


if modal is not None:
    app = modal.App(APP_NAME)
    hf_cache_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

    image = (
        modal.Image.debian_slim(python_version="3.12")
        .uv_pip_install(
            f"vllm=={VLLM_VERSION}",
            f"torch=={TORCH_VERSION}",
            f"flashinfer-python=={FLASHINFER_VERSION}",
            "huggingface_hub[hf_transfer]",
        )
        .env(_CONTAINER_ENV)
    )

    @app.function(image=image, volumes={CACHE_DIR: hf_cache_vol},
                  timeout=3600)
    def warm() -> str:
        """CPU-only snapshot warm — pay the ~30 GB download once WITHOUT an
        H100 attached. Run before the first serve."""
        path = _download_snapshot()
        hf_cache_vol.commit()
        return path

    @app.function(
        image=image,
        gpu=f"{GPU_KIND}:{N_GPU}",
        volumes={CACHE_DIR: hf_cache_vol},
        secrets=[modal.Secret.from_name(SECRET_NAME)],
        scaledown_window=IDLE_TIMEOUT_S,   # scale to zero after idle
        max_containers=1,                  # never a surprise second H100
    )
    @modal.concurrent(max_inputs=64)       # one replica takes the whole wave
    @modal.web_server(port=PROXY_PORT, startup_timeout=STARTUP_TIMEOUT_S)
    def serve() -> None:
        model_path = _download_snapshot()
        hf_cache_vol.commit()
        cmd = vllm_cmd(model_path)
        print("[arc3-vllm] starting:", " ".join(cmd), flush=True)
        process = subprocess.Popen(cmd)   # logs inherit -> Modal log stream
        _wait_for_vllm(VLLM_READY_TIMEOUT_S, process)
        _start_lifetime_watchdog()
        _run_auth_proxy()                 # port opens only once vLLM is ready

    @app.local_entrypoint()
    def smoke(url: str = "", token: str = "") -> None:
        """Hit /v1/models (unauthenticated, mirrors pc_driver's probe), prove
        auth rejects a bad bearer, then run the scored kernel's exact smoke
        chat completion with the real token."""
        base = (url or serve.get_web_url()).rstrip("/")
        token = token or os.environ.get("ARC3_VLLM_TOKEN", "")
        if not token:
            raise SystemExit("pass --token or set ARC3_VLLM_TOKEN "
                             "(the value from the arc3-vllm-token secret)")

        print(f"[smoke] GET {base}/v1/models (no auth — must be exempt)")
        with urllib.request.urlopen(f"{base}/v1/models", timeout=STARTUP_TIMEOUT_S) as r:
            ids = [m.get("id") for m in json.loads(r.read()).get("data", [])]
        assert SERVED_MODEL_NAME in ids, f"served {ids}, want {SERVED_MODEL_NAME!r}"
        print(f"[smoke] models OK: {ids}")

        payload = json.dumps({   # byte-for-byte the scored kernel's smoke test
            "model": SERVED_MODEL_NAME,
            "messages": [{"role": "user",
                          "content": "Answer in one short sentence: what is 2 + 2?"}],
            "temperature": 0.0,
            "max_tokens": 96,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode()

        bad = urllib.request.Request(
            f"{base}/v1/chat/completions", data=payload, method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer wrong-token"})
        try:
            urllib.request.urlopen(bad, timeout=60)
            raise SystemExit("SMOKE FAIL: bad token was ACCEPTED — auth is off")
        except urllib.error.HTTPError as e:
            assert e.code == 401, f"bad token got {e.code}, expected 401"
            print("[smoke] bad token rejected with 401: auth is ON")

        good = urllib.request.Request(
            f"{base}/v1/chat/completions", data=payload, method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(good, timeout=600) as r:
            content = json.loads(r.read())["choices"][0]["message"].get("content", "")
        print(f"[smoke] chat completion OK: {content.strip()!r}")
        print("[smoke] PASS")
