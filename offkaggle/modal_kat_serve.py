# =============================================================================
# KAT-Coder-V2.5-Dev probe serve — Track-1 final falsifier (2026-08-15).
#
# Variant of offkaggle/modal_vllm_serve.py (which stays byte-untouched: it is
# the scored-config fidelity rig). This file serves the Track-1 KAT probe:
#   model : cyankiwi/KAT-Coder-V2.5-Dev-AWQ-INT4 (24.4 GB, compressed-tensors
#           pack-quantized INT4, arch Qwen3_5MoeForConditionalGeneration ->
#           vLLM model_type qwen3_5_moe — same arch family as the 35B-A3B).
#   flags : identical semantic surface to the scored rig (qwen3_coder tool
#           parser, qwen3 reasoning parser, generation-config vllm, prefix
#           caching, preserve_thinking, max-model-len 65536) PLUS
#           --language-model-only, which the KAT card marks REQUIRED on vLLM:
#           the open-weight release ships no vision tower, and without the
#           flag vLLM tries to initialize the missing visual weights and dies.
#           (Flag verified present in vLLM v0.19.0: vllm/config/model.py:312.)
#   chat template: verified to honor BOTH kwargs our stack sends —
#           preserve_thinking (line 100) and enable_thinking (line 149).
#   sampling: KAT's card HAS an opinion (generation_config: temp 1.0,
#           top_p 0.95, top_k 20; SWE evals ran temp 1.0 / top_p 0.95). We
#           keep --generation-config vllm for rig parity and inject the card
#           sampling per-request from the harness (TYCHO_EVAL_KAT_SAMPLING).
#
# Same volume (arc3-hf-cache) and same auth secret (arc3-vllm-token) as the
# main rig. Deploy / run:
#     modal run    offkaggle/modal_kat_serve.py::warm     # CPU-only download
#     modal deploy offkaggle/modal_kat_serve.py           # prints web URL
#     modal run    offkaggle/modal_kat_serve.py::smoke --url https://...
#     modal app stop arc3-kat-serve                       # when done
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

try:
    import modal  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    modal = None

APP_NAME = "arc3-kat-serve"
VOLUME_NAME = "arc3-hf-cache"       # SAME volume as arc3-vllm (shared HF cache)
SECRET_NAME = "arc3-vllm-token"     # SAME bearer token secret

GPU_KIND = "H100"
N_GPU = 1

VLLM_VERSION = "0.19.0"
TORCH_VERSION = "2.10.0"
FLASHINFER_VERSION = "0.6.6"
# QUANT SWAP 2026-08-15 (measured, not speculative): the planned
# cyankiwi/KAT-Coder-V2.5-Dev-AWQ-INT4 quant FAILED to load on the pinned
# vLLM 0.19.0 — its pack-quantized weights are ASYMMETRIC ("symmetric": false)
# and vLLM's compressed-tensors MoE path asserts
# "Only symmetric quantization is supported for MoE". Replacement is the SAME
# model, SAME compressed-tensors pack-quantized INT4 family, but symmetric
# (group 128, 20.9 GB): sahilchachra/KAT-Coder-V2.5-Dev-W4A16. Verified: full
# tokenizer + chat_template (preserve_thinking/enable_thinking) + identical
# generation_config sampling opinion (temp 1.0 / top_p 0.95 / top_k 20).
HF_MODEL_REPO = "sahilchachra/KAT-Coder-V2.5-Dev-W4A16"
SERVED_MODEL_NAME = "sahilchachra/KAT-Coder-V2.5-Dev-W4A16"
VLLM_MAX_MODEL_LEN = 65536
VLLM_TENSOR_PARALLEL_SIZE = 1
VLLM_EVAL_FLAGS = [
    "--enable-auto-tool-choice",
    "--tool-call-parser", "qwen3_coder",
    "--generation-config", "vllm",
    "--enable-prefix-caching",
    "--default-chat-template-kwargs", '{"preserve_thinking": true}',
    "--reasoning-parser", "qwen3",
    "--max-model-len", str(VLLM_MAX_MODEL_LEN),
    # KAT delta: REQUIRED — open weights ship no vision tower (model card note;
    # arch is the ConditionalGeneration/multimodal class).
    "--language-model-only",
]

VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8000
PROXY_PORT = 8080
CACHE_DIR = "/cache"
STARTUP_TIMEOUT_S = 45 * 60
VLLM_READY_TIMEOUT_S = 40 * 60
PROXY_UPSTREAM_TIMEOUT_S = 3600

IDLE_TIMEOUT_S = int(os.environ.get("ARC3_IDLE_TIMEOUT_S", "600"))
# Probe budget guard: ~$5 total => cap the container harder than the main rig.
MAX_LIFETIME_S = int(os.environ.get("ARC3_MAX_LIFETIME_S", str(3 * 3600)))

AUTH_EXEMPT = {
    ("GET", "/v1/models"),
    ("GET", "/health"),
}


def vllm_cmd(model_path: str) -> list[str]:
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
    from huggingface_hub import snapshot_download

    t0 = time.time()
    path = snapshot_download(repo_id=HF_MODEL_REPO)
    print(f"[arc3-kat] snapshot ready at {path} ({round(time.time() - t0, 1)}s)",
          flush=True)
    try:  # arch verification requested by the probe protocol
        with open(os.path.join(path, "config.json")) as f:
            cfg = json.load(f)
        print(f"[arc3-kat] model_type={cfg.get('model_type')} "
              f"archs={cfg.get('architectures')} "
              f"quant={cfg.get('quantization_config', {}).get('format')}",
              flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[arc3-kat] config probe failed: {e!r}", flush=True)
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
                print(f"[arc3-kat] vLLM ready, serving {ids}", flush=True)
                return
            raise RuntimeError(f"vLLM serves {ids}, expected {SERVED_MODEL_NAME!r}")
        except (urllib.error.URLError, OSError, TimeoutError):
            time.sleep(5)
    raise TimeoutError(f"vLLM not ready after {timeout_s}s")


def _start_lifetime_watchdog() -> None:
    cap = int(os.environ.get("ARC3_MAX_LIFETIME_S", str(MAX_LIFETIME_S)))

    def _reaper() -> None:
        time.sleep(cap)
        print(f"[arc3-kat] MAX LIFETIME {cap}s reached — exiting container",
              flush=True)
        os._exit(0)

    threading.Thread(target=_reaper, daemon=True).start()


def _run_auth_proxy() -> None:
    import http.server

    token = os.environ.get("TOKEN", "").strip()

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
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
                if not token:
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
            except urllib.error.HTTPError as e:
                self._reply(e.code, e.read() or b"{}")
            except Exception as e:  # noqa: BLE001
                self._reply(502, json.dumps(
                    {"error": f"upstream vLLM unreachable: {e!r}"}).encode())

        do_GET = _handle    # noqa: N815
        do_POST = _handle   # noqa: N815

    srv = http.server.ThreadingHTTPServer(("0.0.0.0", PROXY_PORT), Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[arc3-kat] auth proxy on :{PROXY_PORT}", flush=True)


_CONTAINER_ENV = {
    "USE_TF": "0",
    "TRANSFORMERS_NO_TF": "1",
    "TRANSFORMERS_NO_TORCHVISION": "1",
    "VLLM_NO_USAGE_STATS": "1",
    "HF_HOME": f"{CACHE_DIR}/huggingface",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "ARC3_MAX_LIFETIME_S": str(MAX_LIFETIME_S),
    # qwen3_5_moe is a hybrid-GDN arch like the 27B: flashinfer JIT-compiles
    # GDN modules at first request; persist artifacts in the shared volume.
    "FLASHINFER_WORKSPACE_BASE": f"{CACHE_DIR}/flashinfer",
}


if modal is not None:
    app = modal.App(APP_NAME)
    hf_cache_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

    image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
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
        path = _download_snapshot()
        hf_cache_vol.commit()
        return path

    @app.function(
        image=image,
        gpu=f"{GPU_KIND}:{N_GPU}",
        volumes={CACHE_DIR: hf_cache_vol},
        secrets=[modal.Secret.from_name(SECRET_NAME)],
        scaledown_window=IDLE_TIMEOUT_S,
        max_containers=1,
    )
    @modal.concurrent(max_inputs=64)
    @modal.web_server(port=PROXY_PORT, startup_timeout=STARTUP_TIMEOUT_S)
    def serve() -> None:
        model_path = _download_snapshot()
        hf_cache_vol.commit()
        cmd = vllm_cmd(model_path)
        print("[arc3-kat] starting:", " ".join(cmd), flush=True)
        process = subprocess.Popen(cmd)
        _wait_for_vllm(VLLM_READY_TIMEOUT_S, process)
        _start_lifetime_watchdog()
        _run_auth_proxy()

    @app.local_entrypoint()
    def smoke(url: str = "", token: str = "") -> None:
        """models probe + auth negative test + plain chat + TOOL-CALL PARSE
        test (the probe-critical property: qwen3_coder parser must emit a
        structured tool_calls array for KAT output)."""
        base = (url or serve.get_web_url()).rstrip("/")
        token = token or os.environ.get("ARC3_VLLM_TOKEN", "")
        if not token:
            raise SystemExit("pass --token or set ARC3_VLLM_TOKEN")

        def _post(payload: dict, tok: str, timeout: int = 900):
            req = urllib.request.Request(
                f"{base}/v1/chat/completions",
                data=json.dumps(payload).encode(), method="POST",
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {tok}"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())

        print(f"[smoke] GET {base}/v1/models (no auth — must be exempt)")
        with urllib.request.urlopen(f"{base}/v1/models",
                                    timeout=STARTUP_TIMEOUT_S) as r:
            ids = [m.get("id") for m in json.loads(r.read()).get("data", [])]
        assert SERVED_MODEL_NAME in ids, f"served {ids}, want {SERVED_MODEL_NAME!r}"
        print(f"[smoke] models OK: {ids}")

        basic = {
            "model": SERVED_MODEL_NAME,
            "messages": [{"role": "user",
                          "content": "Answer in one short sentence: what is 2 + 2?"}],
            "temperature": 0.0,
            "max_tokens": 96,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            _post(basic, "wrong-token", timeout=60)
            raise SystemExit("SMOKE FAIL: bad token ACCEPTED — auth is off")
        except urllib.error.HTTPError as e:
            assert e.code == 401, f"bad token got {e.code}, expected 401"
            print("[smoke] bad token rejected with 401: auth is ON")

        out = _post(basic, token)
        print(f"[smoke] chat OK: "
              f"{(out['choices'][0]['message'].get('content') or '').strip()!r}")

        # Tool-call parse test: model must return a STRUCTURED tool_calls array
        # (not tool markup leaked into content) through the qwen3_coder parser.
        tool_payload = {
            "model": SERVED_MODEL_NAME,
            "messages": [{"role": "user",
                          "content": "Use the run_python tool to compute 6*7. "
                                     "Call the tool now."}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "run_python",
                    "description": "Execute a python expression and return it.",
                    "parameters": {
                        "type": "object",
                        "properties": {"expr": {"type": "string"}},
                        "required": ["expr"],
                    },
                },
            }],
            "temperature": 1.0, "top_p": 0.95, "top_k": 20,  # KAT card sampling
            "max_tokens": 2048,
            "chat_template_kwargs": {"enable_thinking": True},
        }
        out = _post(tool_payload, token)
        msg = out["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        print(f"[smoke] tool test: finish={out['choices'][0].get('finish_reason')} "
              f"n_tool_calls={len(calls)}")
        if calls:
            fn = calls[0].get("function", {})
            print(f"[smoke] tool_call[0]: {fn.get('name')}({fn.get('arguments')})")
        assert calls, ("SMOKE FAIL: no structured tool_calls — qwen3_coder "
                       f"parser did not fire. content={msg.get('content')!r}")
        print("[smoke] PASS")
