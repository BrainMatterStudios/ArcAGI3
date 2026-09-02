# Flash-Next (keithtyser V14 regime) on Modal — status

Date: 2026-09-02. Files: `offkaggle/modal_flashnext_serve.py` (the app),
`offkaggle/KEITH_REGIME.md` (every arg/env with evidence + the machine-readable
pin), `offkaggle/test_flashnext_serve.py` (8 host-only tests, all passing),
`offkaggle/flashnext_patches/` (Keith's PLE patch bundle, verbatim, hash-checked).

## What is reproduced EXACTLY

| layer | how | proof |
|---|---|---|
| vLLM build `0.1.dev20073+g8e685d198` | the SAME docker image Keith ships as Kaggle layer blobs, pulled by amd64 manifest digest `sha256:0aea3024…` (`vllm/vllm-openai:qwen38-flash-next`) | Docker Hub tag/digests verified live; the Modal build log copies the six layer blobs Keith pins in `EXPECTED_RUNTIME_LAYERS` (`f4f325a9…`, `a5881cf3…`, `692ea0d2…`, `651e9c66…`, `10dce885…`, `49a63ac9…`) |
| PLE-FP8 loader patch | Keith's own applicator run at image build against the same site-packages path | build log: stock sha `a71144c1…` → `patched` → patched sha `a8a06474…` verified by `grep -q` |
| model `RadixArk/Qwen3.8-Flash-Next-NVFP4` @ `7b71922…` | HF snapshot pinned to that revision in the Volume | HF API: rev exists, public; `config.json` sha `e765305d…` re-verified at every serve start |
| the `vllm serve` argv | `vllm_cmd()` — token-for-token Keith's `vllm-server-identity.json` argv (8 seqs, 5 GiB KV, 8192 batched, cudagraph 32, MTP 3, prefix caching OFF, async sched, chunked prefill, bf16, modelopt_fp4, mp, qwen3_coder tool parser, qwen3 reasoning parser, explicit chat template, no request logs) | `test_vllm_cmd_matches_keith_argv_token_for_token` |
| the server process env | `server_env()` — all 24 exact keys (PLE CPU offload, no expandable segments, spawn, OMP/MKL 1, offline HF, arch 12.0, seed 0, patch gate + config sha), `PYTORCH_ALLOC_CONF` removed, PATH/PYTHONPATH/LD_LIBRARY_PATH prepended in Keith's order | `test_server_env_matches_regime` |
| served model name / host / port | `Qwen/Qwen3.8-Flash-Next-NVFP4` on `127.0.0.1:1234` behind the proxy | — |
| GPU class | Modal `RTX-PRO-6000` (Blackwell 96 GB; the Kaggle eval GPU) | modal.com/docs/guide/gpu |
| Python / CUDA / glibc | same image: CPython 3.12.13, CUDA 13.0 toolkit, glibc 2.35 | — |

## Every deviation from the public config, with reason

1. **Host RAM**: Modal `memory=131072` MiB request vs Kaggle 189 GB. Keith's own gate is ≥ 64 GiB available (`MIN_HOST_AVAILABLE_BYTES`), enforced in `serve()` too. Needed for the FP8 PLE CPU offload (~44 GiB of the 126 GiB checkpoint lives in host RAM: GPU load 81.8 GiB).
2. **Host driver**: Modal 580.95.05 vs Kaggle 580.159.04 — both 580-series CUDA-13.0 drivers; `VLLM_ENABLE_CUDA_COMPATIBILITY=0` kept (real driver, no compat libs).
3. **Compile caches persisted** on the Volume (`/cache/flashnext/compile-cache/{torchinductor,triton,cuda,flashinfer}`) instead of Keith's fresh `/tmp/...-compile-cache` per run — saves torch.compile (~20 s) + flashinfer JIT per cold start; artifacts are deterministic for the same wheel + arch. Same relative layout.
4. **HF cache paths**: `HF_HOME=/cache/huggingface` (the Volume) vs `/tmp/qwen38-flash-next-vllm-cache/huggingface`; inert at serve time (`HF_HUB_OFFLINE=1`).
5. **Native image paths** for `CUDA_HOME`, site-packages, `LD_LIBRARY_PATH` entries (`/usr/local/cuda-13.0`, `/usr/local/lib/python3.12/dist-packages`) — Keith relocates the extracted runtime to `/tmp/qwen38-flash-next-vllm-runtime/...`; the values are the same directories at their native location.
6. **Image additions**: `python -> python3` symlink (Modal needs `python` on PATH to detect the interpreter; it installs nothing else — verified in `modal/_image.py::_registry_setup_commands` for builder ≥ 2025.06), `/opt/arc3/vllm-patches`, and the `TAAF_VLLM_*` record env. Keith's `__pycache__` prohibition after patching is honoured (`-B` + rm).
7. **Not reproduced (harness-side / Kaggle-side, not serving)**: layer-extraction + whiteout verification, Kaggle model-manifest verification, owned-process bookkeeping and the 2-restart watchdog (Modal restarts the container on the next request instead), `nvidia-smi` hard gate (printed, not enforced — `gpu=` selects the class).
8. **Modal-only guards**: bearer proxy on :8080 (`GET /v1/models` and `GET /health` exempt; `GET /arc3/identity` returns the startup-log evidence incl. the "Maximum concurrency" line), `scaledown_window=900`, 4 h hard lifetime cap, `max_containers=1`, `cpu=8`.
9. **Modal web-endpoint HTTP semantics**: single responses cap at 150 s then continue via 303 redirects (documented in `offkaggle/README.md` caveat 3); the harness's `requests.post` follows redirects.

Expected startup line to match: `GPU KV cache size: 105,202 tokens, Maximum concurrency for 32,768 tokens per request: 3.21x` (Keith's log line 499). The number of KV tokens may differ slightly if free memory at load differs; the 5 GiB reservation is fixed by `--kv-cache-memory-bytes`.

## Unverified risks (need the first GPU smoke)

- `/dev/shm` sizing on Modal for the PLE-offload IPC (Keith's setup records shm but does not enforce a minimum; `probe` prints it at the serve-time resource request).
- Volume read throughput for the 126 GiB weight load (Keith: 328 s from NFS).
- Whether the RTX-PRO-6000 nodes admit a 128 GiB memory request (deploy would fail with `InvalidError` — lower `MEMORY_MIB`, never below 65,536).

## Warm status

(filled in below)

## Deploy (done 2026-09-02, ~19:5x CEST)

- `modal deploy offkaggle/modal_flashnext_serve.py` → **https://a-m-mobasher--arc3-flashnext-serve.modal.run** (harness base URL = that + `/v1`). App `arc3-flashnext`; dashboard https://modal.com/apps/a-m-mobasher/main/deployed/arc3-flashnext. Costs nothing until the first request.
- Attempt 1 failed at Modal's Python detection (the image ships only `python3`); fixed with a `setup_dockerfile_commands` symlink `python -> python3` — no second interpreter, nothing pip-installed (builder ≥ 2025.06 mounts the Modal client at runtime).
- Build log evidence: stock `ple_layer.py` sha `a71144c1…` → applicator printed `patched` → sha `a8a06474…` verified; the pulled blobs include Keith's six pinned layer hashes.
- Functions accepted as declared: `serve` = `gpu="RTX-PRO-6000:1"`, `cpu=8`, `memory=131072` MiB, `scaledown_window=900`, `max_containers=1`, 45-min startup timeout.

### CPU probe of the built image (`modal run …::probe`, rc=0, no GPU)

```
vllm 0.1.dev20073+g8e685d198   (== Keith's pin)
torch 2.13.0+cu130  transformers 5.15.1  flashinfer 0.6.17  triton 3.7.1  cuda 13.0
ple_layer.py sha256 a8a064744efc… (patched == pin)   nvcc /usr/local/cuda-13.0/bin/nvcc   cutlass .pth present
container: 24 vCPU, MemAvailable 780 GB (the 128 GiB request was honoured), /dev/shm free 16 GiB
server_env applied in-container == KEITH_REGIME §4 (all 24 exact keys; PYTHONPATH/PATH/LD_LIBRARY_PATH prepended in Keith's order with native paths)
```

`/dev/shm` = 16 GiB vs Keith's 92 GB free: not enforced by his setup; the PLE-offload worker keeps the ~44 GiB table in its own process memory and talks over a ZMQ `ipc://` socket in `TMPDIR`, so this is not expected to bind, but it is unverified until the GPU smoke.

## Smoke command (GPU minutes — orchestrator's call)

```sh
# ~$3.03/h on RTX PRO 6000; first hit cold-starts (image already built; weights from the Volume): expect 10-20 min.
/Users/ahmed/Library/Python/3.14/bin/modal run offkaggle/modal_flashnext_serve.py::smoke \
    --url https://a-m-mobasher--arc3-flashnext-serve.modal.run --token "$(cat ~/.config/arc3/vllm_token)"
# (the token file is also read automatically if --token is omitted)
```

The smoke: `GET /v1/models` without and with the token (expects exactly
`["Qwen/Qwen3.8-Flash-Next-NVFP4"]`, `max_model_len` 32768), a wrong-token
`POST` must 401, `GET /arc3/identity` prints the vLLM version, the argv,
`nvidia-smi`, ready/cold-start seconds and the grep'd startup lines (max
concurrency, KV reservation, NvFp4 MoE backend, PLE offload, cudagraph sizes),
then ONE thinking-enabled chat completion at the harness sampling settings
(temperature 0.6, top_p 0.95, top_k 20, `enable_thinking: true`, 512 max
tokens) reporting completion tokens/s. Afterwards `modal app stop arc3-flashnext`
to skip the 15-min idle window.

Telemetry hooks for the experiments: `GET /metrics` (vLLM Prometheus, bearer
required) is forwarded by the proxy — the equivalent of Keith's
`vllm-metrics-final.prom`.
