# The public-notebook serving regime (keithtyser V14) and its Modal reproduction

Purpose: pin, with file:line evidence, every vLLM argument and environment
variable of the best public ARC-AGI-3 notebook's serving stack, and record
next to each one whether `offkaggle/modal_flashnext_serve.py` reproduces it
identically or deviates (and why). `offkaggle/test_flashnext_serve.py`
parses the machine-readable block at the bottom of this file and fails if
the module's constants drift from it.

## Sources (pulled copies; paths as of 2026-09-02)

| id | file | what it is |
|---|---|---|
| NB | `submission/_keith_copy/duck-qwen3-8-flash-next-nvfp4-mtp.ipynb` | the public notebook (cell 3 = the `TAAF_VLLM_*` profile env; cell 7 = attached datasets; cell 13 = harness settings) |
| SS | `scratchpad/search/field-delta/keith_smoke_v1/serving_setup.py` | the bundle's serving setup script (122,872 bytes, sha256 `037c041c…`) — turns `TAAF_VLLM_*` env into `vllm serve` args, builds the process env, applies the PLE patch |
| SI | `scratchpad/search/field-delta/keith_smoke_v1/SOURCE_IDENTITY.json` | pinned identities: HF repo + revision, docker image digests, vLLM version, PLE patch hashes |
| PI | `scratchpad/search/field-delta/keith_smoke_v1/vllm-patches/PATCH_IDENTITY.json` (copied verbatim to `offkaggle/flashnext_patches/`) | the PLE patch identity |
| LOG | `scratchpad/search/judge/kout/keithtyser/vllm-openai-server.log` | the commit-run vLLM server log (V14, 09-01 02:39Z) |
| VSI | `scratchpad/search/judge/kout/keithtyser/vllm-server-identity.json` | the exact `argv` the setup launched (sha256 of argv `60b4549a…`) |
| PROV | `scratchpad/search/judge/kout/keithtyser/vllm-setup-provenance.json` | runtime environment as resolved on Kaggle (python 3.12.13, glibc 2.35, CUDA 13.0, driver 580.159.04, RTX PRO 6000 Blackwell Server Edition 97,887 MiB, host RAM 189 GB) |

The scratchpad copies are session-local (`/private/tmp/claude-501/…/scratchpad/`);
the values below are transcribed so this file stands on its own.

## 1. Model

| item | public config | evidence | Modal reproduction |
|---|---|---|---|
| HF repo | `RadixArk/Qwen3.8-Flash-Next-NVFP4` | SI `model.hf_repo`; PROV `model_hf_repo` | **identical** — `HF_MODEL_REPO` |
| HF revision | `7b719225242aacd3dbd3f9407468c2ee9a9d2594` | SI `model.hf_revision`; PROV `model_hf_revision`; verified live 2026-09-02: `GET /api/models/RadixArk/Qwen3.8-Flash-Next-NVFP4` → sha `7b71922…`, public, not gated, 419 files, lastModified 2026-08-26 | **identical** — `HF_MODEL_REVISION`, pinned in `snapshot_download(revision=…)` |
| config.json sha256 | `e765305daba0951974308f4d32c075b52a6a45974730d273f2216718a994d624` | SS:40 `MODEL_CONFIG_SHA256`; PI `exact_model_config_sha256`; verified live: `resolve/7b71922…/config.json` hashes to the same value | **identical** — checked at serve start before launch (`_verify_model_dir`) |
| size / files | 135,253,622,894 bytes, 419 files, 206 safetensors shards | SI `model.total_bytes`, `file_count`; LOG:110 "Checkpoint size: 125.91 GiB" | identical bytes (HF snapshot of the same revision); Kaggle mounted a plain dir, Modal mounts an HF-cache snapshot dir (symlinks to blobs) — path plumbing only |
| served model name | `Qwen/Qwen3.8-Flash-Next-NVFP4` | SS:125 `SERVED_MODEL_NAME`; VSI argv; LOG:7 `served_model_name` | **identical** |
| chat template | `<model_dir>/chat_template.jinja` passed explicitly with `--chat-template` | SS:2354-2355; VSI argv; LOG:7 `chat_template` | **identical** (same file from the same revision) |

## 2. Runtime (vLLM build)

| item | public config | evidence | Modal reproduction |
|---|---|---|---|
| vLLM version | `0.1.dev20073+g8e685d198` (a dev build, NOT a PyPI release) | SS:58; SI `runtime.vllm_version`; LOG:3 banner "version 0.1.dev20073+g8e685d198" | **identical** — the SAME docker image is pulled by digest (below), so the wheel is byte-identical. No pip re-creation was needed; `wheels.vllm.ai/nightly` was not used. |
| runtime image | `vllm/vllm-openai:qwen38-flash-next`, index digest `sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8`, amd64 manifest `sha256:0aea30240f3e3d9ffae8526643950e170eb5fa07fc427016a9dd90892afa2aa3` | SS:49-53; SI `runtime.*`; verified live 2026-09-02 on Docker Hub: tag exists, index digest and amd64 digest match exactly | **identical layers** — `modal.Image.from_registry("vllm/vllm-openai@sha256:0aea3024…")`. Keith extracts 6 selected layers of this image over the Kaggle host (SI `runtime.composition`); Modal runs the whole image. Additions to the image: (a) the PLE patch (below), (b) a `python -> python3` symlink at `/usr/local/bin/python` so Modal can detect the interpreter (Modal builder ≥ 2025.06 installs nothing into the image; its client is mounted at runtime — verified in `modal/_image.py::_registry_setup_commands`), (c) `/opt/arc3/vllm-patches` (the patch bundle) and the `TAAF_VLLM_*` record env. site-packages are otherwise byte-identical (the `probe` entrypoint prints the versions and the PLE file hash). |
| python | CPython 3.12.13 at `/usr/bin/python3`, glibc 2.35, site-packages `/usr/local/lib/python3.12/dist-packages` | PROV `environment.python`, `site_packages`; VSI argv[0] | **identical** — same image; the server is launched with `/usr/bin/python3` |
| pinned libraries inside the image (observed by `probe` on Modal, 2026-09-02) | torch `2.13.0+cu130`, transformers `5.15.1`, flashinfer `0.6.17`, triton `3.7.1`, CUDA `13.0` | Modal probe log (`scratchpad/flashnext_probe.log`) — Keith's provenance skips the import probe in fast mode so these are the image's own values | **identical** by construction (same image digest) |
| CUDA toolkit | 13.0 at `usr/local/cuda-13.0`, `TORCH_CUDA_ARCH_LIST=12.0`, `VLLM_ENABLE_CUDA_COMPATIBILITY=0` (real host driver, no compat libs) | PROV `environment.cuda_home`, `torch_cuda_arch_list`; SS:1412 | **identical** toolkit (same image). Host driver: Kaggle 580.159.04 vs Modal 580.95.05 (modal.com/docs/guide/cuda: "driver 580.95.05, CUDA Driver API 13.0") — both are 580-series CUDA-13.0 drivers; deviation is the host, not the config. |
| PLE patch | `radixark_nvfp4_ple_fp8.patch` applied to `vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py`: stock sha `a71144c1…`, patched sha `a8a06474…` | SS:59-60, SS:960-1057 `patch_ple_layer()`; PI | **identical** — the same applicator script (sha `a9f88c01…`, copied verbatim to `offkaggle/flashnext_patches/`) runs at image build against the same site-packages path; it refuses to run unless the stock hash matches and verifies the patched hash after writing. |
| GPU | 1× NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB, compute cap 12.0 | PROV `inventory.gpu_rows`; SS:1828 `gpu_inventory()` requires "rtx pro 6000" | **identical GPU class** — Modal `gpu="RTX-PRO-6000"` (string verified on modal.com/docs/guide/gpu, 2026-09-02; pricing page "Nvidia RTX PRO 6000 $0.000842/s"). Modal does not say Server vs Workstation edition; both are GB202 96 GB SM120. |
| host RAM | 189 GB total, 187 GB available; setup requires ≥ 64 GiB available for the FP8 PLE CPU offload | PROV `inventory.host_mem_available_bytes`; SS:191 `MIN_HOST_AVAILABLE_BYTES` | **deviation (host)**: Modal `memory=131072` MiB request (128 GiB) — above the 64 GiB gate, below Kaggle's 189 GB. If Modal rejects the request at deploy, lower it but never below 65,536. |

## 3. The `vllm serve` command

Exact argv from VSI (the model directory is the only host-specific token):

```
/usr/bin/python3 -m vllm.entrypoints.cli.main serve <MODEL_DIR>
  --served-model-name Qwen/Qwen3.8-Flash-Next-NVFP4
  --host 127.0.0.1 --port 1234
  --load-format safetensors
  --dtype bfloat16
  --quantization modelopt_fp4
  --tensor-parallel-size 1
  --distributed-executor-backend mp
  --kv-cache-memory-bytes 5368709120
  --max-model-len 32768
  --max-num-seqs 8
  --max-num-batched-tokens 8192
  --async-scheduling
  --enable-chunked-prefill
  --max-cudagraph-capture-size 32
  --no-enable-prefix-caching
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --reasoning-parser qwen3
  --chat-template <MODEL_DIR>/chat_template.jinja
  --speculative-config {"method":"mtp","num_speculative_tokens":3}
  --no-enable-log-requests
  --disable-uvicorn-access-log
  --uvicorn-log-level info
```

How each argument is produced (SS `server_command()`, lines 2261-2391) and where the value comes from:

| arg | value | produced by | notebook source | Modal |
|---|---|---|---|---|
| entrypoint | `python3 -m vllm.entrypoints.cli.main serve` (i.e. `vllm serve`, NOT `openai.api_server`) | SS:2276-2279 | — | **identical** |
| `--served-model-name` | `Qwen/Qwen3.8-Flash-Next-NVFP4` | SS:2289 | SS:125 | **identical** |
| `--host` / `--port` | `127.0.0.1` / `1234` | SS:2291-2294 | SS:126-127 | **identical** (loopback; only the auth proxy is public) |
| `--load-format` | `safetensors` | SS:2295 | fixed | **identical** |
| `--dtype` | `bfloat16` | SS:2297 | fixed ("BF16 compute", NB cell 0) | **identical** |
| `--quantization` | `modelopt_fp4` | SS:2299 | fixed | **identical** |
| `--tensor-parallel-size` | `1` | SS:2301 | fixed | **identical** |
| `--distributed-executor-backend` | `mp` | SS:2303 | fixed | **identical** |
| `--moe-backend` | ABSENT (`TAAF_VLLM_MOE_BACKEND` unset → `moe_backend: null`) | SS:2306-2307 | VSI `vllm_tuning.moe_backend: null` | **identical** (absent; vLLM auto-picked `FLASHINFER_CUTLASS`, LOG:52) |
| `--kv-cache-memory-bytes` | `5368709120` (5 GiB); replaces `--gpu-memory-utilization 0.92` which is only used when the bytes are 0 | SS:2308-2318 | NB cell 3 `TAAF_VLLM_KV_CACHE_MEMORY_BYTES=5368709120` | **identical** |
| `--max-model-len` | `32768` | SS:2321 | SS:129 `ANALYZER_CONTEXT = 32_768` | **identical** |
| `--max-num-seqs` | `8` | SS:2323 | NB cell 3 `TAAF_VLLM_MAX_NUM_SEQS=8` (overrides SS:133 default 28) | **identical** |
| `--max-num-batched-tokens` | `8192` | SS:2325 | NB cell 3 `TAAF_VLLM_MAX_NUM_BATCHED_TOKENS=8192` | **identical** |
| `--async-scheduling` | present | SS:2327 | fixed | **identical** |
| `--enable-chunked-prefill` | present | SS:2273-2275, 2328 | fixed (`enable_chunked_prefill=True`) | **identical** |
| `--kv-cache-dtype` | ABSENT (only emitted when ≠ `auto`) | SS:2330-2334 | NB cell 3 `TAAF_VLLM_KV_CACHE_DTYPE=auto` | **identical** (absent) |
| `--max-cudagraph-capture-size` | `32` | SS:2335-2341 | NB cell 3 `TAAF_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE=32` | **identical** |
| `--no-enable-prefix-caching` | present | SS:2344-2348 | NB cell 3 `TAAF_VLLM_ENABLE_PREFIX_CACHING=0` | **identical** |
| `--enable-auto-tool-choice` | present | SS:2349 | fixed | **identical** |
| `--tool-call-parser` | `qwen3_coder` | SS:2350-2351 | fixed | **identical** |
| `--reasoning-parser` | `qwen3` | SS:2352-2353 | fixed | **identical** |
| `--chat-template` | `<MODEL_DIR>/chat_template.jinja` | SS:2354-2355 | fixed | **identical** |
| `--speculative-config` | `{"method":"mtp","num_speculative_tokens":3}` (compact separators; no `index_share_for_mtp_iteration`, no `num_speculative_tokens_per_batch_size`) | SS:2358-2381 | NB cell 3 `TAAF_VLLM_MTP_TOKENS=3`; VSI `mtp_index_share_for_iteration: false`, `mtp_dynamic_batch_schedule: null` | **identical** |
| `--no-enable-log-requests` | present | SS:2382 | fixed | **identical** |
| `--disable-uvicorn-access-log` | present | SS:2383 | fixed | **identical** |
| `--uvicorn-log-level` | `info` | SS:2384-2385 | fixed | **identical** |
| `--generation-config` | ABSENT (vLLM default = the model's `generation_config.json`) — note our 27B/3.8-27B rigs pass `--generation-config vllm`; Keith does not | absent from SS `server_command()` | — | **identical** (absent) |
| `--default-chat-template-kwargs` | ABSENT — our rigs pass `{"preserve_thinking": true}`; Keith does not | absent from SS | — | **identical** (absent) |

Expected startup evidence to reproduce (LOG): line 7 `non-default args: {…'kv_cache_memory_bytes': 5368709120, 'enable_prefix_caching': False, 'max_num_batched_tokens': 8192, 'max_num_seqs': 8, 'enable_chunked_prefill': True, 'async_scheduling': True, 'max_cudagraph_capture_size': 32, 'speculative_config': {'method': 'mtp', 'num_speculative_tokens': 3}}`; line 498 `reserved 5.0 GiB memory for KV Cache as specified by kv_cache_memory_bytes`; **line 499 `GPU KV cache size: 105,202 tokens, Maximum concurrency for 32,768 tokens per request: 3.21x`**; line 466 `Model loading took 81.8 GiB memory`; line 42 `cudagraph_capture_sizes: [1, 2, 4, 8, 16, 24, 32]`; line 52 `Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend`; line 54 `PleOffload: spawning worker`; line 416 `PLE offload matched 132 checkpoint tensor(s)`. The `serve` function tees the vLLM log to a file and the proxy exposes those lines at `GET /arc3/identity` (bearer-protected).

## 4. Process environment of the vLLM server

Built by SS `runtime_environment()` (lines 1301-1470). `os.environ.copy()` of the notebook process, then:

| env | value | evidence | Modal |
|---|---|---|---|
| `PYTORCH_ALLOC_CONF` | REMOVED; `expandable_segments:true` anywhere is a hard error | SS:1312-1327, 1389 | **identical** |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:False` | SS:1390 | **identical** |
| `PYTHONPATH` | `<site>/nvidia_cutlass_dsl/dsl_packages:<site>` (+ prior) | SS:1391-1394 | **identical** values with the image's native site path (`/usr/local/lib/python3.12/dist-packages`); in the image the `.pth` already does this — belt and braces |
| `PATH` | `<cuda-13.0>/bin:<runtime>/usr/local/bin:` + prior | SS:1395-1397 | **identical intent** with the image's native paths (`/usr/local/cuda-13.0/bin:/usr/local/bin:…`) |
| `LD_LIBRARY_PATH` | `[cuda lib, cuda lib64, torch/lib, site/nvidia/*/lib, <real driver dir>, /usr/local/nvidia/lib64, /usr/lib/x86_64-linux-gnu]` filtered by `is_dir()` + prior | SS:1362-1371, 1398-1401 | **same list, same order**, native paths, `is_dir()`-filtered; the driver dir on Modal is wherever `libcuda.so` resolves (`/usr/lib/x86_64-linux-gnu`) — Keith's relocated-runtime paths are the host difference, not a config one |
| `HF_HOME` | `<CACHE_ROOT>/huggingface` (CACHE_ROOT=`/tmp/qwen38-flash-next-vllm-cache`) | SS:204, 1374 | **deviation (plumbing)**: `/cache/huggingface` on the Modal Volume — that is where the snapshot lives; offline flags make it inert at serve time |
| `XDG_CACHE_HOME` | `<CACHE_ROOT>` | SS:1375 | value points at the volume cache root (`/cache/flashnext/cache`) |
| `TORCH_HOME` | `<CACHE_ROOT>/torch` | SS:1376 | same relative layout under the volume root |
| `TORCHINDUCTOR_CACHE_DIR` / `TRITON_CACHE_DIR` / `CUDA_CACHE_PATH` / `FLASHINFER_WORKSPACE_BASE` | `<COMPILE_CACHE_ROOT>/{torchinductor,triton,cuda,flashinfer}` (COMPILE_CACHE_ROOT=`/tmp/qwen38-flash-next-vllm-compile-cache`, fresh every run) | SS:205, 1377-1380 | **deviation (non-semantic)**: same layout but under `/cache/flashnext/compile-cache` on the Volume so torch.compile (LOG:486 16.9 s) and flashinfer JIT are paid once, not per cold start. Compiled artifacts are deterministic functions of the same wheel + same GPU arch. |
| `CUDA_DEVICE_ORDER` | `PCI_BUS_ID` | SS:1405 | **identical** |
| `CUDA_VISIBLE_DEVICES` | `0` | SS:1406 | **identical** |
| `CUDA_HOME` / `CUDACXX` | `<cuda-13.0>` / `<cuda-13.0>/bin/nvcc` | SS:1407-1408 | **identical** with the image's native `/usr/local/cuda-13.0` (the `probe` entrypoint asserts nvcc exists there) |
| `TMPDIR` | `/tmp/qwen38-flash-next-vllm-tmp` | SS:206, 1409 | **identical** path |
| `VLLM_PLE_CPU_OFFLOAD` | `1` | SS:1410 | **identical** |
| `VLLM_PLE_OFFLOAD_READY_TIMEOUT` | `1500` (= `SERVER_READY_TIMEOUT`) | SS:141, 1411 | **identical** |
| `VLLM_ENABLE_CUDA_COMPATIBILITY` | `0` | SS:1412 | **identical** |
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` | SS:1413 | **identical** |
| `VLLM_NO_USAGE_STATS` / `DO_NOT_TRACK` | `1` / `1` | SS:1414-1415 | **identical** |
| `VLLM_RADIXARK_QWEN38_NVFP4_PLE_FP8` | `1` (the patch gate) | SS:1416 | **identical** |
| `VLLM_RADIXARK_QWEN38_NVFP4_CONFIG_SHA256` | `e765305d…` | SS:1417 | **identical** |
| `HF_HUB_OFFLINE` / `HF_DATASETS_OFFLINE` / `TRANSFORMERS_OFFLINE` | `1` / `1` / `1` | SS:1418-1420 | **identical** |
| `PYTHONDONTWRITEBYTECODE` | `1` | SS:1421 | **identical** |
| `PYTHONHASHSEED` | `0` | SS:1422 | **identical** |
| `TORCH_CUDA_ARCH_LIST` | `12.0` | SS:1423 | **identical** |
| `TOKENIZERS_PARALLELISM` | `false` | SS:1424 | **identical** |
| `OMP_NUM_THREADS` / `MKL_NUM_THREADS` | `1` / `1` | SS:1425-1426; NB cell 3 `TAAF_VLLM_OMP_THREADS=1` | **identical** |
| inherited from the notebook process | `MPLBACKEND=Agg`, `TAAF_*`, `ONLY_RESET_LEVELS=true`, `LIBRARY_PATH=/usr/local/nvidia/lib64:…` (NB cell 3) | NB cell 3 | not set — harness-side, no effect on vLLM |
| `TAAF_VLLM_*` | the profile itself: `MAX_NUM_SEQS=8`, `KV_CACHE_MEMORY_BYTES=5368709120`, `MAX_CUDAGRAPH_CAPTURE_SIZE=32`, `MAX_NUM_BATCHED_TOKENS=8192`, `MTP_TOKENS=3`, `ENABLE_PREFIX_CACHING=0`, `KV_CACHE_DTYPE=auto`, `OMP_THREADS=1` (profile name `kv5-bf16-mtp3-c8-cg32`) | NB cell 3 | consumed by SS only; the resolved args above carry the effect. Recorded as `PUBLIC25_VLLM_PROFILE_ENV` in the module and exported into the container env for the record. |

## 5. Harness-side analyzer settings (not serving; for the telemetry experiments)

From PROV `persisted_analyzer_environment` — what the Duck harness sends to this server in the public config:
`LOCAL_ANALYZER_CONTEXT_WINDOW=32768`, `LOCAL_ANALYZER_MAX_OUTPUT=0`, `LOCAL_ANALYZER_ENABLE_THINKING=true`, `LOCAL_ANALYZER_TEMPERATURE=0.6`, `LOCAL_ANALYZER_TOP_P=0.95`, `LOCAL_ANALYZER_TOP_K=20`, `LOCAL_ANALYZER_YIELD_SECONDS=60`, `LOCAL_ANALYZER_TOOL_STEPS=0`, `LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS=1024`, `LOCAL_ANALYZER_TOOL_TIMEOUT=30`, `MULTIMODAL_CONTEXT=current_grid`, `MULTIMODAL_UPSCALE=4`, `INFERENCE_ANALYZER_MODEL=Qwen/Qwen3.8-Flash-Next-NVFP4`, `LOCAL_ANALYZER_PROVIDER=vllm`, base URL `http://127.0.0.1:1234/v1`. Harness: concurrency 28, 7920 s/game, analyzer_timeout 900 s (NB cell 13). The `smoke` entrypoint's chat completion uses the same sampling (temperature 0.6, top_p 0.95, top_k 20, thinking on).

## 6. Things Keith's setup does that the Modal app deliberately does NOT do

- Extract docker layers from a Kaggle dataset, verify layer manifests, apply whiteouts (SS:769-960): replaced by pulling the same image by digest.
- Verify the Kaggle model dataset manifest (SS:700-760): replaced by the HF revision pin + config.json sha check.
- Owned-process/session bookkeeping, watchdog with 2 restarts (`vllm_server_watchdog.py`): replaced by Modal's container lifecycle (the container exits if vLLM dies; Modal cold-starts on the next request).
- `nvidia-smi` gate "exactly one RTX PRO 6000": reproduced as a log line at serve start (`nvidia-smi` output is printed), not a hard gate — Modal's `gpu=` selects the class.

## 7. Cost/ops guards (Modal-only; not part of the public config)

`scaledown_window=900` (15 min idle → scale to zero), in-container hard lifetime cap 4 h (`ARC3_MAX_LIFETIME_S`), `max_containers=1`, bearer-token proxy on port 8080 (`GET /v1/models` and `GET /health` exempt, mirroring the other rigs), `memory=131072` MiB, `cpu=8`.

## Machine-readable pin (parsed by `offkaggle/test_flashnext_serve.py`)

<!-- REGIME-JSON-BEGIN -->
```json
{
  "gpu": "RTX-PRO-6000",
  "hf_model_repo": "RadixArk/Qwen3.8-Flash-Next-NVFP4",
  "hf_model_revision": "7b719225242aacd3dbd3f9407468c2ee9a9d2594",
  "model_config_sha256": "e765305daba0951974308f4d32c075b52a6a45974730d273f2216718a994d624",
  "served_model_name": "Qwen/Qwen3.8-Flash-Next-NVFP4",
  "vllm_version": "0.1.dev20073+g8e685d198",
  "vllm_image": "vllm/vllm-openai:qwen38-flash-next",
  "vllm_image_index_digest": "sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8",
  "vllm_image_amd64_manifest_digest": "sha256:0aea30240f3e3d9ffae8526643950e170eb5fa07fc427016a9dd90892afa2aa3",
  "site_packages": "/usr/local/lib/python3.12/dist-packages",
  "python": "/usr/bin/python3",
  "ple_patch_target": "vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py",
  "ple_stock_sha256": "a71144c1d36e06f22a2da1b1ada900076597fe5e824a911e7ada86249a0993e7",
  "ple_patched_sha256": "a8a064744efc3c99eefff649f50395d77e1c36085266034d17b51722f68176ed",
  "ple_applicator_sha256": "a9f88c019d04d7c8804ca6532e1c689d2ea97cf085cfec37dd4a779cb2c2c3fd",
  "ple_patch_diff_sha256": "e3b83f9a65e436d21d3e757ddc8dd2699f3c59de9f73967107b3b20c101d0a9f",
  "vllm_host": "127.0.0.1",
  "vllm_port": 1234,
  "argv_after_model_dir": [
    "--served-model-name", "Qwen/Qwen3.8-Flash-Next-NVFP4",
    "--host", "127.0.0.1",
    "--port", "1234",
    "--load-format", "safetensors",
    "--dtype", "bfloat16",
    "--quantization", "modelopt_fp4",
    "--tensor-parallel-size", "1",
    "--distributed-executor-backend", "mp",
    "--kv-cache-memory-bytes", "5368709120",
    "--max-model-len", "32768",
    "--max-num-seqs", "8",
    "--max-num-batched-tokens", "8192",
    "--async-scheduling",
    "--enable-chunked-prefill",
    "--max-cudagraph-capture-size", "32",
    "--no-enable-prefix-caching",
    "--enable-auto-tool-choice",
    "--tool-call-parser", "qwen3_coder",
    "--reasoning-parser", "qwen3",
    "--chat-template", "<MODEL_DIR>/chat_template.jinja",
    "--speculative-config", "{\"method\":\"mtp\",\"num_speculative_tokens\":3}",
    "--no-enable-log-requests",
    "--disable-uvicorn-access-log",
    "--uvicorn-log-level", "info"
  ],
  "public25_vllm_profile_name": "kv5-bf16-mtp3-c8-cg32",
  "public25_vllm_profile_env": {
    "TAAF_VLLM_ENABLE_PREFIX_CACHING": "0",
    "TAAF_VLLM_KV_CACHE_DTYPE": "auto",
    "TAAF_VLLM_KV_CACHE_MEMORY_BYTES": "5368709120",
    "TAAF_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE": "32",
    "TAAF_VLLM_MAX_NUM_BATCHED_TOKENS": "8192",
    "TAAF_VLLM_MAX_NUM_SEQS": "8",
    "TAAF_VLLM_MTP_TOKENS": "3",
    "TAAF_VLLM_OMP_THREADS": "1"
  },
  "server_env_exact": {
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUDA_VISIBLE_DEVICES": "0",
    "CUDA_HOME": "/usr/local/cuda-13.0",
    "CUDACXX": "/usr/local/cuda-13.0/bin/nvcc",
    "TMPDIR": "/tmp/qwen38-flash-next-vllm-tmp",
    "VLLM_PLE_CPU_OFFLOAD": "1",
    "VLLM_PLE_OFFLOAD_READY_TIMEOUT": "1500",
    "VLLM_ENABLE_CUDA_COMPATIBILITY": "0",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "VLLM_NO_USAGE_STATS": "1",
    "DO_NOT_TRACK": "1",
    "VLLM_RADIXARK_QWEN38_NVFP4_PLE_FP8": "1",
    "VLLM_RADIXARK_QWEN38_NVFP4_CONFIG_SHA256": "e765305daba0951974308f4d32c075b52a6a45974730d273f2216718a994d624",
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "TORCH_CUDA_ARCH_LIST": "12.0",
    "TOKENIZERS_PARALLELISM": "false",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1"
  },
  "server_env_removed": ["PYTORCH_ALLOC_CONF"],
  "server_env_cache_keys": [
    "HF_HOME", "XDG_CACHE_HOME", "TORCH_HOME", "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "FLASHINFER_WORKSPACE_BASE"
  ],
  "expected_max_concurrency_line": "GPU KV cache size: 105,202 tokens, Maximum concurrency for 32,768 tokens per request: 3.21x",
  "min_host_available_bytes": 68719476736,
  "modal_guards": {
    "idle_timeout_s": 900,
    "max_lifetime_s": 14400,
    "max_containers": 1,
    "memory_mib": 131072
  }
}
```
<!-- REGIME-JSON-END -->
