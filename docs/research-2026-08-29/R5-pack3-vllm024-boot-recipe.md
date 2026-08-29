# pack3 recipe — how jacquesbuis boots vLLM 0.24.0 on the Kaggle RTX PRO 6000, and what we'd copy

Read-only research, 2026-08-29. Nothing in the bundle or in our repo was modified.

## 0. Sources actually read

| What | Path / ref |
|---|---|
| Their bundle (dataset `jacquesbuis/taaf-src-mc-r4-trt-s20260829`) | `/private/tmp/claude-501/-Users-ahmed-Documents-ArcAGI3/336208eb-4625-4ff9-b10d-1664cc846e5d/scratchpad/r1/ds/jacquesbuis_taaf-src-mc-r4-trt-s20260829/` — `setup_commands.json` (1 command, 73,379 chars), `teardown_commands.json`, `run_config.json`, `source-manifest.json`, `preamble.txt`, `git_status.txt`, `src/ARC3-Inference/inference/framework/{kaggle.py,qwen38_manifest.py,vllm_watchdog.py}`, `src/ARC3-Inference/inference/tools/metacontrol_gate.py`, `src/ARC3-Inference/configs/inference.kaggle-v2.metacontrol-turn-contract.json`, `src/tufa-arc-agi-framework/src/taaf/deploy_kaggle.py`, `src/tufa-arc-agi-framework/src/taaf/kaggle/taaf_kaggle_run.ipynb` |
| Their wheelhouse (fetched with the Kaggle CLI, read-only) | `jcole75/arc3-qwen36-runtime-wheels` file listing; `requirements-runtime.txt`, `WHEELHOUSE_MANIFEST.json`, `.arc3-generated-output.json` saved to `scratchpad/wh_jcole75/` |
| saltb0x 0.27.1 wheelhouse (for the risk section) | `saltb0x/arc3-vllm-wheelhouse-v0271-cu129` file listing; `requirements.lock`, `ovl_manifest.json` saved to `scratchpad/wh_saltb0x/`; `saltb0x/qwen3-8-27b-fp8` `config.json` |
| Our failed 0.27.1 attempt | `/Users/ahmed/Documents/ArcAGI3/submission/_serving_lab2/{build_serving_lab2.py,kernel-metadata.json,v1-boot-fail-flashinfer-sm75.log,v2-boot-fail-cuda129-nvcc.log}` |
| Our working 0.19 stack | `/Users/ahmed/Documents/ArcAGI3/scratchpad/bundles/anim_20260807/setup_commands.json` |
| Our MTP standing rule | `/Users/ahmed/Documents/ArcAGI3/docs/RESEARCH-2026-08-21-bug-lever-hunt.md` lines 150-155; `docs/HANDOFF-2026-08-21-pack-parity-and-big-levers.md` lines 83-91 |

Provenance of the bundle: `git_status.txt` says both repos are at `820bb68 DIRTY apex` ("apex: flag-split addendum/notices, a16 progress-stall trigg…"). `run_config.json` has `"dry_run": true`, `"kernel_slug": "arc3-mc-r4-trt-p25-s20260829"`, `"accelerator": "NvidiaRtxPro6000"`, `"enable_internet": false`, `"run_as_submission": false`, `"concurrent_jobs": 28`, `"max_runtime_minutes_per_game": 132.0`, `"max_experiment_runtime_hours": 9.0`. The kernel itself is private (`kaggle kernels status` → permission denied), so I could not read a run log; every "it booted" claim below comes from comments in their own code, not from an observed run.

---

## 1. What the bundle mounts and how pip is invoked

### 1.1 Kaggle inputs

Declared by the solver via `kaggle_dataset_sources` (`deploy_kaggle.py` line 303-308), rendered from `kaggle.py`:

```python
DEFAULT_VLLM_WHEELHOUSE_DATASET_SOURCE = "jcole75/arc3-qwen36-runtime-wheels"
DEFAULT_QWEN_MODEL_DATASET_SOURCE = QWEN38_KAGGLE_DATASET_SOURCE      # = "saltb0x/qwen3-8-27b-fp8"
DEFAULT_SERVED_MODEL_NAME = QWEN38_HUGGING_FACE_REPO                  # = "Qwen/Qwen3.8-27B-FP8"
```

and in the shipped script:

```python
WHEELHOUSE_OWNER = 'jcole75'
WHEELHOUSE_SLUG = 'arc3-qwen36-runtime-wheels'
MODEL_OWNER = 'saltb0x'
MODEL_SLUG = 'qwen3-8-27b-fp8'
SERVED_MODEL_NAME = 'Qwen/Qwen3.8-27B-FP8'
```

| Input | Kind | Pinned identity in the script |
|---|---|---|
| `jcole75/arc3-qwen36-runtime-wheels` | Kaggle **dataset**, version 3 (`WHEELHOUSE_KAGGLE_DATASET_VERSION = 3`). Listed title "ARC3 Qwen3.6 NVFP4 Runtime Wheels", created 2026-07-10, 6,448,804,222 bytes listed, 153 downloads. Root members must be exactly `{'.arc3-generated-output.json', 'WHEELHOUSE_MANIFEST.json', 'requirements-runtime.txt', 'wheels'}`; manifest must say `wheel_count == 197`, `total_bytes == 6486476943`, `cuda_wheel_target == 'cu129'`. | `WHEELHOUSE_MANIFEST_SHA256 = 'b765fe47…22d2'`, `WHEELHOUSE_REQUIREMENTS_SHA256 = '4d436150…7b47'`, `WHEELHOUSE_GENERATED_METADATA_SHA256 = 'cd319d1f…3bf0'`; then every one of the 197 wheels is size+SHA-256 checked against the manifest. |
| `saltb0x/qwen3-8-27b-fp8` | Kaggle **dataset**, version 1 (`MODEL_KAGGLE_DATASET_VERSION = 1`); an 81-file mirror of HF `Qwen/Qwen3.8-27B-FP8` at revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`, `MODEL_TOTAL_BYTES = 30890049597`. Includes `('mtp.safetensors', 477202224, 'e5e4464a3793cc261de536592830bca40e7f3af159ed038c358f5660917cf43b')`. Layout is the official one: `layers-0..63.safetensors`, `outside.safetensors`, `mtp.safetensors`. | every file size+SHA-256 checked; `config.json` asserted to be `architectures == ['Qwen3_5ForConditionalGeneration']`, `model_type == 'qwen3_5'`, `language_model_only is False`, 64 layers with 16 `full_attention`, `max_position_embeddings == 262144`, `quantization_config` = `fp8`/`e4m3`/`dynamic`/`weight_block_size [128,128]`; `model.safetensors.index.json` must reference exactly the 66 `.safetensors` files (so `mtp.safetensors` **is** in the index). |
| `jacquesbuis/taaf-src-mc-r4-trt-s20260829` | the source bundle itself (attached as `dataset_ref`) | verified by `source-manifest.json` SHA-256s in the notebook |
| `arc-prize-2026-arc-agi-3` | competition source (`competition_sources`) | the notebook pip-installs `arc-agi` from `/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels` |

**There is no separate CUDA-toolkit dataset.** The CUDA 13.3 compiler comes from wheels inside the jcole75 wheelhouse (section 2).

Mount resolution (identical in our 0.19 script): the notebook writes `TAAF_KAGGLE_INPUT_PATHS` (JSON `{owner/slug: path}`) after trying `/kaggle/input/<slug>` then `/kaggle/input/datasets/<owner>/<slug>`; the setup script's `resolve_kaggle_dataset_path()` uses that map and falls back to the same two candidates. Note this only handles *datasets* — a Kaggle *model* mount like ours (`/kaggle/input/models/foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1`) would not be found by it.

### 1.2 The wheelhouse's own requirements file (verbatim, `requirements-runtime.txt`, 773 bytes)

```
# Built for Kaggle's ARC-AGI-3 Python 3.12 / RTX PRO 6000 Blackwell runtime.
# Keep this file and every resolved wheel together in one Kaggle Dataset.
vllm==0.24.0
transformers==5.13.0
nvidia-modelopt==0.45.0
# FlashInfer JITs SM120 kernels on Kaggle. Keep the packaged CUDA 13 compiler,
# headers, runtime, and cuRAND headers on one coherent 13.3 toolchain.
nvidia-cuda-nvcc==13.3.73
nvidia-cuda-runtime==13.3.29
nvidia-cuda-cccl==13.3.3.4.1
nvidia-cuda-nvrtc==13.3.33
nvidia-curand==10.4.3.29
nvidia-cublas==13.3.0.5
# OmegaConf's matching ANTLR runtime is source-only on PyPI; the wheelhouse
# builder compiles this pure-Python wheel before resolving the binary closure.
antlr4-python3-runtime==4.9.3
torchvision>=0.20,<1
Pillow>=11.0,<13
numpy>=1.26,<3
orjson>=3.10,<4
```

`WHEELHOUSE_MANIFEST.json` header: `created_utc 2026-07-10T03:47:39Z`, `builder_python 3.12.13`, `platform Linux-6.12.90+-x86_64-with-glibc2.35`, `cuda_wheel_target: cu129`, `cuda_jit_toolchain: {'nvidia_cuda_nvcc': '13.3.73', 'nvidia_cuda_runtime': '13.3.29', 'nvidia_cuda_cccl': '13.3.3.4.1', 'nvidia_cuda_nvrtc': '13.3.33', 'nvidia_curand': '10.4.3.29', 'nvidia_cublas': '13.3.0.5'}`.

Key wheels in `wheels/` (from the CLI listing):

```
vllm-0.24.0-cp38-abi3-manylinux_2_28_x86_64.whl                     279,209,310
torch-2.11.0+cu129-cp312-cp312-manylinux_2_28_x86_64.whl          1,164,267,510
torchvision-0.26.0+cu129-cp312-…                                      9,127,040
torchaudio-2.11.0+cu129-cp312-…                                       1,572,646
transformers-5.13.0-py3-none-any.whl
flashinfer_python-0.6.12-py3-none-any.whl                            13,985,243
flashinfer_cubin-0.6.12-py3-none-any.whl                            447,533,460
triton-3.6.0-cp312-…  +  tokenspeed_triton-3.7.10.post20260531-cp312-abi3-…
nvidia_cuda_nvcc-13.3.73-py3-none-manylinux2014_x86_64…             44,942,138   <- real nvcc (CUDA 13 layout)
nvidia_cuda_crt-13.3.73-…                                                157,352   <- nvcc's crt/ headers
nvidia_cuda_runtime-13.3.29-…                                          2,339,786   <- cuda_runtime.h + libcudart.so.13
nvidia_cuda_cccl-13.3.3.4.1-…                                          3,454,034
nvidia_cuda_nvrtc-13.3.33-…                                           51,110,910
nvidia_nvvm-13.3.73-…                                                 69,250,424
nvidia_cublas-13.3.0.5-…  nvidia_curand-10.4.3.29-…  nvidia_cutlass_dsl_libs_cu13-4.5.2-…
nvidia_*_cu12 (cublas 12.9.1.4, cudnn 9.17.1.4, nccl 2.28.9, nvrtc 12.9.86, runtime 12.9.79, nvjitlink 12.9.86 …)  <- what torch+cu129 links
cuda_toolkit-12.9.1, cuda_bindings-12.9.7, cuda_python-12.9.7, ninja-1.13.0, setuptools-80.10.2, nvidia_modelopt-0.45.0
```

So the house carries **two CUDA generations on purpose**: the cu12 (12.9) runtime libraries that `torch-2.11.0+cu129` links against, plus a self-contained CUDA **13.3** compiler toolchain used only for FlashInfer JIT.

### 1.3 The pip invocation (verbatim from `install_vllm_wheelhouse()`)

```python
cmd = [
    sys.executable, '-m', 'pip', 'install',
    '--no-index',
    '--find-links', str(WHEELHOUSE_WHEELS),          # <wheelhouse>/wheels
    '--requirement', str(WHEELHOUSE_REQUIREMENTS),   # <wheelhouse>/requirements-runtime.txt
    '--target', str(SITE_PACKAGES),                  # $TAAF_KAGGLE_WORKING_DIR/vllm-site-packages  (= /kaggle/working/vllm-site-packages)
    '--upgrade',
    '--ignore-installed',
    '--only-binary', ':all:',
    '--no-compile',
    '--no-cache-dir',
    '--disable-pip-version-check',
    '--no-warn-conflicts',
]
subprocess.run(cmd, check=True)
```

* One pip call; pip resolves the full closure from the 197 wheels. **No `--no-deps`, no constraints file, no pinned install order** — the requirements file lists only top-level pins and pip picks the rest from the wheelhouse.
* `--target` into a private dir; the runtime is exposed to vLLM through `PYTHONPATH` (prepended), never installed into the system site-packages.
* Preflight before install: `shutil.disk_usage(WORKING_DIR).free >= WHEELHOUSE_INSTALLED_PAYLOAD_BYTES + MIN_WORKING_HEADROOM_BYTES` = `13_250_838_162 + 2*1024**3` bytes, else `RuntimeError('Insufficient /kaggle/working capacity …')`.
* Cache stamp: `INSTALL_STAMP = SITE_PACKAGES / '.arc3-qwen36-runtime-wheels'` with `STAMP_TEXT = 'dataset=jcole75/arc3-qwen36-runtime-wheels@3 vllm==0.24.0 torch==2.11.0+cu129 transformers==5.13.0 flashinfer==0.6.12 model=Qwen/Qwen3.8-27B-FP8\n'`.
* Post-install gates (both also run on the cached path):
  * `assert_installed_runtime_versions()` — `importlib.metadata.version` must equal `{'vllm': '0.24.0', 'torch': '2.11.0+cu129', 'transformers': '5.13.0', 'flashinfer-python': '0.6.12'}` exactly.
  * `assert_cuda_jit_toolchain()` — see section 2.
* Order of operations in the whole setup: `assert_expected_cuda_gpu()` (nvidia-smi name contains `rtx pro 6000` and memory ≥ 90000 MiB) → `verify_model_assets()` (SHA every model file, ~30.9 GB read) → `start_vllm_server()` which calls `install_vllm_wheelhouse()` (`verify_wheelhouse_assets()` SHAs all 197 wheels, then pip) → launch → `wait_for_vllm_server()` → `run_vllm_api_smoke_test()` → attestation → boot receipt → watchdog → write `TAAF_KAGGLE_SETUP_ENV`.
* Earlier, in the notebook (cell 2), the competition SDK is installed into the *system* interpreter: `pip install --no-index --no-warn-conflicts --disable-pip-version-check --find-links /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels arc-agi`. TAAF/ARC3-Inference sources are not pip-installed; the notebook writes a `taaf_kaggle_sources.pth` into `sysconfig.get_paths()["purelib"]` and prepends `sys.path`.

---

## 2. The CUDA_HOME / nvcc trick

### 2.1 Where the toolkit comes from

```python
SITE_PACKAGES = WORKING_DIR / 'vllm-site-packages'
CUDA_HOME_PATH = SITE_PACKAGES / 'nvidia' / 'cu13'
CUDA_BIN_PATH = CUDA_HOME_PATH / 'bin'
CUDA_LIB_PATH = CUDA_HOME_PATH / 'lib'
SITE_BIN_PATH = SITE_PACKAGES / 'bin'
```

The CUDA **13.x** NVIDIA pip wheels (`nvidia-cuda-nvcc`, `nvidia-cuda-runtime`, `nvidia-cuda-crt`, `nvidia-cuda-cccl`, `nvidia-cuda-nvrtc`, `nvidia-nvvm`, `nvidia-cublas`, `nvidia-curand` — the *unsuffixed* names) install into the unified `site-packages/nvidia/cu13/{bin,include,lib,nvvm}` layout. That is a different layout from the CUDA-12 `-cu12` wheels, which land in `site-packages/nvidia/<component>/`. This is exactly why our v3 plan noted that the pip `nvidia/cuda_nvcc` dir (from `nvidia-cuda-nvcc-cu12==12.9.86`) "ships NO `bin/nvcc` (only `ptxas` + `nvvm`)" (`build_serving_lab2.py` line 75): the cu12 nvcc wheel is not a compiler; the CUDA-13 one is.

The bundle asserts the toolchain is real before launching (`assert_cuda_jit_toolchain()`):

```python
required = (
    CUDA_BIN_PATH / 'nvcc',
    CUDA_HOME_PATH / 'include' / 'cuda_runtime.h',
    CUDA_HOME_PATH / 'include' / 'cuda.h',
    CUDA_LIB_PATH / 'libcudart.so.13',
    Path(sysconfig.get_path('include')) / 'Python.h',
)
… for executable in ('nvcc', 'ninja', 'c++'): shutil.which(executable, path=env['PATH']) …
nvcc = subprocess.run([str(CUDA_BIN_PATH / 'nvcc'), '--version'], env=env, …)
if nvcc.returncode != 0 or 'release 13.3' not in nvcc.stdout:
    raise RuntimeError(f'Pinned CUDA compiler probe failed: …')
print('CUDA 13.3 Blackwell JIT toolchain verified:', CUDA_HOME_PATH, flush=True)
```

### 2.2 The environment vLLM is launched with (`vllm_env()`, verbatim semantics)

```python
env = os.environ.copy()
env.pop('VLLM_ATTENTION_BACKEND', None)          # "deliberately uses vLLM's automatic/default attention backend"
env['PYTHONPATH'] = SITE_PACKAGES [+ ':' + existing]
env['CUDA_HOME'] = str(CUDA_HOME_PATH)          # …/vllm-site-packages/nvidia/cu13
env['CUDA_PATH'] = str(CUDA_HOME_PATH)
env['FLASHINFER_NVCC'] = str(CUDA_BIN_PATH / 'nvcc')
env['FLASHINFER_EXTRA_LDFLAGS'] = f'-L{CUDA_LIB_PATH} -L/usr/local/nvidia/lib64'
PATH            = <cu13/bin>:<site-packages/bin>:$PATH
LD_LIBRARY_PATH = <cu13/lib>:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
LIBRARY_PATH    = <cu13/lib>:/usr/local/nvidia/lib64:$LIBRARY_PATH
CPATH           = <cu13/include>:$CPATH
env.update({
    'USE_TF': '0',
    'TRANSFORMERS_NO_TF': '1',
    'TRANSFORMERS_NO_TORCHVISION': '1',
    'PYTHONDONTWRITEBYTECODE': '1',
    'VLLM_NO_USAGE_STATS': '1',
    'HF_HUB_OFFLINE': '1',
    'VLLM_USE_FLASHINFER_SAMPLER': '0',
    'VLLM_USE_DEEP_GEMM': '0',
    'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
    'XDG_CACHE_HOME': str(CACHE_ROOT),                              # /kaggle/temp/duck-vllm-cache (or $TMPDIR, /tmp)
    'TORCH_EXTENSIONS_DIR': str(CACHE_ROOT / 'torch-extensions'),
    'TRITON_CACHE_DIR': str(CACHE_ROOT / 'triton'),
    'FLASHINFER_WORKSPACE_BASE': str(CACHE_ROOT / 'flashinfer'),
})
```

Their own comments on the two `VLLM_USE_*` switches:

> `# vLLM 0.24 + torch 2.11/CUDA 13.3 can fail while JIT-compiling FlashInfer's SM120 sampler. The native sampler preserves logits and avoids a whole-run boot/engine-death failure on RTX PRO 6000.` → `'VLLM_USE_FLASHINFER_SAMPLER': '0'`

> `# vLLM 0.24's DeepGEMM warm-up can ignore the block-FP8 model's automatic fallback on Blackwell SM120 and abort with an unknown recipe. The standard CUTLASS path preserves the same FP8 model and avoids that model-load failure.` → `'VLLM_USE_DEEP_GEMM': '0'`

**Not set** anywhere in the bundle: `TORCH_CUDA_ARCH_LIST`, `FLASHINFER_CUDA_ARCH_LIST`, `VLLM_ATTENTION_BACKEND` (explicitly removed), `VLLM_FLASH_ATTN_VERSION`, `NVCC_PREPEND_FLAGS`.

### 2.3 Why this dodges the two failures we hit

Our two 0.27.1 boots died in the FlashInfer **sampler** path, both times inside `flashinfer.jit`:

* v1 (`v1-boot-fail-flashinfer-sm75.log` line 172): `topk_topp_sampler.py … flashinfer_sample → get_sampling_module().build_and_load() → gen_jit_spec → check_cuda_arch()` → `RuntimeError: FlashInfer requires GPUs with sm75 or higher` (the image's pre-sm75 `TORCH_CUDA_ARCH_LIST`).
* v2 (`v2-boot-fail-cuda129-nvcc.log` line 84-85): at *import* time, `flashinfer/jit/env.py:_get_workspace_dir_name → CompilationContext() → _normalize_cuda_arch` → `RuntimeError: SM 12.x requires CUDA >= 12.9` (flashinfer parsed the image's pre-12.9 `/usr/local/cuda` `nvcc --version`).

The jcole75 stack attacks the same two seams differently:

1. `VLLM_USE_FLASHINFER_SAMPLER=0` means vLLM never asks FlashInfer for the sampler at all — the exact call-site of our v1 crash and the trigger of our v2 import crash (`topk_topp_sampler.py:51 flashinfer_sampler_supported → from vllm.v1.attention.backends.flashinfer import FlashInferBackend → import flashinfer`). Our own lab script already had this as its fallback (`build_serving_lab2.py` lines 525-530: "retrying with VLLM_USE_FLASHINFER_SAMPLER=0"), but only after a failed boot.
2. If anything else in vLLM *does* import/JIT FlashInfer, the compiler it finds on `PATH`/`CUDA_HOME` is a **13.3** nvcc, so `is_cuda_version_at_least("12.9")`-style checks pass and SM 12.0 kernels can be compiled. The `nvidia_cuda_crt`/`cccl`/`runtime` wheels give it the headers, and `FLASHINFER_EXTRA_LDFLAGS=-L<cu13/lib> -L/usr/local/nvidia/lib64` plus `LD_LIBRARY_PATH` make `libcudart.so.13` resolvable at link and load time. Precompiled kernels still come from `flashinfer_cubin-0.6.12`.
3. Attention backend is left to vLLM's auto-selection. On our 0.27.1 boot vLLM chose `FLASH_ATTN` for this model ("Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION']"), so FlashInfer attention is not on the hot path either way; DeepGEMM is auto-disabled for `qwen3_5_text` on Blackwell in 0.27 too, they just force it in 0.24.

Caveat: I did not open flashinfer 0.6.12's source, so I can't say whether it honours `FLASHINFER_NVCC` specifically; `CUDA_HOME` + `PATH` are what torch's `cpp_extension` uses to find `nvcc`, and those are set consistently. The `FLASHINFER_NVCC` line is belt-and-braces from their side.

---

## 3. The exact vLLM launch

### 3.1 argv (verbatim list, `start_vllm_server()`)

```python
cmd = [
    sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
    '--model', str(MODEL_PATH),                     # /kaggle/input/qwen3-8-27b-fp8 (dataset mount)
    '--served-model-name', 'Qwen/Qwen3.8-27B-FP8',
    '--host', '127.0.0.1',
    '--port', '1234',
    '--tensor-parallel-size', '1',
    '--dtype', 'bfloat16',
    '--max-num-seqs', '32',
    '--gpu-memory-utilization', '0.9',
    '--mm-encoder-tp-mode', 'data',
    '--limit-mm-per-prompt', '{"image": 1, "video": 0}',
    '--mamba-cache-mode', 'align',
    '--enable-auto-tool-choice',
    '--tool-call-parser', 'qwen3_coder',
    '--generation-config', 'vllm',
    '--enable-prefix-caching',
    '--default-chat-template-kwargs', '{"preserve_thinking": true, "reasoning_effort": "xhigh"}',
    '--reasoning-parser', 'qwen3',
    '--max-model-len', '65536',
]
if VLLM_SPECULATIVE_CONFIG and VLLM_SPECULATIVE_CONFIG.strip().lower() not in {'off', 'none', '0'}:
    cmd += ['--speculative-config', VLLM_SPECULATIVE_CONFIG]
```

Flags that are **absent** (and that their attestation records as defaults): `--kv-cache-dtype` (attested as `'auto'`, i.e. bf16 KV), `--attention-backend` (attested `'auto'`), `--enforce-eager` (False), `--trust-remote-code` (False). Constants: `VLLM_MAX_MODEL_LEN = 65536`, `VLLM_MAX_NUM_SEQS = 32`, `VLLM_GPU_MEMORY_UTILIZATION = 0.9`, `ANALYZER_CONTEXT_WINDOW = 32768` (the agent's prompt budget is 32k even though the server allows 64k — comment in `kaggle.py`: "Duck keeps at most 32,768 tokens of prompt-side context … 65,536 gives that lean policy enough reply headroom").

### 3.2 `--speculative-config` handling

* In this bundle: `VLLM_SPECULATIVE_CONFIG = 'off'` (rendered from `os.environ.get("VLLM_SPECULATIVE_CONFIG", "off")` at deploy time). So the flag is **not** on the argv, and `CONFIGURED_RUNTIME_ATTESTATION_SHA256 == OFF_RUNTIME_ATTESTATION_SHA256 == '8759f659a9297c4820eda1e39465c424884e2cbc019102803f0fc1993e83bb90'`.
* Fallback (top level of the script): if `start_vllm_server()` raises and spec-decode was requested, it prints `WARNING: vLLM failed to start WITH speculative decoding; retrying WITHOUT it. Cause: …`, sets `VLLM_SPECULATIVE_CONFIG = ''` and relaunches once. Their comment: "A rejected launch arg kills the process within seconds (wait_for_vllm_server probes the pid), so one clean retry without the flag costs little boot budget".
* Attestation (`verified_runtime_attestation_sha256()`) re-reads the *serialized* argv/env from `vllm-relaunch.json`, parses `--speculative-config`, and records `'speculative_config'` and `'mtp_speculative_decoding': method == 'mtp'`. It hashes the whole phenotype (dtype, kv_cache_dtype, max_model_len, max_num_seqs, tp, gpu_mem_util, generation_config, parsers, `reasoning_effort`, `preserve_thinking`, prefix caching, mm-encoder mode, mm limits, mamba cache mode, attention backend, enforce_eager, spec config, DEEP_GEMM/FLASHINFER_SAMPLER env) and fails closed if it differs from the expected hash.

### 3.3 Process management, readiness, smoke

* `subprocess.Popen(cmd, env=launch_env, stdout=log, stderr=STDOUT, text=True, start_new_session=True)` — own session so `killpg` reaps EngineCore workers. pid → `vllm-openai-server.pid`; log → `vllm-openai-server.log`.
* `vllm-relaunch.json` stores `{'cmd', 'env', 'pidfile', 'base_url', 'served_model_name', 'server_log', 'sentinel'}` so any relaunch is byte-identical.
* `wait_for_vllm_server(timeout_seconds=900)`: polls `GET /v1/models` every 5 s, requires `served_ids == ['Qwen/Qwen3.8-27B-FP8']` exactly, and aborts immediately if the pid dies (`os.kill(pid, 0)`), dumping the last 80 log lines.
* Smoke 1 (`run_vllm_api_smoke_test`): a 32×32 checkerboard PNG as `image_url`, `tools=[python]` with `tool_choice` forced to `python`, `temperature 0.0`, `max_tokens 512`, `chat_template_kwargs {'enable_thinking': True}`, timeout 300 s; must return exactly one parsed tool call whose `code` compiles. This exercises "Qwen3.8 vision preprocessing, thinking chat template, qwen3_coder parsing, and structured tool output" in one request.
* Smoke 2 (retained-reasoning receipt): sends an assistant message with `'reasoning': 'ARC3_QWEN38_REASONING_RETENTION_RECEIPT_7F3A'`, `max_tokens 1`, `enable_thinking False`, `'return_prompt_text': True`, and asserts the sentinel appears in the rendered `prompt_text` — i.e. that `preserve_thinking` actually re-renders prior `reasoning` (vLLM 0.24 supports `return_prompt_text`).

### 3.4 Watchdog / relaunch (W5)

Embedded `vllm_watchdog.py` is written to `WORKING_DIR/vllm_watchdog.py` and started detached (`start_new_session=True`) after the smokes, pid in `vllm-watchdog.pid`, log `vllm-watchdog.log` with `[w5]` prefix. Constants:

```
PROBE_INTERVAL_SECONDS = 30.0      GEN_PROBE_EVERY_TICKS = 2     HEARTBEAT_EVERY_TICKS = 20
MODELS_TIMEOUT_SECONDS = 10        GEN_PROBE_TIMEOUT_SECONDS = 240
HARD_FAIL_TRIGGER = 3              SOFT_FAIL_TRIGGER = 4
COOLDOWN_BASE_SECONDS = 600.0      COOLDOWN_STEP_SECONDS = 300.0   COOLDOWN_MAX_SECONDS = 1800.0
KILL_GRACE_SECONDS = 10.0          PORT_FREE_TIMEOUT_SECONDS = 60.0
READY_TIMEOUT_SECONDS = 900.0      READY_POLL_SECONDS = 5.0        MAX_AGE_SECONDS = 9.75 * 3600.0
```

* Hard probe = pid alive + `GET /v1/models`; soft probe (every 60 s) = a **real 1-token generation** (`"ping"`, `max_tokens 1`, `enable_thinking False`) because "a hung engine keeps /health at 200".
* 3 hard or 4 soft failures → `killpg` server group, wait port free, relaunch from the manifest into `vllm-openai-server.relaunch{N}.log`, wait ready ≤ 900 s, run gen probe, reset counters. No relaunch cap; escalating cooldown 600 s + 300 s·N (max 1800 s).
* Teardown order (`teardown_commands.json`): write `vllm-teardown.sentinel` → SIGTERM watchdog (≤30 s then SIGKILL) → re-read server pidfile → SIGTERM/`killpg` SIGKILL server → tail the watchdog log (120 lines) and any relaunch logs into the notebook output → delete `vllm-relaunch.json` (it contains the full env) → `rmtree(vllm-site-packages)`.
* Motivation quoted from the watchdog docstring: "engine death -> connection-refused storm (observed: the v4 spec-decode pod crash zeroed a commit; solver threads spin-retry forever at ~1s)" and "silent deadlock: FP8 + prefix-caching + concurrent multimodal load can zero throughput while the process stays alive and /health stays 200".

### 3.5 A latent bug in the shipped script

The rendered `setup_commands.json` (and the template in `kaggle.py` line 1477) executes, at top level after both smokes:

```python
for source_key, source_digest in source_bundle_receipt.items():
    if re.fullmatch(r'[0-9a-f]{64}', source_digest) is None:
```

but the import block is `hashlib, json, os, shutil, subprocess, sys, sysconfig, time, urllib.request, pathlib.Path` — **`re` is never imported** (AST check: `'re' imported? False ; 're' referenced? True`). As shipped, this bundle would boot the server, pass both smokes, then die with `NameError: name 're' is not defined` before the boot receipt, and the notebook would run teardown. Either their `apex` working tree (marked DIRTY) fixed it after the push, or this kernel never played a game. Don't copy the tail of that script verbatim.

---

## 4. MTP / speculative-decoding evidence, and what "mc-r4" / "trt" mean

### 4.1 Speculative decoding: what the bundle actually contains

**No measurement of MTP, acceptance rate, or tok/s exists anywhere in the bundle.** Grep over all of `src/` for `acceptance|tok/s|tokens/s|num_speculative|"mtp"` hits only:

* `kaggle.py` lines 382-396 (the only run history, about **ngram**, not MTP):

  > `# L3 ngram speculative decoding — DEFAULT OFF as of 2026-07-20.`
  > `# FALSIFIED ON THE POD: the ngram config below (num_speculative_tokens=3) booted and passed the single-request smoke test, then the vLLM engine DIED under real 28-way multimodal load — a connection-refused storm on port 1234 that zeroed the commit … num_speculative_tokens=4 had already produced EngineDeadError in the v3 probes; 3 is now falsified too. Shipping OFF returns the vLLM launch to the byte-identical proven-stock argv (the 1.49 / 1.07 / 1.77 runs used no spec-decode). The knob is preserved for a RENTED-GPU A/B only (never a blind pod commit). To re-enable, set the env var to the exact ngram value:`
  > `#   VLLM_SPECULATIVE_CONFIG='{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_min":8,"prompt_lookup_max":10}'`
  > `# prompt_lookup_min=8 stays MANDATORY (the vLLM default of 2 speculates inside the qwen3_coder tool-call XML and corrupts a large fraction of tool calls).`

* `kaggle.py` line 133-141 `_uses_mtp_speculative_decoding()` — recognises `{"method": "mtp", …}` and records `mtp_speculative_decoding` in the attestation, so an MTP arm was *anticipated*, never run.
* `metacontrol-turn-contract.json` `_campaign.note`: "Pair only against a fresh current-stack control; **spec-decode OFF mandatory**."
* `diagnostics.py` line 2300 logs `generated tokens/sec: … (job wallclock)` — a generic throughput line, not spec-decode telemetry.

Bottom line: the only field data is that **ngram** spec-decode on this exact stack (0.24.0, FP8, 28 concurrent multimodal streams, `max_num_seqs 32`) killed the engine mid-run; MTP is untested by them. The watchdog exists because of that crash.

### 4.2 Naming

From `metacontrol_gate.py`:

```python
R4_REPLICATE_ID = "GPU-METACTRL-R4-P25-S20260829-01"
R4_ASSIGNMENT_SEED = 20260829
R4_KERNEL_IDS = {"control": "jacquesbuis/arc3-mc-r4-ctrl-p25-s20260829",
                 "treatment": "jacquesbuis/arc3-mc-r4-trt-p25-s20260829"}
R4_DATASET_IDS = {"control": "jacquesbuis/taaf-src-mc-r4-ctrl-s20260829",
                  "treatment": "jacquesbuis/taaf-src-mc-r4-trt-s20260829"}
```

* `mc` = **metacontrol** (hypothesis `H-METACONTROL-001`), `r4` = **replicate 4** of a matched control/treatment pair, `trt` = **treatment** arm, `ctrl` = control, `p25` = the 25 public games, `s20260829` = assignment seed/date.
* Treatment phenotype (`boot_arm = 'treatment'` when `control_memory_enabled and turn_contract_enabled`): `LOCAL_CONTROL_MEMORY=1` with `_REQUIRED/_REQUIRE_FIELDS/_RETRACT=true` (a required structured `memory` object with `goal` + `falsified` in every python tool call, enforced by the xgrammar schema and smoke-tested at boot), `LOCAL_ANALYZER_TURN_CONTRACT=true` (`_MAX_TOKENS 512`), `LOCAL_ANALYZER_OUTPUT_BUDGET=true` (fraction 0.8, 256..1024), `LOCAL_ANALYZER_TOOL_STEPS=2`, `LOCAL_ANALYZER_YIELD_SECONDS=180`, `LOCAL_ANALYZER_SEED=20260829`. Config arm label: `"metacontrol-e1a_fields-e1b-e2-e3"`. **"trt" has nothing to do with TensorRT.**

---

## 5. What our 0.19 `setup_commands.json` does differently (diff-style)

```
- WHEELHOUSE = driessmit1/arc3-vllm-h100-wheelhouse-v3   (wheels at dataset root, 'requirements.lock')
+ WHEELHOUSE = jcole75/arc3-qwen36-runtime-wheels @v3     (wheels/ subdir, 'requirements-runtime.txt', SHA-pinned manifest)
- MODEL      = driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot  served as 'vrfai/Qwen3.6-27B-FP8'
+ MODEL      = saltb0x/qwen3-8-27b-fp8 (81-file official mirror incl. mtp.safetensors) served as 'Qwen/Qwen3.8-27B-FP8'
- STAMP_TEXT = 'vllm==0.19.0 torch==2.10.0 flashinfer==0.6.6\n'
+ STAMP_TEXT = 'dataset=jcole75/arc3-qwen36-runtime-wheels@3 vllm==0.24.0 torch==2.11.0+cu129 transformers==5.13.0 flashinfer==0.6.12 model=Qwen/Qwen3.8-27B-FP8\n'

  pip install --no-index --find-links <wh> --requirement <req> --target <sp> --upgrade --ignore-installed --only-binary :all: --no-compile --disable-pip-version-check --no-warn-conflicts
+   … --no-cache-dir                              (theirs adds this; otherwise identical flags)
+ disk preflight (13.25 GB + 2 GiB), 197-wheel SHA verify, importlib.metadata version assert, nvcc/headers/libcudart.so.13 assert
- (ours: none of the above; cached-install check is `import vllm, torch`)

  vllm_env():
    USE_TF=0 TRANSFORMERS_NO_TF=1 TRANSFORMERS_NO_TORCHVISION=1 VLLM_NO_USAGE_STATS=1 PYTHONPATH=<sp>:…   (both)
+   CUDA_HOME/CUDA_PATH=<sp>/nvidia/cu13, PATH+=<cu13/bin>:<sp>/bin, LD_LIBRARY_PATH/LIBRARY_PATH+=<cu13/lib>:/usr/local/nvidia/lib64, CPATH+=<cu13/include>
+   FLASHINFER_NVCC, FLASHINFER_EXTRA_LDFLAGS, FLASHINFER_WORKSPACE_BASE, TORCH_EXTENSIONS_DIR, TRITON_CACHE_DIR, XDG_CACHE_HOME -> /kaggle/temp/duck-vllm-cache
+   VLLM_USE_FLASHINFER_SAMPLER=0  VLLM_USE_DEEP_GEMM=0  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  HF_HUB_OFFLINE=1  PYTHONDONTWRITEBYTECODE=1
+   env.pop('VLLM_ATTENTION_BACKEND')

  argv (both): --model --served-model-name --host 127.0.0.1 --port 1234 --tensor-parallel-size 1 --enable-auto-tool-choice --tool-call-parser qwen3_coder --generation-config vllm --enable-prefix-caching --reasoning-parser qwen3 --max-model-len 65536
+   --dtype bfloat16  --max-num-seqs 32  --gpu-memory-utilization 0.9  --mm-encoder-tp-mode data  --limit-mm-per-prompt '{"image": 1, "video": 0}'  --mamba-cache-mode align
-   --default-chat-template-kwargs '{"preserve_thinking": true}'
+   --default-chat-template-kwargs '{"preserve_thinking": true, "reasoning_effort": "xhigh"}'
+   optional --speculative-config <VLLM_SPECULATIVE_CONFIG> with one retry without it  (ours: no spec-decode path at all)

- Popen(cmd, env, stdout=log, stderr=STDOUT)                       ; readiness = any 200 from /v1/models
+ Popen(…, start_new_session=True) + vllm-relaunch.json           ; readiness = served ids == ['Qwen/Qwen3.8-27B-FP8'] exactly, pid-death abort
- smoke: text-only "what is 2 + 2?", enable_thinking False, 96 tokens
+ smoke: image + forced python tool call + nested memory schema, thinking on; second request checks preserve_thinking via return_prompt_text
+ watchdog (real-generation liveness probe, byte-identical relaunch), boot receipt JSON, attestation hash
+ teardown: sentinel → kill watchdog → killpg server → dump logs → rm relaunch manifest → rmtree site-packages   (ours: no teardown file in this bundle dir)

  setup_env differences: TOOL_STEPS '0'→'2', YIELD_SECONDS '60'→'180', + LOCAL_ANALYZER_SEED '20260829', LOCAL_ANALYZER_TIMEOUT '900', LOCAL_CONTROL_MEMORY*, OUTPUT_BUDGET*, TURN_CONTRACT*, LOCAL_A6GF_*, LOCAL_CAP_DEFAULT '4096', LOCAL_ESC_*, VLLM_SPECULATIVE_CONFIG 'off', LOCAL_RUNTIME_STACK_ID, LOCAL_RUNTIME_ATTESTATION_SHA256, LOCAL_MODEL_REVISION, and the CUDA/cache env vars re-exported for the analyzer process
```

Our failed lab2 attempt, for contrast, used `saltb0x/arc3-vllm-wheelhouse-v0271-cu129` with `pip install --no-index --find-links <wh> --target /tmp/vllm-site-packages-0271 vllm==0.27.1 flashinfer-python flashinfer-cubin` (no requirements file), `CUDA_HOME=<sp>/nvidia/cuda_nvcc` (the cu12 wheel with no `bin/nvcc`), `TORCH_CUDA_ARCH_LIST=12.0+PTX`, `FLASHINFER_CUDA_ARCH_LIST=12.0f`, and the FlashInfer sampler left on for the first boot.

---

## 6. Risks, and a minimal Kaggle cell for vLLM 0.24 + prefix caching + MTP(2) on `foysalemonshanto/qwen3-8-27b-fp8-repacked-v1`

### 6.1 Concrete risks

1. **The pins do not exist in the saltb0x 0.27.1 wheelhouse.** `saltb0x/arc3-vllm-wheelhouse-v0271-cu129` `requirements.lock` has `vllm==0.27.1`, `torch==2.13.0` (no `+cu129` local tag), `transformers==5.15.0`, `flashinfer-python==0.6.16.post3`, `flashinfer-cubin==0.6.16.post3`, `nvidia-cuda-nvcc-cu12==12.9.86`, `nvidia-cuda-nvdisasm==13.3.73` and **no** `nvidia-cuda-nvcc==13.3.*`, `nvidia-cuda-runtime==13.3.*`, `nvidia-cuda-crt`, `nvidia-cuda-cccl==13.3.*`, `nvidia-nvvm`. It also ships a DFlash2 source overlay (`ovl_manifest.json` → `vllm/v1/worker/gpu/spec_decode/dflash2/…`, `vllm/model_executor/models/qwen3_dflash2.py`, `vllm/config/vllm.py`). So the 0.24 recipe must use the **jcole75** wheelhouse as a whole; do not try to graft the cu13 toolchain onto 0.27.1 (torch 2.13 vs nvcc 13.3 untested).
2. **Driver / CUDA-13 runtime.** FlashInfer kernels JIT-compiled with nvcc 13.3 link `libcudart.so.13`, which needs an R580-class driver; torch+cu129 needs an R575-class driver. Neither of our lab logs prints the driver version. Their code comments say this stack "ran cleanly on the 1.49 / 1.07 / 1.77 commits" but that is second-hand. **First cell must print `nvidia-smi` driver version** and treat < 580 as "expect JIT-linked kernels to fail at load"; with `VLLM_USE_FLASHINFER_SAMPLER=0` and `FLASH_ATTN` auto-selected, nothing should need the JIT at all, which is the real safety net.
3. **sm_120 support** is in the cu129 torch wheel and in `flashinfer_cubin` 0.6.12; the 13.3 nvcc exists only for on-demand JIT. Leave `TORCH_CUDA_ARCH_LIST` unset (their launch does), so the image's pre-sm75 value is *not* inherited — actually it is inherited from `os.environ.copy()`; if the image exports one, pop it or set `12.0` explicitly (our v1 death).
4. **Python/ABI.** All wheels are `cp312` / `manylinux_2_28`; the Kaggle image is Python 3.12 (our logs show `/usr/lib/python3.12`). Fine; would break on a 3.13 image.
5. **Disk.** 13.25 GB install target plus 6.5 GB of wheels already on the read-only mount; they require 15.25 GB free in `/kaggle/working` and push all compiler/Triton/FlashInfer caches to `/kaggle/temp`. Our lab used `/tmp`; either is fine, but do the preflight.
6. **MTP + prefix caching is against our standing rule.** `docs/RESEARCH-2026-08-21-bug-lever-hunt.md` lines 152-155: "**MTP LANDMINE CONFIRMED RELEVANT**: cache hits occur, and MTP+prefix-cache on GDN corrupts generations (fix PR #47861 is post-0.19) — **MTP ships only with `--no-enable-prefix-caching`**." Whether PR #47861 is in 0.24.0 is **unverified** (no vLLM source available offline). Also from the handoff: "FP8+MTP crash (vllm issue #40756); quantized-head acceptance collapse (5–11%); community data says speedup collapses at concurrency 28 → pair with concurrency 8–16." The recipe below therefore boots MTP **with** prefix caching as asked, but the smoke must include a tool-call round-trip and a `/metrics` read of the spec-decode acceptance counters; if either is bad, re-boot with `--no-enable-prefix-caching`.
7. **Their only spec-decode field result is an engine death** under 28-way multimodal load (ngram, nst=3; nst=4 gave `EngineDeadError`). Their watchdog exists for that. If we run MTP live, ship a watchdog or at least `start_new_session=True` + a relaunch loop.
8. **`mamba-cache-mode align`** is on their argv and part of their attested phenotype; it is the mode that makes prefix caching valid for the GDN/Mamba layers of Qwen3.5-family hybrids. Keep it whenever `--enable-prefix-caching` is on.
9. **Model mount shape.** The repack is a Kaggle *model* (`model_sources`), mounted at `/kaggle/input/models/foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1` (from our v1/v2 logs). Their `resolve_kaggle_dataset_path()` would not find it; hard-code the path. The repack's shard layout differs from saltb0x's (16 × `model-layers-000NN-of-00016.safetensors` + `outside.safetensors` + `mtp.safetensors`, `model.safetensors.index.json` 159,995 bytes vs 137,335), so **verify the index maps the `mtp.*` tensors to `mtp.safetensors`**; if it doesn't, vLLM loads the model fine and silently has no draft head. `config.json` (identical size 51,350 in both) carries `text_config.mtp_num_hidden_layers: 1`, `mtp_use_dedicated_embeddings: False`, which is what `method: "mtp"` needs.
10. **`reasoning_effort: "xhigh"`** is their choice, and our own 08-21 hunt flagged an "xhigh dead-decode cluster (~6h)". Not a boot risk; a policy choice — keep ours (`preserve_thinking` only) unless testing that axis.
11. **Version drift of the wheelhouse dataset.** Their script pins v3 by SHA and fails closed. If we attach without pinning the dataset version in the notebook, a future jcole75 bump would change the stack silently; cheap protection is the `STAMP_TEXT` + `importlib.metadata` assert (copy it).
12. **Copying their tail verbatim reproduces the `NameError` (section 3.5).** Add `import re` or drop the receipt block.

### 6.2 Minimal recipe (one Kaggle notebook cell; RTX PRO 6000, internet off)

Attach: dataset `jcole75/arc3-qwen36-runtime-wheels` (v3), model `foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1`, competition `arc-prize-2026-arc-agi-3`. Everything below is lifted from the bundle's proven pieces plus the MTP flag.

```python
import json, os, shutil, subprocess, sys, sysconfig, time, urllib.request
from pathlib import Path

WH   = Path('/kaggle/input/arc3-qwen36-runtime-wheels')          # or /kaggle/input/datasets/jcole75/arc3-qwen36-runtime-wheels
MODEL = Path('/kaggle/input/models/foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1')
SP   = Path('/kaggle/working/vllm-site-packages')                 # 13.25 GB; needs >= 15.25 GB free
CACHE = Path('/kaggle/temp/duck-vllm-cache'); CACHE.mkdir(parents=True, exist_ok=True)
CUDA_HOME = SP / 'nvidia' / 'cu13'
SERVED = 'Qwen/Qwen3.8-27B-FP8'; BASE = 'http://127.0.0.1:1234/v1'

# 0. facts we need before trusting anything (driver >= 580 for CUDA-13-linked JIT kernels)
print(subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv'], capture_output=True, text=True).stdout)
assert shutil.disk_usage('/kaggle/working').free >= 13_250_838_162 + 2 * 1024**3

# 1. offline install, exactly their flags
if not (SP / '.arc3-qwen36-runtime-wheels').exists():
    shutil.rmtree(SP, ignore_errors=True); SP.mkdir(parents=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index',
                    '--find-links', str(WH / 'wheels'), '--requirement', str(WH / 'requirements-runtime.txt'),
                    '--target', str(SP), '--upgrade', '--ignore-installed', '--only-binary', ':all:',
                    '--no-compile', '--no-cache-dir', '--disable-pip-version-check', '--no-warn-conflicts'], check=True)
    (SP / '.arc3-qwen36-runtime-wheels').write_text('vllm==0.24.0 torch==2.11.0+cu129 transformers==5.13.0 flashinfer==0.6.12\n')

# 2. their env: CUDA 13.3 toolchain from the wheelhouse, FlashInfer sampler + DeepGEMM off, caches on /kaggle/temp
env = os.environ.copy()
for k in ('VLLM_ATTENTION_BACKEND', 'TORCH_CUDA_ARCH_LIST'):   # do not inherit the image's pre-sm75 arch list (our v1 death)
    env.pop(k, None)
env['PYTHONPATH'] = f"{SP}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
env.update({'CUDA_HOME': str(CUDA_HOME), 'CUDA_PATH': str(CUDA_HOME),
            'FLASHINFER_NVCC': str(CUDA_HOME / 'bin' / 'nvcc'),
            'FLASHINFER_EXTRA_LDFLAGS': f'-L{CUDA_HOME / "lib"} -L/usr/local/nvidia/lib64'})
for key, vals in {'PATH': (CUDA_HOME / 'bin', SP / 'bin'),
                  'LD_LIBRARY_PATH': (CUDA_HOME / 'lib', Path('/usr/local/nvidia/lib64')),
                  'LIBRARY_PATH': (CUDA_HOME / 'lib', Path('/usr/local/nvidia/lib64')),
                  'CPATH': (CUDA_HOME / 'include',)}.items():
    env[key] = os.pathsep.join([*(str(v) for v in vals), *([env[key]] if env.get(key) else [])])
env.update({'USE_TF': '0', 'TRANSFORMERS_NO_TF': '1', 'TRANSFORMERS_NO_TORCHVISION': '1',
            'PYTHONDONTWRITEBYTECODE': '1', 'VLLM_NO_USAGE_STATS': '1', 'HF_HUB_OFFLINE': '1',
            'VLLM_USE_FLASHINFER_SAMPLER': '0', 'VLLM_USE_DEEP_GEMM': '0',
            'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
            'XDG_CACHE_HOME': str(CACHE), 'TORCH_EXTENSIONS_DIR': str(CACHE / 'torch-extensions'),
            'TRITON_CACHE_DIR': str(CACHE / 'triton'), 'FLASHINFER_WORKSPACE_BASE': str(CACHE / 'flashinfer')})

# 3. their toolchain gate
for p in (CUDA_HOME / 'bin/nvcc', CUDA_HOME / 'include/cuda_runtime.h', CUDA_HOME / 'include/cuda.h',
          CUDA_HOME / 'lib/libcudart.so.13', Path(sysconfig.get_path('include')) / 'Python.h'):
    assert p.is_file(), p
assert 'release 13.3' in subprocess.run([str(CUDA_HOME / 'bin/nvcc'), '--version'], env=env, capture_output=True, text=True).stdout
probe = subprocess.run([sys.executable, '-c', "import importlib.metadata as m; print({n: m.version(n) for n in ('vllm','torch','transformers','flashinfer-python')})"], env=env, capture_output=True, text=True); print(probe.stdout)
# also confirm the repack's index actually points MTP tensors at mtp.safetensors, else there is no draft head
idx = json.loads((MODEL / 'model.safetensors.index.json').read_text())['weight_map']
assert any(v == 'mtp.safetensors' for v in idx.values()), 'repack index does not reference mtp.safetensors'

# 4. launch = their argv + MTP(2); keep --mamba-cache-mode align with prefix caching
def argv(spec):
    a = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
         '--model', str(MODEL), '--served-model-name', SERVED, '--host', '127.0.0.1', '--port', '1234',
         '--tensor-parallel-size', '1', '--dtype', 'bfloat16', '--max-num-seqs', '32', '--gpu-memory-utilization', '0.9',
         '--mm-encoder-tp-mode', 'data', '--limit-mm-per-prompt', '{"image": 1, "video": 0}', '--mamba-cache-mode', 'align',
         '--enable-auto-tool-choice', '--tool-call-parser', 'qwen3_coder', '--generation-config', 'vllm',
         '--enable-prefix-caching', '--default-chat-template-kwargs', '{"preserve_thinking": true}',
         '--reasoning-parser', 'qwen3', '--max-model-len', '65536']
    return a + (['--speculative-config', spec] if spec else [])

def boot(spec, log):
    p = subprocess.Popen(argv(spec), env=env, stdout=open(log, 'w'), stderr=subprocess.STDOUT, text=True, start_new_session=True)
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if p.poll() is not None:
            raise RuntimeError(f'server died rc={p.returncode}; tail:\n' + ''.join(open(log).readlines()[-60:]))
        try:
            ids = [m['id'] for m in json.load(urllib.request.urlopen(f'{BASE}/models', timeout=5))['data']]
            if ids == [SERVED]: return p
        except Exception: time.sleep(5)
    raise TimeoutError(log)

MTP = json.dumps({"method": "mtp", "num_speculative_tokens": 2})
try:
    server = boot(MTP, '/kaggle/working/vllm-openai-server.log')
except Exception as e:                       # their fallback: never let the spec flag zero the run
    print('WARNING: MTP boot failed, retrying without it:', repr(e)); server = boot('', '/kaggle/working/vllm-openai-server.nospec.log')

# 5. smoke: forced tool call with an image (their smoke), then read spec-decode counters from /metrics
req = lambda payload, t: json.load(urllib.request.urlopen(urllib.request.Request(f'{BASE}/chat/completions', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'}), timeout=t))
r = req({'model': SERVED, 'messages': [{'role': 'user', 'content': "Call the python tool with code exactly print('ready'). Do not answer in prose."}],
         'tools': [{'type': 'function', 'function': {'name': 'python', 'parameters': {'type': 'object', 'properties': {'code': {'type': 'string'}}, 'required': ['code']}}}],
         'tool_choice': {'type': 'function', 'function': {'name': 'python'}}, 'temperature': 0.0, 'max_tokens': 512,
         'chat_template_kwargs': {'enable_thinking': True}}, 300)
print(r['choices'][0]['message'].get('tool_calls'))
metrics = urllib.request.urlopen('http://127.0.0.1:1234/metrics', timeout=20).read().decode()
print('\n'.join(l for l in metrics.splitlines() if 'spec_decode' in l and not l.startswith('#')))   # num_drafts / num_draft_tokens / num_accepted_tokens
```

What to read in the server log before trusting the boot: the `Initializing a V1 LLM engine (v0.24.0) with config: … speculative_config=SpeculativeConfig(method='mtp', …)` line (if it says `speculative_config=None` the flag was dropped), the attention-backend line (expect `FLASH_ATTN`), and no `flashinfer.jit` frames anywhere. Acceptance from `/metrics`: `vllm:spec_decode_num_accepted_tokens_total / vllm:spec_decode_num_draft_tokens_total`; below ~0.5 per position on a bf16 KV, FP8 weights run, the 08-21 "quantized-head acceptance collapse (5-11%)" warning applies and MTP is not worth the engine-death exposure.

Things deliberately **not** copied from their stack: `reasoning_effort: "xhigh"`, the control-memory/turn-contract analyzer env, the 197-wheel + 81-file SHA sweep (minutes of boot budget; keep the version assert instead), and the boot-receipt block with the missing `import re`.
