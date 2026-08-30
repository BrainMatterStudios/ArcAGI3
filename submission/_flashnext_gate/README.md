# arc3-flashnext-gate — serve sonpham's Flash-Next NVFP4 on ONE Kaggle RTX Pro 6000

Built 2026-08-30 on `winning/duck-patched`. NO games, NO submission, one GPU
commit, budget <= 2.5 h. Question: **can the sonphamorg Flash-Next NVFP4
serving package boot on one 96 GB RTX Pro 6000, and how fast is it on
duck-shaped load vs our Qwen3.8-27B-FP8 baseline?**

## Pre-registered decision rules (fixed BEFORE the run)

* **GATE PASS** = boots on 1 GPU (either rung) **AND** stable conc-28 gen
  tok/s aggregate **>= 450** (>= ~1.5x the 27B's 296.8, lab3 2026-08-26, same
  GPU pool) **AND** tool-call parse rate **>= 0.95** (qwen3_xml, 20-prompt
  battery: 10 forced `tool_choice` + 10 instructed-auto; attempts = forced +
  auto responses that emitted or tried to emit a call).
* Boot only on the **reduced rung** (util 0.92 / 16k ctx / 12 seqs) = boot
  PASS with a **DEGRADED-CONFIG** flag: their exact GCP-parity config did not
  fit, and any adoption plan must re-derive capacity numbers.
* Spec-decode counters are recorded but expected **ZERO**: their package
  ships `FLASHNEXT_MTP_ENABLED=0` (no `--speculative-config`).
* 27B reference numbers (lab3, vLLM 0.19, 27B-FP8, prompt ~21.5k):
  conc-1 40.6 tok/s / 2435 t/m/sess; conc-8 215.9 / 1619.4;
  conc-28 296.8 / 636.1. Caveat: lab3's session shape is closest to our
  "stable" regime; the optional churn conc-28 arm is reported against the
  same conc-28 row for context only.

## What their package actually is (decoded from part-A `source-bundle/`)

Sources read on 2026-08-30: `kaggle_flashnext_setup.py`,
`kaggle_flashnext_preconverted_setup.py`, `setup_commands.json`,
`FLASHNEXT_PACKAGE_LOCK.json`, `FLASHNEXT_GCP_MODEL_INFO.json`,
`container-requirements.freeze`, `qualification-notes.md`, the three dataset
READMEs, and the HF model card.

* **Engine: pinned vLLM, NOT SGLang.** The runtime tarball
  (`flashnext-gcp-container-site-packages.tar.zst`, 6 GB, sha
  `c06a78d5...`) is the extracted site-packages of container
  `vllm/vllm-openai@sha256:fc120ece...` carrying dev wheel
  **`vllm 0.1.dev20073+g8e685d198`** + torch 2.13.0+cu130 + transformers
  5.15.1 (no sglang anywhere in the freeze). The model card's SGLang
  `qwen4_exp` flags apply to the upstream HF checkpoint only.
* **Single-GPU mechanism: `--tp 1` + PLE CPU offload.** Their `serving_env()`
  sets **`VLLM_PLE_CPU_OFFLOAD=1`** (+ `VLLM_PLE_OFFLOAD_READY_TIMEOUT=1800`):
  the ten `model-plebf16-*.safetensors` PLE n-gram tables (~104 GB BF16,
  10.4 GB each) stay in host RAM (~177 GB available), leaving ~82 GB of the
  186 GB payload GPU-resident under `--gpu-memory-utilization 0.96`.
  `FLASHNEXT_PACKAGE_LOCK.json` records this exact shape (tp1, 22 seqs, 32k
  ctx, kv auto/BF16, `ple_cpu_offload: true`, `mtp_enabled: false`, 28 client
  workers, 7920 s/game) as the config behind their GCP mean **9.63**.
  So single-GPU serving is their design, not our gamble — the kernel is a
  verification, not a NO-GO-by-construction case.
* **Their launch argv (copied verbatim into the kernel):**
  `--tensor-parallel-size 1 --distributed-executor-backend mp
  --gpu-memory-utilization 0.96 --max-model-len 32768 --max-num-seqs 22
  --max-num-batched-tokens 6144 --kv-cache-dtype auto --enable-prefix-caching
  --no-enable-flashinfer-autotune --enable-auto-tool-choice
  --tool-call-parser qwen3_xml --generation-config vllm
  --default-chat-template-kwargs '{"preserve_thinking": true}'
  --reasoning-parser qwen3`
* **Model layout.** Their preconverted wrapper verifies a **private** Kaggle
  MODEL (`sonphamorg/qwen3-8-flash-next-plebf16-gcp-exact` — 403 for us), so
  this kernel assembles the identical flat view by **symlink-union** of the
  three public datasets' `serving-part-000/001/002` dirs (206 safetensors,
  186.5 GB, zero copies), copies `model.safetensors.index.json` as a real
  file, asserts index coverage, and replays their PLE-conversion provenance
  check against `FLASHNEXT_GCP_MODEL_INFO.json` (sizes + manifest hashes; the
  104 GB is not re-hashed). Extraction uses **their bundled `zstd`**
  (sha-pinned) because the Kaggle image lacks the CLI.
* The part-A source-bundle also reveals the operator: the TAAF/duck-harness
  stack (`benchmark.label: duck-harness-kaggle`, concurrency 28,
  `ARC3-Inference` @ aa69123) — same loop shape as ours, bigger model.

## Kernel phases (all failure-wrapped; partial results always in `/kaggle/working/flashnext_gate_results.json`)

0. nvidia-smi driver FIRST, df -h / free -g, mount tree, RTX-Pro-6000
   fail-fast assert.
1. ASSEMBLE: sha-verify + extract runtime (to the biggest scratch mount, NOT
   /kaggle/working), pinned-version import check
   (`0.1.dev20073+g8e685d198 2.13.0+cu130 5.15.1 13.0`), symlink-union model
   view, index coverage, PLE provenance.
2. BOOT: their argv + env verbatim; 1800 s readiness (their number); on
   failure one pre-registered retry rung (0.92 / 16k / 12 seqs); logs the
   PLE/offload + KV-cache engine lines, GPU + host RAM after boot.
3. SMOKE: `/v1/models`; 3-prompt parser round-trip; one-image multimodal
   probe (failure flips images off in the generator instead of dying);
   20-prompt tool battery (the parse gate); 20-prompt greedy battery.
4. LOAD (lab1/lab4 duck-shaped generator, ~17k-token seeded sessions, one
   board PNG per newest turn, 1-3k gen tokens): stable conc-1 probe /
   conc-8 / conc-28; churn conc-28 if elapsed < 100 min. Reports gen tok/s
   aggregate, tok/min/session, window prefix-hit rate, KV usage,
   preemptions, spec-decode counters.
5. VERDICT: pre-registered gate + side-by-side table vs the 27B baseline.

## License (prize-eligibility note)

Part A ships `LICENSE` = **Qwen Community License 1.0** (Qwen3.8-Flash-Next
is the base model; the NVFP4 checkpoint inherits it):

> "Permission is hereby granted, free of charge, to any person obtaining a
> copy of this software, including the model weights, parameters,
> configuration files, inference code and associated documentation files
> (collectively, the 'Software'), to deal in the Software without
> restriction, including without limitation the rights to use, copy, modify,
> merge, publish, distribute, sublicense, sell, deploy, host, fine-tune, and
> create derivative works from ... copies of the Software"

Conditions: attribution notice; name display only for products with >100M MAU
or >US$20M monthly revenue; a separate license only for "Model as a Service"
/ "AI Work Assistant" **businesses** (internal use exempt). None of these
blocks redistribution for an open-sourced competition submission, so the
package is compatible with ARC Prize's reproducibility/open-source
requirement. `NOTICE.txt` carries the required attribution.

## Open risks (recorded before the run)

1. **Python ABI**: the runtime is cp312; if the Kaggle image's notebook
   python is not 3.12 the import check fails immediately (clear, cheap
   signal — phase 1 error, no GPU time burned).
2. **Driver**: their container targets CUDA 13.0; the pool's R580 driver
   (580.159.04 on 08-26) satisfies it, but a driver rehome below R580 would
   surface at import/boot.
3. **Boot time**: 186 GB weight read from the dataset mounts + 104 GB PLE
   staging to host RAM; the 1800 s readiness window is their own number but
   a cold/slow mount could still blow it (recorded as BOOT TIMEOUT with the
   log tail).
4. **/v1 parity**: `qwen3_xml` tool parser + `preserve_thinking` are their
   flags; if the vLLM-dev build's parser behaves differently under our
   OpenAI-style payloads, the tool battery measures exactly that (that is
   the point of the gate).
5. **Regime comparability**: lab3's baseline session shape is append-mostly
   (closest to "stable"); the churn arm here is context, not gate input.

## Files

* `build_flashnext_gate.py` — builder (reuses lab1's load-generator via the
  lab4 block-replacement recipe; run it to regenerate the notebook).
* `arc3-flashnext-gate.ipynb` — generated kernel (8 cells).
* `kernel-metadata.json` — `ahmedmobasher86/arc3-flashnext-gate`,
  NvidiaRtxPro6000, internet OFF, four sonphamorg datasets +
  `arc-prize-2026-arc-agi-3` competition source (the RTX-pool gate),
  no model_sources (their Kaggle MODEL is private).
* `validate_flashnext_gate.py` — structure/syntax/invariant/freshness checks.
* `FLASHNEXT_GCP_MODEL_INFO.json` — reference copy of their PLE-conversion
  provenance (the kernel reads the mounted original at runtime).

## Push (DO NOT run without Ahmed's go — costs a GPU commit, not a slot)

```sh
kaggle kernels push -p submission/_flashnext_gate
# then watch:  kaggle kernels status ahmedmobasher86/arc3-flashnext-gate
```
