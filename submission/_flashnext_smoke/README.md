# arc3-flashnext-smoke — STOCK duck × Flash-Next NVFP4 (the model-swap read)

**The decisive model-axis experiment (2026-08-30).** The stock duck harness
(anim-20260807 bundle — the bytes that flew 1.55) plays all 25 public
ARC-AGI-3 games at eval geometry, served by sonpham's **Qwen3.8-Flash-Next
NVFP4** (~180B MoE, NVFP4 experts + PLE-CPU-offload) on ONE Kaggle RTX Pro
6000, instead of our Qwen3.8-27B-FP8. Same harness, same games, same
geometry, same GPU class — only the model changes.

## What runs

| Piece | Source | Status |
|---|---|---|
| ASSEMBLE (runtime tarball + cu13 nvcc + symlink-union model view) | `submission/_flashnext_gate/build_flashnext_gate.py`, reused verbatim via runpy | booted on Kaggle v4 in 740 s (`_flashnext_gate/results_v4/`) |
| BOOT (their launch argv, `VLLM_PLE_CPU_OFFLOAD=1`, ladder gcp_exact → arch120a → reduced) | same gate build | v4 boot landed on rung `gcp_exact`, first try |
| Game cells (GAMES_25, bm.run @ 7,920 s/game, conc 28, per-phase /metrics telemetry) | `submission/_tp_smoke/build_tp_smoke.py` pattern | flew twice on the GPU (arc3-tp-smoke, arc3-tp2-smoke) |

ONE phase: `("flashnext", GAMES_25, 7920)`. `soft_end = NOTEBOOK_START_EPOCH
+ 11,400 s` (assemble+boot ~25 min + one 2.2 h wave + teardown ≤ 3.2 h).

## Harness: STOCK

* No grafts (no `graft_*` embedded or installed; builder asserts it).
* `ONLY_RESET_LEVELS=true` set before the deploy pkls load (v12 cell kept
  verbatim).
* Sampling untouched: temp 0.6 / top-p 0.95 / top-k 20 / thinking on / yield
  60 s / tool steps unlimited — the exact setup_env the 27B bundle exports.
* Concurrency 28 comes from the serialized solver. Their server runs
  `--max-num-seqs 22`, so requests queue by design; a 60 s sampler records
  `num_requests_running/waiting` and `kv_cache_usage_perc` into the phase
  record.

## Wiring changes vs the 27B kernel (the ONLY deltas)

* 27B Kaggle Model NOT attached (`model_sources: []`); the v12 model-mount
  assert cell and the `setup_commands.json` vLLM-boot cell are replaced by the
  gate's assemble/boot cells. Bundle discovery, sys.path, deploy-pkl and run
  cells are kept verbatim.
* Analyzer env exported by the notebook itself — the 27B bundle's setup_env
  key-for-key, except:
  * `LOCAL_ANALYZER_BASE_URL=http://127.0.0.1:1234/v1`,
    `LOCAL_ANALYZER_MODEL_ID=INFERENCE_ANALYZER_MODEL=RadixArk/Qwen3.8-Flash-Next-NVFP4`
    (their FLASH_SERVE_FLAGS port + served name), `LOCAL_ANALYZER_PROVIDER=vllm`.
  * **DEVIATION:** `LOCAL_ANALYZER_CONTEXT_WINDOW=24576` +
    `LOCAL_ANALYZER_MAX_OUTPUT=4096`. Their server is `--max-model-len 32768`;
    the duck's default (32768 window, 512+512 reserve, **no max_tokens sent**)
    could push prompt+completion past the server cap. 24576 gives a prompt
    budget of 19,968 tokens (verified against the bundle's ToolAgent:
    `_context_budget_tokens=19968`, `max_tokens=4096` now sent) — worst case
    ~23.5k real prompt + 4k output < 32k. The 27B baseline ran a 32k window on
    a 65k server; fair enough — the stock 27B's effective history is 4-9 turns
    anyway.
  * `PYTHONPATH` NOT exported into the notebook (the 27B flow exported its
    vLLM site-packages). The cu130 serving runtime must not shadow the
    harness deps; the harness talks HTTP via `requests`, and only the vLLM
    server subprocess sees the extracted site-packages (via `serving_env()`).
* Boot attestation swapped for the Flash-Next signature: `architectures ==
  ["Qwen4ExpForConditionalGeneration"]`, `model_type == "qwen4_exp"`, 206
  shards > 186 GB, plus a greedy decode fingerprint through the analyzer
  endpoint. (PLE-conversion provenance is hash-verified against
  `FLASHNEXT_GCP_MODEL_INFO.json` in the assemble cell, as in the gate.)
* If boot lands on the DEGRADED rung (16k server), the window shrinks to
  12288/2048 and the run is flagged by the boot verdict.

## Pre-registered read (fixed before flight)

Baseline = pooled STOCK 27B phases, same GPU class, same geometry (3 kernels:
arc3-tp-smoke / arc3-tp1b-smoke / arc3-tp1c-smoke, 2026-08-29):
levels/game **1.00 / 1.04 / 0.84-0.88**, zero-level games **8-9 of 25**, mean
local score **3.5-4.9**.

* **PASS** — flashnext levels/game ≥ 1.3, OR zero-level ≤ 6 with levels ≥ 1.0.
* **INCONCLUSIVE** — levels/game 0.9-1.3.
* **FAIL** — levels/game < 0.9.

Secondary (recorded, not gating): per-game actions — expect MORE actions/game
from the faster serving (gate conc-28: 412 vs the 27B's 297 gen tok/s
aggregate at prompt ~22k); prefix-cache hit rate; prefill/gen ratio; queue
running/waiting; per-game turns and session tokens.

Outputs: `/kaggle/working/flashnext_smoke_results.json` (phases + verdict),
`flashnext_smoke_boot.json` (assemble/boot record), `vllm-gcp_exact.log`,
`stdout.log`. Grep the log for `FLASHNEXT SMOKE READ:`.

## Files

* `build_flashnext_smoke.py` — builder (reads the duck38-v12 notebook + the
  flashnext-gate build; writes the .ipynb + kernel-metadata.json)
* `arc3-flashnext-smoke.ipynb` — the kernel (17 cells)
* `kernel-metadata.json` — id `ahmedmobasher86/arc3-flashnext-smoke`,
  NvidiaRtxPro6000, internet off, competition source attached (the RTX Pro
  6000 gate), 6 datasets, **no model_sources**
* `validate_flashnext_smoke.py` — structure + per-cell syntax (await-wrapped)
  + metadata + freshness + pyflakes

## Run

```sh
.venv/bin/python submission/_flashnext_smoke/build_flashnext_smoke.py
.venv/bin/python submission/_flashnext_smoke/validate_flashnext_smoke.py
# push (ONLY on Ahmed's go — costs a GPU commit, not a submission slot):
cd submission/_flashnext_smoke && python3 -m kaggle kernels push -p .
```
