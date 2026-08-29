# arc3-serving-lab4 — vLLM 0.24 boot · prefix regimes · MTP

Private RTX Pro 6000 probe kernel. **No games, no submission, no slot cost.**
Budget ≤ 2.5 h (hard cap: no new phase after 135 min).

Built 2026-08-29 from `docs/research-2026-08-29/R5-pack3-vllm024-boot-recipe.md`
§6.2 and the working lab3 load generator (`submission/_serving_lab3/`).

```
.venv/bin/python submission/_serving_lab4/build_serving_lab4.py      # -> arc3-serving-lab4.ipynb + kernel-metadata.json
.venv/bin/python submission/_serving_lab4/validate_serving_lab4.py   # structure, per-cell syntax, invariants, freshness
# push only on Ahmed's go:  kaggle kernels push -p submission/_serving_lab4
```

## Questions

| | Question | What lands in `serving_lab4_results.json` |
|---|---|---|
| Q1 | Does vLLM **0.24.0** boot on the Kaggle image with the pack3 recipe (jcole75 wheelhouse v3, CUDA‑13.3 nvcc under `site-packages/nvidia/cu13`, `VLLM_USE_FLASHINFER_SAMPLER=0`, `VLLM_USE_DEEP_GEMM=0`, `--mamba-cache-mode align`, prefix caching ON) serving the foysalemonshanto repack? | `meta.driver_version` (printed first), `boots.base024.engine_lines` (engine‑config line incl. `speculative_config=`, attention backend, flashinfer.jit frame count, KV cache size), `meta.mtp_index` (the index must map tensors to `mtp.safetensors`), `verdicts.q1_boot` |
| Q2 | conc‑28 throughput in two prompt regimes — **churn** (every request drops its oldest block; shipped duck) vs **stable** (long fixed prefix, append‑only; Pack‑1 hysteresis) | per phase: `gen_tok_s_aggregate_metric`, `gen_tok_min_session_metric`, `prefix_cache_hit_rate_window` (Δhits/Δqueries over the measured window), prompt/latency stats; `verdicts.q2_prefix_stable` |
| Q3 | MTP (`{"method":"mtp","num_speculative_tokens":2}`) **with** prefix caching on 0.24: tool‑call round‑trip, 20‑prompt greedy battery vs the non‑MTP server, `vllm:spec_decode_*` acceptance, conc‑28 stable load again | `verdicts.greedy_mismatch_<rung>_vs_base`, `verdicts.greedy_floor_base_vs_base` (noise floor), `phases.stable_conc28_mtp2_*`, `verdicts.q3_mtp` |

## Pre‑registered decision rules

* **MTP is worth adopting only if** `greedy mismatches == 0` **AND**
  `stable‑regime conc‑28 gen tok/s with MTP >= 1.3x without` **AND**
  `acceptance >= 0.5` (accepted draft tokens / drafted tokens). A non‑zero
  base‑vs‑base greedy floor voids the mismatch instrument (reported, not hidden).
* **Prefix‑stable regime is confirmed if** its window prefix hit rate `>= 0.5`
  **AND** its gen tok/s `>= 1.5x churn`.
* Q1 is binary: the 0.24 engine serves `Qwen/Qwen3.8-27B-FP8` with
  `speculative_config=None` on the first boot.

## Boot ladder (nothing kills the kernel)

`base024` (Q1) → stop → `mtp2_prefix_on` → on boot failure `mtp2_prefix_off`
(`--no-enable-prefix-caching`, no mamba flag — `mamba_cache_mode` must be
`none` without prefix caching) → on failure `base024_recover`. On the MTP
server the cheap quality gate runs first (spec‑decode probe, parser, greedy
battery, ~6 min); if the prefix‑on arm looks corrupt (greedy mismatches > 0,
parser FAIL, or engine death) the kernel switches to the prefix‑off rung
*before* spending the load budget (time‑gated, < 100 min elapsed). Every phase
is try/except‑wrapped and results are rewritten after each one.

Expected timeline: install+boot ~25 min · Q2 ~35 min · MTP boot ~15 min ·
quality ~6 min · loads ~13–20 min · final — ~100–115 min; hard cap 135 min for
starting any new phase.

## Deliberate deviations from R5 §6.2 / lab3

* `--limit-mm-per-prompt '{"image": 1, "video": 0}'` is kept from the recipe
  and the generator sends **one** image per request (duck ships
  `MULTIMODAL_CONTEXT=current_grid`); lab3 sent up to 5.
* Sessions are pre‑seeded with 12 history turns so the *first* request is
  already ~17k tokens (lab3 calibration: real tokens ≈ 1.13× the generator's
  estimate); the churn regime always drops the oldest turn after a request
  (steady‑state duck, ~17–18k), the stable regime only appends (~25k by the end
  of a window) and never trims below 30k tokens.
* `reasoning_effort: "xhigh"` (pack3 policy) is not copied.
* No 197‑wheel SHA sweep: identity = manifest `wheel_count == 197`,
  `total_bytes`, `cuda_wheel_target == cu129`, eight named wheels present, plus a
  warning‑only manifest sha prefix/suffix check.

## Verified vs. unverified (2026‑08‑29)

Verified against the **v0.24.0 tag on GitHub** (raw files fetched):
`--mamba-cache-mode` (`none|all|align`, `vllm/config/cache.py`),
`--mm-encoder-tp-mode`, `--speculative-config`, `--limit-mm-per-prompt`,
`--enable-prefix-caching` / `--no-enable-prefix-caching` (BooleanOptionalAction),
`--max-num-seqs`, `--gpu-memory-utilization` (`vllm/engine/arg_utils.py`);
metric names `vllm:prefix_cache_queries`, `vllm:prefix_cache_hits`,
`vllm:generation_tokens`, `vllm:prompt_tokens`, `vllm:num_requests_running`,
`vllm:num_requests_waiting`, `vllm:num_preemptions` (`vllm/v1/metrics/loggers.py`),
`vllm:spec_decode_num_drafts`, `_num_draft_tokens`, `_num_accepted_tokens`,
`_num_accepted_tokens_per_pos` and the log line `Avg Draft acceptance rate`
(`vllm/v1/spec_decode/metrics.py`). Wheelhouse identity numbers come from the
local Kaggle‑CLI copy of `WHEELHOUSE_MANIFEST.json` (197 files, 6,486,476,943 B).

**Not verified** (best effort, never fatal in the kernel):
`POST /reset_prefix_cache` on 0.24 (result recorded per phase);
`vllm:kv_cache_usage_perc` gauge name; whether the reasoning field is
`reasoning_content` or `reasoning` in 0.24 (both are read); the exact
`Using <X> attention backend` log phrasing in 0.24 (taken from our 0.27.1 logs;
the raw engine‑config line is always captured); whether PR #47861 (MTP +
prefix‑cache GDN corruption) is in 0.24 — that is exactly what the greedy
battery measures; that `--mamba-cache-mode align` is rejected (vs. ignored)
with prefix caching off — the OFF rung simply omits it.
