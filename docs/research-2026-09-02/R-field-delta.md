# field-delta — final report (2026-09-02)

## 1. Findings
**F1. The top tier moved in two synchronized waves; both are model/serving releases.** Tong Hui Kang's leaderboard monitor (thread 709355 → https://arc3.huikang.dev/leaderboard, dumped to field-delta/lb_history.json; 2,719 teams). Days with ≥0.8-point single-step jumps: ≤3 teams/day through 08-10, then 7–15/day 08-14→08-31, then 21 (09-01) and 23 (09-02). HF creation dates: Qwen3.8-27B-FP8 08-13; Qwen3.8-Flash-Next 08-24; RadixArk NVFP4 quant 08-25.
Per-team jumps: Franzen 1.24→2.58 (08-14)→4.05 (08-30). Son Pham 1.22→2.03 (08-16), 2.43→3.58 (08-30)→4.42→4.52 (09-01; runtime 10h0m). Tong Hui Kang 1.06→2.24 (08-23)→3.39 (08-25)→4.45 (08-31). Hiểu Vy 1.70→2.73 (08-31)→4.11 (09-01). Ebi 0.29→3.85 in 5 subs (08-28→08-31). KaiN 3.77 on first sub (09-01). jinbo wang1 2.01→5.49 (09-02). keithtyser 1.26→2.13 (08-18), 2.36→3.38 (09-02). Us: 1.30→1.74 (08-17, 27B), 1.88 (08-31, Flash-Next).
Three teams reached ≥4.5 BEFORE the public Flash-Next release: Tufa 3.04→4.58 (08-23), Lord Han Solo 3.36→4.99 (08-24), cstl 3.57→5.99 (08-24)→7.51 (08-30). Not public.

**F2. The best public notebook = byte-identical June stock agent code + Flash-Next NVFP4 + tuned vLLM serving, scoring 2.4–3.4 live.** keithtyser/duck-qwen3-8-flash-next-nvfp4-mtp V14: Kaggle "Best Score 3.38"; forks: jakobbrggen anim-base 3.22, sukirman 2.86, Kashyapsinh 2.80, zoli800 2.38; keithtyser's earlier draw 2.36. `diff -rq june_stock/src/ARC3-Inference field-delta/keith_smoke_v1/src/ARC3-Inference` → EMPTY. Author's comment (09-01): "Had to offload some of the model to RAM to fit it. The first version performed poorly because it was too slow (114 tok/s) and only scored 3.23 on the 25 public games. After tuning, was able to improve it to 236 tok/s and 6.76." Self-reported ×2.1 local from serving throughput alone.
His serving (keith_smoke_v1/serving_setup.py:129-158, 2270-2400): --max-num-seqs 28, --max-model-len 32768, --max-num-batched-tokens 8192, --async-scheduling, chunked prefill, --speculative-config {"method":"mtp","num_speculative_tokens":3} (optional per-batch-size MTP schedule), cudagraph capture, --quantization modelopt_fp4, prefix caching off, gpu-mem 0.92, plus a 3.9 KB vLLM PLE-loader patch.
Our Flash-Next flight (submission/_flashnext_flight/arc3-flashnext-flight.ipynb lines 560-584, 816-849): NO MTP, --max-num-seqs 22 (fallback 12), --max-num-batched-tokens 6144, LOCAL_ANALYZER_CONTEXT_WINDOW 24576, kv auto, PLE CPU offload. Local smoke 1.16 lv/game, LB 1.88. Six live draws of keithtyser's config average 2.83 (sd ≈0.42); our 1.88 is ~2.3 sd below.

**F3. Son Pham's public repo (github.com/sonpham-org/arc-3, 324 commits).** Kaggle sub 55551321 = 2.03 (08-16) = pristine TAAF + Qwen3.8-27B-FP8 + reasoning_effort xhigh + temp 0.6 + 65536/32768 context + 4× image, differing from stock in two files / 19 lines (8-action checkpoint per action() call). README ablation (ex-ft09, 2-pass): baseline-v12 1.21; frame-full 1.44; no-impact band 1.62 vs 1.05 control (+55%); state graph regressed 3×; "model swaps all failed"; "tempo regime (60 s yield, act-look-act) dominates everything else". Their 3.58→4.52 (08-30→09-01) coincides with their datasets sonphamorg/arc3-flashnext-serving-part-{a,b,c}-v1 (08-29; RadixArk NVFP4, pinned vLLM). No harness commits after 08-19; the 4.5 is Flash-Next on cap-8/xhigh TAAF.

**F4. reasoning_effort** ∈ {xhigh (default), medium, low} in the Qwen3.8 chat template; stock sends only enable_thinking → already xhigh.

**F5. Tufa's public code unchanged since June** (github.com/Tufalabs/duck-harness, 2 commits 07-01). Tufa path on the monitor: 1.45 (07-18), 1.62 (08-10), 2.07 (08-19), 2.97 (08-20), 3.04 (08-21), 4.58 (08-23), 4.71 (08-30, 6h39m). June write-up: "main driver … better base models and multi-modality. Hand-crafting specific tools did not help."

**F6. Forum 737617 (08-26):** no disclosure. Pellegrin (736578): own harness 5.0–5.4 local → 1.4 LB while Tufa stock 2.1 local → 1.4 LB; Son Pham: public set is easier and trained-on.

## 2. Ranked step-change candidates
**#1 Serve Flash-Next NVFP4 at ≥230 tok/s decode — MTP-3 + async + 28 seqs + chunked prefill 8K: adopt keithtyser V14's serving_setup.py verbatim on stock code.** Ceiling: author's 3.23→6.76 local (×2.1); field 2.36–3.38 live vs our 1.88. Contradicting: one live draw of his config scored 2.36; sd ~0.4 on 6 draws. Cheapest decisive test: fork keithtyser V14 unchanged, run public-25 offline (2h23m per his log), read levels/game and tok/s; ≈2.5 GPU-h, 0 slots. Rule: local mean ≥6.0 and ≥1.6 lv/game (vs our 1.12–1.20) → submit as-is next slot; live: ≥2.4 transfers, <2.1 noise/serving mismatch — pull the vLLM log for tok/s.
**#2 reasoning_effort medium vs xhigh** on the same serving: ceiling unknown; Son Pham's brevity note argues against. 25-game A/B ~2.5 GPU-h; needs ≥+0.5 lv/game paired.
Sub-threshold riders: Son Pham's 8-action checkpoint (19 lines), no-impact guard (~+0.2 lv/game).

## 3. Common factor of the 4+ tier
Every ≥4.0 team with observable timing jumped inside 08-23→09-02, and every one with public artifacts runs Qwen3.8-Flash-Next NVFP4 on vLLM with the stock or near-stock duck loop. Runtimes ~9h; Son Pham's 4.52 ran 10h0m. A model+serving tier, not a harness tier.

## 4. Could not verify
What cstl / Lord Han Solo / Tufa (4.58 on 08-23, before Flash-Next release) run. Our Flash-Next flight's measured tok/s (no throughput line in logs). Son Pham's later private submissions. Whether the private 85 games rescale these gains (public→LB ratios 0.27–0.67 across harnesses).
Artifacts: scratchpad/search/field-delta/ (NOTES.md, lb_history.json, keith_smoke_v1/, sonpham repo clone, taaf_src_0901/).
