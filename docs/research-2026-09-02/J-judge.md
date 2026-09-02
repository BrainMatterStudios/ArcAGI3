# Judge report — step-change search (2026-09-02 ~16:05 UTC), fresh-context adversarial

## Findings
F1. The leaderboard monitor (field-delta/lb_history.json) stores running MAXIMA per team, not draws. model-axis's "Tufa's last three draws span 0.13 => live noise ~0.1" is a fallacy; field-delta's "six live draws averaging 2.83" are team maxima.
F2. Kaggle "Best Score" = best over a notebook's versions' submissions. Verified per fork (Playwright, logged in): keithtyser V14 3.38 (svid 346399440; 14 versions, 72 subs, 109 copies; V13 differs only in markdown; V10 and earlier had NO serving profile = the "114 tok/s, 3.23 local" version). kashyapsinhgohil: 1 version, LB 1.65→2.73 (09-01)→2.80 (09-02) = two draws of the same bytes. jakobbrggen "Helmut AGI": 1 version, 3.22. zoli800: no submission from that notebook (field-delta's 2.38 wrong). sukirman: unverified. kashyap/zoli800/jakob forks are cell-identical to V14. Best estimate of the exact-bytes per-draw live mean: verified {2.80, 3.22, 3.38} + probable {2.73}: mean ≈2.9, sd ≈0.35, n=3–5 (mild selection: each conditioned on exceeding a prior max of 1.65/2.51/2.36). The 09-01/09-02 wave (58 teams up >0.5; the 20 with prior ≤2.0 landed mean 2.78, sd 0.29) is consistent. Defensible per-draw mean for a byte-copy: 2.6–2.9.
F3. Byte-identity holds: agent code and framework diff empty vs june_stock / taaf_scored_ref; SOURCE_IDENTITY policy_source_changes: [].
F4. field-delta mis-stated keith's serving: NOT 28 seqs. Notebook cell 3 profile kv5-bf16-mtp3-c8-cg32: MAX_NUM_SEQS=8, KV_CACHE 5 GiB, cudagraph 32, batched 8192, MTP 3, prefix caching OFF. His log: "Maximum concurrency for 32,768 tokens per request: 3.21x"; mean generation 249 tok/s, running 3.1 reqs, waiting 21.6; 1,371 requests (55/game), queue 124 s, inference 18 s, e2e 142 s, TPOT 11.6 ms (≈86 tok/s per running stream), MTP acceptance 60%, 57 preemptions.
| knob | keith V14 | our Flash-Next flight |
|---|---|---|
| vLLM | pinned image + 3.9 KB PLE-FP8 patch | sonpham gcp runtime, PLE CPU offload |
| seqs / KV | 8 seqs, KV 5 GiB (≈3 running) | 22 seqs, gpu-mem 0.96 (18 running) |
| batched / async / cudagraph | 8192 / async+chunked / 32 | 6144 / default / default |
| prefix caching | off | on (45% hit) |
| MTP | mtp,3 | none |
| parser / template | qwen3_coder, explicit chat_template, model gen-config | qwen3_xml, --generation-config vllm, preserve_thinking |
| analyzer env | CONTEXT_WINDOW 32768, MAX_OUTPUT 0 | 24576, 4096 |
| harness | conc 28, 7920 s/game, analyzer_timeout 900 | conc 28, max(6600,(32400-elapsed-300)/4) |
| aggregate gen tok/s | 249 | 465 (27B stock 489–563) |
| requests/game | 55 | 185 |
| reasoning chars/call | mean 3,406, median 2,206; 53 steps/game | mean 1,536, median 575; 125 steps/game |
| local 25-game | 1.44 lv, 6.76 pts | 1.16 lv, 3.87 pts |
F5. Three identical-bytes LOCAL draws (forks' commit runs): 1.44/6.76, 1.80/9.00, 1.28/6.42 → mean 1.51 lv, 7.39 pts. Our 27B stock 1.18 lv / 5.22; our Flash-Next 1.16 / 3.87. Local delta ≈ +0.33 lv (~2 SE) — BELOW the local bar; the live delta is ≈ +1.0.
F6. The 114→236 tok/s story is recovery from a pathological config (28 seqs, no KV cap → ~4 tok/s per stream → thoughts hit the 900 s timeout), not a throughput lever above ours. Our stacks run 2× his aggregate. "3.23→6.76" is SCORE not levels (V14 local levels 1.44).

## Which arithmetic is wrong
- loss-ledger's hazard math is right but answers the wrong question (2× actions ≈ +0.15 lv). Our Flash-Next smoke: 2.85× the actions of 27B stock and a worse score. Budget is not the lever.
- field-delta's mechanism is wrong: the winning config runs the SAME cadence as our 27B stock (53 vs 55 calls/game), +35% actions, HALF our aggregate throughput. What differs is the per-call regime: fewer, un-truncated, longer-reasoning calls at 32k context.
- model-axis's "same model ⇒ same regime; loop bound by the 60 s yield" is falsified by the same model producing 53×3.4k in keith's serving and 125×1.5k in ours. Its model-swap kill stands; its serving/env kill does not.
- Which knob carries the effect is NOT identifiable from this data and cannot be resolved by 25-game score A/Bs (identical bytes vary 1.28–1.80). Don't isolate before flying the bytes.

## Max-statistic trap
V31 (The AGI Boys) 2.66 was a max over 50 subs (+2σ tail) → our copy drew 1.71 as expected. keith V14 is different in kind: three independent teams with ONE version each drew 2.80/2.73, 3.22, 3.38 on first exposure. Expected live of a byte-copy ≈2.7 (90% interval 2.1–3.3); P(≥2.2) ≈0.9; P(≥2.7) ≈0.55. Versus our 1.71/1.72/1.88: +0.8 to +1.0 expected — at the LB bar, the only candidate with live evidence.

## Other candidates
- memory-channel rebuild: DEAD as a step change (the identical dead note is present in the 2.8–3.4 config).
- protocol-lite: research only; Stage-0 kill test (≤1 GPU-h) legitimate when GPU is idle; Tycho-27B 0/4 stands.
- model-axis kill: survives for swaps; wrong for serving regime.
- reasoning_effort medium: DEAD (direction contradicts the data).

## Ranked list
1. Byte-copy keithtyser V14 (svid 346399440; datasets keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1, keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1; model keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1; no grafts, no edits). Decisive experiment = the commit run (≈2h30m) + a slot. Attest before submitting: PUBLIC25_VLLM_PROFILE name=kv5-bf16-mtp3-c8-cg32, VLLM_SETUP_COMPLETE, "Maximum concurrency … 3.2x", mean gen ≥200 tok/s with waiting ≈20, watchdog restarts 0, PUBLIC25_AUDIT runs=25, score.json ≥5.0 and ≥1.1 lv/game.
2. Knob isolation on top of V14 by TELEMETRY (reasoning chars/call, requests/game), not score — after it flies.
3. protocol-lite Stage-0 kill test (idle GPU, 0 slots).
4–6. memory rebuild / effort dial / model swaps: dead.

## Tonight's arm
Fly the V14 byte-copy if it can be committed and attested. Pre-registered live reading vs baseline (27B family n=9 mean 1.54 sd 0.17; V22 1.71/1.72; Flash-Next 1.88; exact-bytes field {2.80, 3.22, 3.38}): ≥2.4 transfer confirmed → new base, redraw once to size the mean; 1.9–2.4 partial → diff commit telemetry vs keith's before any conclusion; <1.9 fork not byte-equivalent in effect → pull the commit log. SAY NO if any attestation line is missing, local read <1.1 lv AND <5.0 pts, boot falls to fallback/restarts, sources fail to attach, GPU quota can't cover 2.5 h, or audit not reached by 23:30 UTC → default to the stock redraw (~1.7, near-zero information).
Could not verify: sukirman's fork; which version produced keith's 2.36; scored-rerun telemetry; pickle contents; remaining GPU quota.
Artifacts: scratchpad/search/judge/ (kout/<user>/ commit outputs incl. benchmark.json, score.json, vllm metrics/logs, transcripts; kernels/ fork sources).
