# Stronger-local-brain scouting — 2026-08-14

Constraint set (all verified from repo code, not assumed):
- Eval GPU: RTX Pro 6000 96GB (Blackwell), pinned stack `vllm==0.19.0 torch==2.10.0 flashinfer==0.6.6`
  (scratchpad/taaf_scored_ref/setup_commands.json). Weights+KV must fit at >=32k ctx; 65536 max-model-len today.
- Probe GPU: Modal H100 80GB (offkaggle/modal_vllm_serve.py, GPU_KIND="H100"), ~$3.95/hr.
- Loop: Tycho-style agentic verification (write/verify world_model.py via tool calls); needs reliable
  multi-turn tool calling + 128k-preferred context.

Raw size/KV measurements: see hf_measurements.md (HF API, 2026-08-14).

## Verdict summary (measured paper numbers; same-harness caveats flagged)

Reference — our CURRENT brain, Qwen/Qwen3.6-27B-FP8 (30.9GB), official card numbers:
SWE-bench Verified **77.2**, SWE-bench Pro 53.5, SWE-Multilingual 71.3, Terminal-Bench 2.0 **59.3**, LCB v6 83.9.
**No open-weight model that fits 96GB posts clearly higher agentic-coding numbers than the 27B we already run.**
The scouting case is therefore not "bigger = smarter" but (a) tool-loop reliability, (b) decode speed
(more verification iterations per wallclock), (c) family diversity.

| candidate | fits 96 / fits 80-probe | vLLM 0.19.0 arch | agentic evidence (cited) | Kaggle dataset | verdict |
|---|---|---|---|---|---|
| KAT-Coder-V2.5-Dev (35B-A3B MoE, apache-2.0) | YES / AWQ-INT4 24.4GB YES, BF16 69.3GB marginal | qwen3_5_moe — in vLLM since v0.17.0 (GH PR #37068, issue #35344); same `--reasoning-parser qwen3 --tool-call-parser qwen3_coder` flags our serve already uses | Model card: SWE-V 69.40 vs base Qwen3.6-35B-A3B 64.40 (same harness), Terminal-Bench 2.1 41.02 vs 32.02, SWE-Pro 45.96 vs 40.63; RL cut abnormal tool labels 9.34%→0.28% | none — upload our own (24.4GB AWQ trivial; 27B precedent exists) | **TOP-1 probe** |
| gpt-oss-120b (116B MoE A5B, apache-2.0, MXFP4 65.2GB) | YES / YES | gpt_oss — listed in vLLM 0.19.1 supported models | OpenAI card: SWE-V ~62.4 (o3-mini+ class), agentic tool RL, 131k ctx. HARNESS-SENSITIVE: 62.4 native-harmony vs 26 wrong-harness on SWE leaderboard; our own DUCK harness measured <=0.13. vLLM chat-completions tool_call was broken early (GH #22578, works on /v1/responses) — Tycho HAS a responses backend | none found — 65GB upload needed | **TOP-2 probe** (only via harmony-correct path) |
| Qwen3.6-35B-A3B-FP8 (37.5GB) | YES / YES | qwen3_5_moe (as above); prior duck-sparse v7 wiring already built | Official card: SWE-V 73.4, TB2.0 51.5 vs 27B's 77.2/59.3 — WEAKER on paper but A3B ≈ several× decode speed; public duck-fork 35B swap scored 0.000 (DUCK harness, confounded) | GGUF only (tahsinekajolasalami) — would upload FP8 snapshot | **TOP-3 probe** (speed hypothesis only) |
| Qwen3.5-122B-A10B (AWQ 80.0GB / NVFP4 71.3GB) | YES tight / **NO — fits 96 not 80** | qwen3_5_moe | Card: SWE-V 72.0, TB2 49.4, BFCL-V4 72.2, TAU2 79.5, ctx 262k — one generation older; below 27B on coding, strong pure tool-calling | **YES**: ippeiogawa/qwen35-122b-a10b-nvfp4 (71.3GB, 2026-07-17) | bench case too weak vs 27B; probe needs H200 (~$4.5-5.5/hr) |
| Nemotron-3-Super-120B-A12B NVFP4 (80.3GB) | tight / **NO** | nemotron_h listed in 0.19.1 docs (as Nemotron-H-8B; Super-120B revision unproven on 0.19) | nvidia-open license; benchmarks not scouted deeper — blocked by probe-GPU fit + license + arch risk | none | drop |
| Devstral-Small-2-24B (51.6GB bf16, apache-2.0) | YES / YES | mistral3 listed | SWE-focused but tuned for OpenHands scaffold; 24B dense, weaker general reasoning class | GGUF only | reserve |
| Seed-OSS-36B (72GB bf16) | YES / marginal | seed_oss listed | 512k ctx; no standout agentic-coding evidence found vs 27B | none | drop |
| gpt-oss-puzzle-88B (50GB, nvidia pruned) | YES if arch loads / YES | gpt_oss_puzzle NOT in 0.19.1 docs | pruned gpt-oss-120b | none | drop (arch unproven on pinned vLLM) |
| GLM-5 / 5.1 / 5.2 (753B, MIT) | NO (FP8 755.6GB; AWQ-INT4 474.2GB) | glm_moe_dsa NOT in 0.19.1 docs | frontier-class but physically impossible here; NO Air-class <=120B GLM-5 release exists on HF (searched) | — | impossible |
| DeepSeek-V4-Flash (291B, MIT, 159.6GB; NVFP4 168.3GB) | NO | deepseek_v4 NOT in 0.19.1 docs | the DVM forum post's model — open weights but 2× our VRAM; API-only for that team presumably | — | impossible |
| Kimi-K3 (2.78T, 1.56TB) | NO | kimi_k3 NOT in 0.19.1 docs | teacher-only via API (as we already use) | — | impossible |
| MiniMax-M2.5/M2.7 (228.7B; NVFP4 139.9GB, AWQ 130.5GB) | NO | minimax_m2 listed in 0.19.1 | good agentic rep | — | impossible at 96GB |

## Top-3 probe plan (single-mode sb26, 300-call cap, Modal)

All three reuse the existing rig unchanged (scratchpad/tycho_eval/run_single.sh + qwen27b_single.yaml with
model/base-url swapped); Modal serve needs only a model-path + parser change in offkaggle/modal_vllm_serve.py.

1. **KAT-Coder-V2.5-Dev-AWQ-INT4** (cyankiwi, 24.4GB) — hypothesis: RL-hardened tool-loop reliability
   converts calls into MORE verified world-model iterations than raw 27B IQ does. Drop-in: same qwen3_coder
   parser, same qwen3 reasoning parser, `--language-model-only`. A3B decode ≈2-4× the 27B's 56 tok/s.
   Probe: ~45-70 min H100 ≈ **$3-5** (+ ~10 min one-time weight pull).
   Caveat: AWQ-INT4 quantizes the probe; eval could ship BF16 (69.3GB, fits 96) — quant delta unmeasured.
2. **gpt-oss-120b MXFP4** (65.2GB) — hypothesis: our ≤0.13 DUCK measurement was harness artifact
   (62.4-vs-26 SWE precedent); a verification loop driven through vLLM's harmony-native path is the fair test.
   Requires: switch Tycho to its responses backend OR verify vLLM 0.19.0 chat-completions gpt-oss tool
   parser now works (was broken in GH #22578; recipes page documents current flags). Probe: ~60-90 min ≈ **$4-6**.
3. **Qwen3.6-35B-A3B-FP8** (37.5GB) — hypothesis: in wallclock-bounded eval, ~3-5× decode speed buys more
   verify-fix cycles than 4pp SWE gives up. Cheapest probe: ~35-60 min ≈ **$2-4**; serving wiring already
   exists from duck-sparse v7.

Flagged fits-96-not-80: Qwen3.5-122B-A10B (AWQ/NVFP4), Nemotron-3-Super-120B (NVFP4), KAT BF16 (marginal-80).
Probe path for those if ever needed: Modal H200 141GB.

## Measured vs speculative
- MEASURED: all disk sizes/params/arch strings/KV configs (HF API); vLLM pin + parser flags (repo code);
  Kaggle dataset hits (kaggle CLI); benchmark numbers as printed on model cards (linked repos); vLLM arch
  support statements (docs.vllm.ai v0.19.1 supported-models page + GH issues #35344/#37068/#22578).
- SPECULATIVE: decode-speed multiples for A3B/A5B MoE (not benchmarked by us); AWQ-vs-BF16 quality delta;
  whether gpt-oss chat-completions tool parsing is fixed by 0.19.0 (must be verified in the probe itself);
  KAT model-card cross-model numbers use KAT's harness (they note ~10pp deflation vs Qwen official).
