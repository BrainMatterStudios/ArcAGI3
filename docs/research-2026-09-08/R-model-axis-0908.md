# Model axis refresh 2026-09-08 (memory-blind; primary sources)

Fit rule: ≤ ~85 GiB of weights on the 96 GB RTX PRO 6000 (Flash-Next NVFP4 loads at 81.8 GiB on the box).

| # | model | params | 4-bit size | fits | agentic evidence | vLLM | verdict |
|---|---|---|---|---|---|---|---|
| 1 | **Qwen3.8-Flash-Next** (Aug 26) — incumbent | 125B MoE, 6B active + 51B n-gram/PLE + 4B MTP | NVFP4 ~80 GiB | yes | SWE-bench Pro 62.5, DeepSWE 58.7, Toolathlon-V 73.5, LCB v6 91.9, GPQA 91.7 (card) | day-0; NVFP4+MTP PR #55513 approved Sept 8; fp8-KV PR #55557 open | baseline |
| 2 | **Qwen3.8-27B** (Aug 14) | 27B dense hybrid | NVFP4 18.9 GiB | yes, ~60 GB spare for KV | below Flash-Next on all 12 shared benchmarks; Toolathlon-V 67.1, SWE-bench Pro 61.7 | mature; MTP acceptance 0.90 NVFP4; fp8 KV | the only candidate with a *mechanism* (10× KV, 1.47× 8-stream throughput in the one independent test) |
| 3 | Nemotron 3 Super 120B-A12B (Mar) | 120B MoE, 12.7B active | NVFP4 77 GB w/o MTP | yes, no MTP | τ-bench V2 61.1, LCB 78.4; text-only | recipe exists; MTP OOMs on RTX 6000 (HF discussion #9) | below Flash-Next |
| 4 | Gemma 4 31B (Apr) | 31B dense, multimodal | FP8 31 GB | yes | τ2 76.9, LCB 80.0; two Milestone-1 winners used it and lost to the 27B duck | mature | below 27B |
| 5 | Mistral Medium 3.5 128B (Apr) | 128B **dense** | NVFP4 ~70–75 GB | marginal | SWE-bench Verified 77.6 | recipe assumes TP4 | 20× Flash-Next's per-token compute → throughput cliff on a queue-bound rig |
| — | GLM-5.3-Flash (Aug 26) | 320B MoE, 18B active | NVFP4 ~115 GB | **no** | beats Flash-Next on every overlap (DeepSWE 63.4, Toolathlon-V 78.4, Terminal-Bench 84.3) | ≥0.29 | would be #1 if it fit |
| — | DeepSeek V4-Flash-0731 | 284B MoE, 13B active | NVFP4 ~150 GB | **no** | **ARC-AGI-2 semi-private verified 61.4%** at $0.04/task | Blackwell build | strongest ARC signal; does not fit |
| — | Step 3.7 Flash, MiniMax M3, Kimi K2.6/K3, GLM-5.3 744B, Qwen3.8-2.4T, Llama 4 Maverick | | | no | | | dead on fit; "Llama 5" does not exist; no GLM-5.x "Air" |

- **No open-weight model at ≤ 85 GiB beats Flash-Next per call on today's evidence.** The two that do (GLM-5.3-Flash, DeepSeek V4-Flash) are ~2× too big and vLLM has no merged expert offload (RFC #38256 / PR #37190 open, enforce-eager only).
- **ARC-AGI-3 open-weight results:** none published outside Kaggle; the official ARC-AGI-3 leaderboard has zero open-weight entries (Astra 62.7, Claude Opus 5 30.2, GPT-5.6 Sol 7.8, rest ≤ 1.5).
- **Serving tricks (vLLM primary sources):** fp8 KV for Flash-Next QSA layers = PR #55557 open (KV pool ×1.77, but MTP acceptance 0.72→0.63, decode −10%); prefix caching + MTP still lossy (#38182 hit 92→71%, #43559 accuracy drop; PR #51113 merged Aug 6 fixes poisoning on hybrid GDN with `mamba_cache_mode=align` — needs ≥ v0.27 and our own measurement); expert offload: no merged path; PLE table can be CPU/NVMe-offloaded (`VLLM_PLE_CPU_OFFLOAD=1`, community disk-backed patch); reduced-vocab MTP draft (65,536 lowest ids) saves draft compute; `--max-num-seqs` sizes the GDN state pool.
- What the top Kaggle teams serve: not public. Tufa's own July finding: "better base models were the main driver". NVARC3 (NVIDIA) and Franzen point, by track record, to fine-tuned small models (inference).

Sources: Qwen3.8-Flash-Next / -27B cards and vLLM recipes; nvidia NVFP4 builds; featherless 3-way comparison; classmethod DGX-Spark test; Nemotron cards/recipe/discussion #9; Gemma 4 card; Mistral Medium 3.5 card; DeepSeek V4-Flash ARC verification; vLLM PRs #55513, #55557, #54846, #51113, issues #38182, #43559, #38256, #41447.
