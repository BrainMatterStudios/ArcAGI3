# brain_scout — HF API measurements (2026-08-14, all numbers pulled live from huggingface.co API)

Eval target: RTX Pro 6000 96GB (Blackwell, native FP4/FP8), vllm==0.19.0 torch==2.10.0 flashinfer==0.6.6
(pin verified in scratchpad/taaf_scored_ref/setup_commands.json). Probe GPU: Modal H100 80GB (offkaggle/modal_vllm_serve.py).
KV bytes/token computed from config.json as 2*layers*kv_heads*head_dim*2 (bf16) — an OVERCOUNT for
hybrid-linear-attention models (Qwen3.5/3.6 MoE: only every 4th layer is full attention; gpt-oss: half the
layers are 128-tok sliding window), so real KV is smaller than listed.

| model (HF id) | params | arch (model_type) | weights on disk | ctx | KV bf16 @32k (overcount) | fits 96GB | fits H100-80 probe | license |
|---|---|---|---|---|---|---|---|---|
| Qwen/Qwen3.6-27B-FP8 (current brain) | 27.8B dense | qwen3_5 | 30.9 GB | 262k | 8.6 GB | YES (shipped) | YES (serving now) | apache-2.0 |
| Qwen/Qwen3.6-35B-A3B-FP8 | 36.0B MoE A3B | qwen3_5_moe | 37.5 GB | 262k | 2.7 GB | YES | YES | apache-2.0 |
| Kwaipilot/KAT-Coder-V2.5-Dev (BF16) | 34.7B MoE A3B | qwen3_5_moe | 69.3 GB | 262k | 2.7 GB | YES | MARGINAL (~69.3+KV+overhead vs 76GB budget at util .95) | apache-2.0 |
| cyankiwi/KAT-Coder-V2.5-Dev-AWQ-INT4 | " | qwen3_5_moe | 24.4 GB | 262k | 2.7 GB | YES | YES | apache-2.0 |
| Qwen/Qwen3.5-122B-A10B-FP8 | 125B MoE A10B | qwen3_5_moe | 127.2 GB | 262k | 3.2 GB | NO | NO | apache-2.0 |
| cyankiwi/Qwen3.5-122B-A10B-AWQ-4bit | " | qwen3_5_moe | 80.0 GB | 262k | 3.2 GB | YES (tight: ~85-87 used) | NO (weights alone = full card) | apache-2.0 |
| ippeiogawa Kaggle NVFP4 snapshot | " | qwen3_5_moe | 71.3 GB | 262k | 3.2 GB | YES (Blackwell-native FP4) | vLLM NVFP4-on-Hopper = emulated; needs check | apache-2.0 |
| openai/gpt-oss-120b (MXFP4) | 116.8B MoE A5B | gpt_oss | 65.2 GB (root; repo total 195.7 incl original/+metal) | 131k | 2.4 GB | YES | YES | apache-2.0 |
| nvidia/gpt-oss-puzzle-88B | 90.8B | gpt_oss_puzzle | 50.0 GB | ? | ? | YES if arch supported | YES if arch supported | nvidia-open (other) |
| nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 | 120B MoE A12B (67.2B stored NVFP4) | nemotron_h | 80.3 GB | 262k | 3.0 GB | YES (tight) | NO | nvidia-open (other) |
| Hcompany/Holo-3.1-35B-A3B | 35.1B MoE A3B | qwen3_5_moe | 70.2 GB bf16 | 262k | 2.7 GB | YES | MARGINAL (same as KAT bf16) | apache-2.0 |
| mistralai/Devstral-Small-2-24B-Instruct-2512 | 24.0B dense | mistral3 | 51.6 GB bf16 | 393k | 5.4 GB | YES | YES | apache-2.0 |
| ByteDance-Seed/Seed-OSS-36B-Instruct | 36B dense | seed_oss | 72 GB bf16 | 524k | 8.6 GB | YES | MARGINAL/NO at 32k | apache-2.0 |
| zai-org/GLM-5.2-FP8 | 753B MoE | glm_moe_dsa | 755.6 GB | — | — | NO (AWQ-INT4 still 474.2 GB) | NO | mit |
| deepseek-ai/DeepSeek-V4-Flash | 291B | deepseek_v4 | 159.6 GB (NVFP4 168.3) | — | — | NO | NO | mit |
| moonshotai/Kimi-K3 | 2.78T | kimi_k3 | 1560.9 GB | — | — | NO | NO | other (modified) |
| MiniMaxAI/MiniMax-M2.7 | 228.7B MoE | minimax_m2 | 230.1 GB (NVFP4 139.9, AWQ 130.5) | — | — | NO | NO | other |
| mistralai/Devstral-2-123B-Instruct-2512 | 125B dense | ministral3 | 256.5 GB bf16 (AWQ repo 149.9) | — | — | NO | NO | other |

Kaggle dataset search (kaggle CLI, 2026-08-14):
- Qwen3.5-122B-A10B: ippeiogawa/qwen35-122b-a10b-nvfp4 (71.3 GB, updated 2026-07-17) and two GGUF sets — EXISTS.
- Qwen3.6-35B-A3B: tahsinekajolasalami/qwen36-35b-a3b-gguf (23.1 GB GGUF); no HF-format FP8 snapshot found — we would upload our own (as done for the 27B: driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot).
- gpt-oss-120b full weights: NO obvious dataset in top results — would need our own ~65GB upload.
- KAT-Coder, Nemotron-3-Super, Seed-OSS: NONE found.
- Devstral-Small-2: naox3456256/devstral-small-2-24b-instruct-2512-q6-k-l (19.6 GB GGUF only; GGUF is llama.cpp, not vLLM-preferred).
Uploading our own HF snapshot as a private Kaggle dataset is the proven path (done for 27B FP8, 31GB); Kaggle per-dataset cap ~200(?)GB — 65-80GB uploads feasible but slow.
