# Stage A runbook — LoRA-GRPO go/no-go gate (duck + Qwen3.6-27B VLM)

Goal of Stage A (one rented day): prove the stack runs at all.
1. Rollouts through the REAL duck harness against a vLLM endpoint.
2. Trajectories captured token-faithfully (incl. grid images).
3. ONE LoRA-GRPO optimizer step completes on those captured VLM traces.

Everything in `scratchpad/rl_gate/` is verified locally except the trainer
step (needs GPUs). The capture path is proven end-to-end by
`smoke_test.py` (mock brain, real TAAF game, real ToolAgent loop): full
message prefixes with intact `data:image/png;base64` parts, sampling params,
usage, per-episode reward = `levels_completed`.

## Hardware

| Item | Spec |
|---|---|
| Rental | 4x H100 80GB SXM (NVLink preferred), 1 node |
| CPU/RAM | >= 32 vCPU / 256 GB (duck sandbox spawns subprocesses; TAAF is CPU-light) |
| Disk | >= 500 GB NVMe (bf16 27B ~55 GB + vLLM cache + traces) |
| Layout | GPU 0-1: vLLM rollout server (TP=2). GPU 2-3: trainer (FSDP/deepspeed) |

Rationale: our prior Kaggle measurement (single RTX 6000 96GB) showed vLLM
~34 GB + bf16 LoRA trainer ~54 GB do NOT fit together on one 96 GB card;
splitting server/trainer across dedicated GPUs removes that wall. A 27B in
bf16 is ~55 GB weights; TP=2 across 2x80GB serves it with room for the
multimodal KV cache; the trainer with LoRA (frozen base in bf16, FSDP
param sharding across 2 GPUs + activation checkpointing) fits in 2x80GB.

## Model source

- Rollouts and training MUST use the ORIGINAL bf16 checkpoint. Our Kaggle
  snapshot is FP8-quantized — NOT trainable (FP8 weight dequant + LoRA
  merge was already the failure mode in the July GRPO attempt).
- bf16 HF repo id (verified 2026-07-21 via the HF API + raw config.json, not
  memory): **`Qwen/Qwen3.6-27B`** — https://huggingface.co/Qwen/Qwen3.6-27B
  - public, not gated; 15 safetensors shards ~57 GB; `text_config.dtype:
    "bfloat16"`, no `quantization_config`.
  - vision tower INCLUDED (`vision_config`: SigLIP-style ViT, depth 27,
    hidden 1152, patch 16; chat template has `<|vision_start|>/<|image_pad|>`).
  - arch `Qwen3_5ForConditionalGeneration`, model_type `qwen3_5` — a HYBRID
    3:1 linear/full-attention 64-layer stack (matters for trainer support).
  - our Kaggle snapshot's sibling is `Qwen/Qwen3.6-27B-FP8` (separate repo);
    do not train from it.
- Sanity check on the box before anything else:
  `python -c "from transformers import AutoConfig; print(AutoConfig.from_pretrained('Qwen/Qwen3.6-27B'))"`
  and confirm dtype bf16 + vision tower present.

## Rollout serving (vLLM)

```bash
vllm serve Qwen/Qwen3.6-27B \
  --served-model-name duck-model \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --limit-mm-per-prompt image=16 \
  --port 8000
# CUDA_VISIBLE_DEVICES=0,1
```

Duck rollouts through the capture proxy (per episode):

```bash
.venv/bin/python scratchpad/rl_gate/run_rollout.py \
  --game <game> --upstream http://127.0.0.1:8000/v1 \
  --model-id duck-model --multimodal \
  --max-actions 80 --max-runtime-s 1800
```

Notes verified from source:
- The duck's only model call is a blocking non-streaming
  `requests.post(.../chat/completions)` (`tool_agent.py::_chat_completion`,
  `openai_compat.py` sets `"stream": False`) — the proxy's buffered JSON path
  is the real path.
- Images enter requests only when `MULTIMODAL_CONTEXT=current_grid`
  (`vision_context.py`); `run_rollout.py --multimodal` sets it. Decide
  BEFORE Stage A whether the trained policy is the image-on duck — train
  exactly the config you deploy.
- GRPO needs groups: run >= 4 episodes per game (same game, same start) so
  advantages don't degenerate. `run_rollout.py` is one-episode; loop it.

## Trainer for the single-step smoke test

See researched support below. Plan:
1. Convert traces: `traces_to_grpo.py <episode dirs>` -> grouped per-turn
   samples (done, generic JSON).
2. Re-tokenize with the model's processor (chat template + image processor)
   and CHECK token counts against captured `usage.prompt_tokens` (this is the
   token-faithfulness gate — mismatch = silent off-policy training).
3. One GRPO step, LoRA r=16 on attention+MLP of the LLM trunk (vision tower
   frozen), group-normalized advantage from `manifest.reward`
   (= levels_completed), importance ratio from recomputed log-probs.

## Kill criteria (pre-committed — decide NO-GO, do not rationalize)

| # | Criterion | Deadline |
|---|---|---|
| K1 | bf16 checkpoint not downloadable/servable by vLLM with images | hour 4 |
| K2 | Duck rollout against the rented endpoint fails or captures malformed traces | hour 8 |
| K3 | Re-tokenization mismatch vs served tokens that cannot be reconciled | hour 14 |
| K4 | ONE LoRA-GRPO optimizer step (loss backward + step, no NaN) does not complete on captured VLM traces by end of rental day | hour 24 → NO-GO |

NO-GO means: stop the GRPO line, keep the duck as-is; the fallback lever
remains rollout-quality work (shadow replay), not training.

## Cost table

| Item | Est. |
|---|---|
| 4x H100 SXM on-demand (Lambda/RunPod/Voltage Park, 2026 rates ~$2.2-3.0/GPU-h) | ~$9-12/h |
| Stage A: 1 day | ~$220-290 |
| Stage B (if GO): GRPO on 20 dev games, ~3-4 days incl. eval on 5 held-out | ~$650-1150 |
| Storage/egress | < $20 |
| Total worst case A+B | ~$1,500 |

## Stage A hour-by-hour

| Hours | Task |
|---|---|
| 0-2 | Provision, clone repo, venv, download bf16 weights |
| 2-4 | vLLM up (TP=2), curl a multimodal chat completion (K1 gate) |
| 4-8 | 4 duck episodes x 2 games via run_rollout.py + proxy; inspect traces (K2 gate) |
| 8-14 | traces_to_grpo + trainer-native tokenization; token-faithfulness check (K3 gate) |
| 14-22 | Wire trainer (see below), run ONE GRPO step on captured traces (K4 gate) |
| 22-24 | Snapshot adapters + logs, write GO/NO-GO verdict |

## Researched trainer support (verified 2026-07-21)

(All claims below verified by direct fetches of HF API / GitHub / readthedocs,
not memory.)

**Choice: verl** (repo moved to https://github.com/verl-project/verl), FSDP
backend + LoRA, vLLM rollout — the only framework with the exact `qwen3_5`
architecture in-tree for GRPO:

- GRPO + VLMs: maintained examples for Qwen2.5-VL / Qwen3-VL AND
  `examples/grpo_trainer/run_qwen3_5_35b_fsdp.sh` (+ Megatron variant) with a
  dedicated `verl/models/transformers/qwen3_5.py`. No 27B script — adapt the
  35B FSDP one to `Qwen/Qwen3.6-27B`.
- LoRA: documented for PPO/GRPO incl. "LoRA training for VLMs"
  (`freeze_vision_model` / `freeze_vision_projection`; docs recommend
  lora_rank >= 32): https://verl.readthedocs.io/en/latest/advance/ppo_lora.html.
  Caveat: sglang rollout needs merged weights (no native adapter loading).
- Multi-turn with images: documented via sglang multi-turn
  (https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html) —
  tools may return `ToolResponse(image=[...])`; in-tree test
  `tests/experimental/agent_loop/test_multi_modal.py` exercises a multimodal
  tool returning images. vLLM-side multi-turn image tool responses are LESS
  proven (open issue #4613) — prefer sglang for the trainer-side rollout if
  Stage B goes on-policy.
- Live risks: issue #6319 "loss always nan when sft qwen3.5" (OPEN — watch the
  first optimizer steps); #6681 Qwen3.5-VL Megatron TP restriction (use FSDP).

Fallback: **ms-swift** — explicit "Multimodal Data Override" support for
multi-turn GRPO with images plus a Qwen3.5 best-practice doc; no ~27B
end-to-end example. TRL GRPOTrainer supports VLM+LoRA but its tested VLM list
stops at Qwen2.5-VL (no qwen3_5) and images only in the first prompt.

**Honest precedent statement: NO framework has published precedent for the
full combination** (multi-turn trajectories with interleaved images + LoRA +
~27B VLM — let alone the qwen3_5 hybrid 3:1 linear/full-attention stack).
Closest published pieces: DeepEyes recipe (multi-turn interleaved-image GRPO,
Qwen2.5-VL-7B, full-parameter, https://arxiv.org/abs/2505.14362); verl #6490
(LoRA + GSPO on Qwen3-Omni-30B on 4xH100); verl qwen3_5 GRPO scripts
(exact arch, text-prompt). Stage A would be the first assembly of all three —
that is precisely why the K4 kill criterion exists.

## Top risks

1. **No published precedent for the exact combination** (multi-turn tool-use
   trajectories with interleaved images + LoRA + 27B VLM GRPO) — see research
   section; expect glue code, budget the whole trainer day for it.
2. **Token-faithfulness across the vLLM/trainer boundary**: chat template,
   `enable_thinking`, and image tokenization must match exactly between the
   serving path and the trainer's re-tokenization; silent mismatch = training
   on off-policy tokens. Mitigation: the K3 usage-token cross-check.
3. **Reward sparsity on dev games**: base duck completes 0 levels on many
   games; all-zero groups give zero advantage. Mitigation: pick the games the
   duck sometimes solves (group variance > 0), or shape with per-turn
   level-delta from tool payloads.
