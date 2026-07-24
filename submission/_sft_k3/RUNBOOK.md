# SFT-K3 RUNBOOK — teacher-distillation LoRA on Kaggle (Qwen3.6-27B-VL)

Nothing here has been pushed to Kaggle yet. Local build + validation only.

## 0. Refresh data (data grows nightly)
```sh
.venv/bin/python scratchpad/rl_gate/harvest_sft.py            # rebuild all_wins.jsonl
.venv/bin/python submission/_sft_k3/prep_dataset.py           # rebuild corpus/ (+ stats)
.venv/bin/python submission/_sft_k3/test_loss_mask.py         # must print ALL ... PASSED
```

## 1. Upload corpus dataset (first time: `create`)
```sh
sh submission/_sft_k3/upload_data.sh create      # later versions: sh upload_data.sh
```
Deps: reuse kernel `ahmedmobasher86/arc3-deps-prep` (transformers-main ≥5.6.2 — has qwen3_5 VL).
No new deps dataset needed (torchvision/Pillow come from the Kaggle image).

## 2. Push + run the training kernel
```sh
.venv/bin/python submission/_sft_k3/build_sft_k3.py
cd submission/_sft_k3 && kaggle kernels push --accelerator NvidiaRtxPro6000
```
RTX Pro 6000 (96GB) needs ALL of: competition attached + `--accelerator NvidiaRtxPro6000`
+ `enable_internet: false` (July-proven; the metadata `machine_shape` field does NOT select it).
Interactive/commit fallback GPUs (P100/T4) hit the CPU-safe guard: plan printed, no training.
Env knobs: `SFT_MAX_LEN` (default **24576** — safety margin after the 07-23 OOM; restore
32768 once a clean run's per-step peak prints show headroom), `SFT_EPOCHS` (default 3).
Expected: ~33 optim steps (88×3 / 8), ~60-90 s/sample → ~5-7 h; per-step peak GiB is printed.

### Post-mortem, run 1 (2026-07-23): OOM at step 5 inside compressed_tensors QDQ
Config-level quant-stripping left compressed-tensors' instance-level `forward` wrappers
live, fake-quantizing weights every forward (fp32 clamp temporaries → OOM with ~1GB free).
Fixed by `sft_common.strip_quantization_runtime` (called right after load; unit test
`test_strip_quant.py` proves against compressed-tensors 0.17.1 that the wrapper is removed
and dense-bf16 forward is restored bit-identical). Run 2 RESTARTS from step 0 — deliberate:
checkpoint-4 lives only in the dead run's output (not auto-preserved) and was trained under
QDQ forward semantics; resuming it under the now-pure-bf16 forward would mix inconsistent
dynamics. 4 steps (~45 min) is cheap to redo.

## 3. Resume after the 12h cap
Checkpoints land in `/kaggle/working/sft_out/checkpoint-N` every 4 optim steps (≤3 kept).
`/kaggle/working` is NOT auto-preserved across runs. To resume: add the dead run's kernel
output as a source (its version output via `kernel_sources`, or a dataset built from
`kaggle kernels output`) and re-push — the notebook globs
`/kaggle/input/**/sft_out/checkpoint-*`, copies into `/kaggle/working/sft_out`, and
`trainer.train(resume_from_checkpoint=...)` continues (optimizer + LR schedule + epoch
state restored by HF Trainer). Only resume checkpoints produced AFTER the QDQ fix.

## 4. Harvest the adapter
Kernel outputs: `sft_adapter/` (adapter_model.safetensors + adapter_config.json + processor files)
+ `sft_out/checkpoint-*`. Download: `kaggle kernels output ahmedmobasher86/arc-agi-3-sft-k3 -p /tmp/sftk3`.
In-kernel gate already asserts: val target-loss (adapter ON) ≠ and < (adapter OFF).

## 5. SERVING-VERIFICATION GATE (the July failure was a LoRA that never served)
Before any submission uses this model, a separate commit kernel must prove the adapter's
weights actually reach the served model:
1. **Merge + requantize**: load bf16 base, `PeftModel.from_pretrained(...).merge_and_unload()`,
   save; FP8-requantize with llmcompressor using the snapshot's own `recipe.yaml`
   (FP8 Linear except `lm_head/embed/visual/linear_attn/*norm` — same ignore list). Publish as a
   new model dataset. (vLLM `--enable-lora` on qwen3_5's `in_proj_a/b/z/qkv` targets is
   UNVERIFIED — memory 2026-07-06 — so merge-then-serve is the primary path, not runtime LoRA.)
2. **Serve + assert diff**: in one commit, serve BASE and MERGED with the duck's exact vLLM
   line (`--tool-call-parser qwen3_coder --reasoning-parser qwen3
   --default-chat-template-kwargs '{"preserve_thinking": true}'`), send the SAME 3 fixed
   prompts (greedy, temperature=0, seed fixed; take 2 from `corpus/val.jsonl` incl. images +
   1 plain-text) to both, and **assert outputs differ on ≥1 prompt AND the merged model still
   emits a well-formed `<tool_call>` python call on all 3**. Identical outputs = the July
   failure again → stop, do not submit.
3. Only then wire the merged snapshot into the duck submission (setup_commands.json model path).

## Gaps that need the GPU to verify (cannot be validated on this Mac)
- ~~Model load on the FP8 VL snapshot~~ PROVEN by run 1 (loaded, trained 4 steps, saved).
- `strip_quantization_runtime` on the REAL 27B in-kernel (unit-proven on a toy model vs
  compressed-tensors 0.17.1; the in-kernel assert `quantized_modules > 0` + per-step peak
  prints will confirm on run 2).
- Real peak memory at 24576 (then 32768) — quantified per optim step from run 2 onward.
- Training throughput (~60-90 s/sample estimate) vs the 12 h cap.
