"""build_sft_k3.py — emit sft-k3.ipynb + kernel-metadata.json (K3 teacher-distillation SFT).

Offline LoRA fine-tune of Qwen3.6-27B-VL on teacher win-trajectories, Kaggle RTX Pro 6000.
Run: .venv/bin/python submission/_sft_k3/build_sft_k3.py   (build only; push via RUNBOOK)
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

MD_HEADER = """\
# ARC-3 SFT K3 — LoRA Qwen3.6-27B-VL on teacher (Kimi-K3) win-trajectories (OFFLINE)

July text-LoRA recipe (deps bundle / FP8-upcast / r=16) + multimodal delta. **Memory math @ 96GB, L=32768, batch 1:**

| item | GB |
|---|---|
| weights bf16 (27B text + 0.44B vision) | ~55 |
| LoRA r=16 (~0.23B): adapter + fp32 grads + AdamW m,v | ~3.3 |
| grad-ckpt layer boundaries: 64 x L x 5120 x 2B | 21.5 |
| logits via `logits_to_keep=n_target+1` (<=4K pos, fp32 CE) | ~6 |
| recompute workspace + misc | ~4 |
| **total** | **~90** |

Full-vocab logits at 32K would be ~49GB alone -> `logits_to_keep` + manual CE is what makes 32K fit.
Loss = target assistant turn only (mask verified by `test_loss_mask.py`). preserve_thinking=True matches
vLLM serving.

**2026-07-24 OOM fix (run 1 died at step 5):** config-level quant-stripping left compressed-tensors'
INSTANCE-level `forward` wrappers live -> weight fake-quant (QDQ) ran every forward with big fp32 clamp
temporaries, blowing the ~1GB margin. `strip_quantization_runtime` (sft_common, unit-tested against
compressed-tensors 0.17.1) now unwraps them so forward is pure dense bf16. Until a clean run reports the
true peak (printed per optim step), default is the safety fallback `SFT_MAX_LEN=24576` (-5.4GB);
restore 32768 via env after verifying headroom. This run RESTARTS from step 0: checkpoint-4 sits only in
the dead run's output AND was trained under QDQ forward semantics — resuming it under the changed
(pure-bf16) forward would mix inconsistent dynamics; 4 steps is cheap to redo.
"""

C_GUARD = """\
import glob, json, os, subprocess, sys
deps = sorted(glob.glob("/kaggle/input/**/deps", recursive=True))
if deps: sys.path.insert(0, deps[0])
print("deps on path:", deps[:1])
import torch
DEV = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
# CPU-safe guard (July pattern): interactive/commit sessions may get P100/CPU -> print plan, skip training
TRAIN = "RTX PRO 6000" in DEV.upper() or os.environ.get("SFT_FORCE") == "1"
print(f"device: {DEV} | TRAIN={TRAIN}")
import transformers, peft
print("transformers", transformers.__version__, "| peft", peft.__version__)
if TRAIN:
    assert transformers.__version__.split(".")[:2] >= ["5", "6"] or "dev" in transformers.__version__, \\
        "qwen3_5 needs transformers>=5.6.2 — refresh the arc3-deps-prep bundle"
MAX_LEN = int(os.environ.get("SFT_MAX_LEN", 24576))  # safety default after 07-23 OOM; 32768 once true peak known
EPOCHS = float(os.environ.get("SFT_EPOCHS", 0.30))  # 07-26 run-6 bring-up: ~14 steps completes inside the pre-reset quota AND banks the A1-favored early checkpoints (8 + final ~14) on the FIXED base; raise toward 2 only after the checkpoint sweep says deeper helps
"""

C_INPUTS = """\
CORPUS = os.path.dirname(sorted(glob.glob("/kaggle/input/**/train.jsonl", recursive=True))[0])
sys.path.insert(0, CORPUS)
from sft_common import encode_with_mask
MODEL = next(os.path.dirname(p) for p in glob.glob("/kaggle/input/**/config.json", recursive=True)
             if "tokenizer_bundle" not in p and json.load(open(p)).get("model_type") == "qwen3_5")
print("corpus:", CORPUS, "| model:", MODEL)
from transformers import AutoProcessor
try:
    proc = AutoProcessor.from_pretrained(MODEL)
except Exception as e:  # snapshot missing processor files -> corpus fallback (same repo files)
    print("processor from snapshot failed:", e)
    proc = AutoProcessor.from_pretrained(os.path.join(CORPUS, "tokenizer_bundle"))
print("processor:", type(proc).__name__)
TOOLS = json.load(open(os.path.join(CORPUS, "tools.json")))
train_rows = [json.loads(l) for l in open(os.path.join(CORPUS, "train.jsonl"))]
val_rows = [json.loads(l) for l in open(os.path.join(CORPUS, "val.jsonl"))]
print(f"{len(train_rows)} train / {len(val_rows)} val samples")
# smoke: one sample end-to-end through the REAL processor before touching the GPU
_f, _i = encode_with_mask(proc, train_rows[0]["messages"], train_rows[0]["target"], TOOLS, MAX_LEN)
assert _f is not None and _i["n_target"] > 0, _i
print("encode smoke:", _i)
"""

C_MODEL = """\
if TRAIN:
    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map={"": 0})
    for a in ("quantization_config", "_pre_quantization_dtype"):
        if hasattr(model.config, a):
            try: setattr(model.config, a, None)
            except Exception: pass
    model.is_quantized = False
    if hasattr(model, "hf_quantizer"): model.hf_quantizer = None
    model.config.use_cache = False
    # 07-26 ROOT-CAUSE FIX (runs 1-5 invalid): the VL snapshot loads with 256 Linears kept as
    # f8e4m3 STORAGE (w_true/scale). Scales MUST be applied before strip deletes them —
    # runs 1-5 trained against scale-less casts (base NLL ~14.7 nats = worse than uniform).
    from sft_common import dequantize_fp8_inplace, strip_quantization_runtime
    dstats = dequantize_fp8_inplace(model)
    print("dequantize_fp8_inplace:", dstats)
    # 07-24 OOM fix: compressed-tensors also leaves instance-level QDQ forward wrappers.
    qstats = strip_quantization_runtime(model)
    print("strip_quantization_runtime:", qstats)
    assert qstats["quantized_modules"] > 0, \\
        "no quantized modules found — load path changed, verify forwards are clean before training"
    from collections import Counter
    dtypes = Counter(str(p.dtype) for p in model.parameters())
    print("param dtypes:", dtypes)
    assert "torch.float8_e4m3fn" not in dtypes, "f8 params survived — dequant failed"
    n_par = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"loaded {model.config.model_type}: {n_par:.1f}B params, class {type(model).__name__}")

    # LoRA r=16 on LANGUAGE linears only — vision tower frozen (no adapters, no grads)
    import torch.nn as nn
    from peft import LoraConfig, get_peft_model
    SKIP = ("visual", "lm_head", "mtp", "embed")
    targets = sorted({n for n, m in model.named_modules()
                      if isinstance(m, nn.Linear) and not any(s in n for s in SKIP)})
    print(f"{len(targets)} LoRA target linears (e.g. {targets[:3]} ... {targets[-2:]})")
    assert targets and not any("visual" in t for t in targets)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=targets, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    model.enable_input_require_grads()
else:
    print("PLAN: load bf16 27B-VL, strip quant, LoRA r=16 on language linears, train 3 epochs")
"""

C_TRAINER = """\
if TRAIN:
    from torch.utils.data import Dataset
    from transformers import Trainer, TrainerCallback, TrainingArguments
    from transformers.trainer_utils import get_last_checkpoint
    import shutil, torch.nn.functional as F

    class PeakMem(TrainerCallback):
        # per-optim-step peak so the next log quantifies the real memory budget
        def on_step_end(self, args, state, control, **kw):
            print(f"[step {state.global_step}] peak GPU mem "
                  f"{torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)
            torch.cuda.reset_peak_memory_stats()

    class Rows(Dataset):
        def __init__(self, rows): self.rows = rows
        def __len__(self): return len(self.rows)
        def __getitem__(self, i): return self.rows[i]

    def collate(batch):
        r = batch[0]  # batch size 1 (memory budget; variable-length multimodal)
        feats, info = encode_with_mask(proc, r["messages"], r["target"], TOOLS, MAX_LEN)
        assert feats is not None, info
        if "pixel_values" in feats: feats["pixel_values"] = feats["pixel_values"].to(torch.bfloat16)
        return feats

    class TargetLossTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            n_tgt = int((labels != -100).sum())
            out = model(**inputs, logits_to_keep=n_tgt + 1, use_cache=False)
            logits = out.logits[:, :-1, :]          # predict the last n_tgt tokens
            tgt = labels[:, -n_tgt:].to(logits.device)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), tgt.reshape(-1))
            return (loss, out) if return_outputs else loss

    OUT = "/kaggle/working/sft_out"
    # 12h-cap resume: attach the previous run's output as a dataset/kernel source; checkpoints copy in
    prev = sorted(glob.glob("/kaggle/input/**/sft_out/checkpoint-*", recursive=True))
    if prev and not os.path.exists(OUT):
        os.makedirs(OUT, exist_ok=True)
        for p in prev: shutil.copytree(p, os.path.join(OUT, os.path.basename(p)))
        print("resuming from copied checkpoints:", [os.path.basename(p) for p in prev])
    if not prev:
        # PRE-FLIGHT BASE-HEALTH GATE (fresh runs only; lora_B=0 -> adapter is identity, so this
        # measures the BASE). Runs 1-5 would have failed here at ~14.7. Never train a broken base.
        model.train(False)
        with torch.no_grad():
            nlls = []
            for r in val_rows[:2]:
                feats, _ = encode_with_mask(proc, r["messages"], r["target"], TOOLS, MAX_LEN)
                if "pixel_values" in feats: feats["pixel_values"] = feats["pixel_values"].to(torch.bfloat16)
                labels = feats.pop("labels")
                n_tgt = int((labels != -100).sum())
                feats = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in feats.items()}
                out = model(**feats, logits_to_keep=n_tgt + 1, use_cache=False)
                logits = out.logits[:, :-1, :]
                tgt = labels[:, -n_tgt:].to(logits.device)
                nlls.append(float(F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), tgt.reshape(-1))))
        base_nll = sum(nlls) / len(nlls)
        print(f"PRE-FLIGHT base per-token NLL (2 val rows): {base_nll:.3f}  {nlls}")
        assert base_nll < 6.0, f"BASE MODEL BROKEN (NLL {base_nll:.2f}) — fix the load path, do not train"
        model.train(True)
    args = TrainingArguments(
        output_dir=OUT, num_train_epochs=EPOCHS,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        learning_rate=1e-4, lr_scheduler_type="cosine", warmup_ratio=0.03,
        bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1, save_strategy="steps", save_steps=8, save_total_limit=8,
        remove_unused_columns=False, dataloader_num_workers=0, report_to="none")
    trainer = TargetLossTrainer(model=model, args=args, train_dataset=Rows(train_rows),
                                data_collator=collate, callbacks=[PeakMem()])
    ckpt = get_last_checkpoint(OUT) if os.path.isdir(OUT) else None
    print("resume checkpoint:", ckpt)
    trainer.train(resume_from_checkpoint=ckpt)
    trainer.save_model("/kaggle/working/sft_adapter")
    proc.save_pretrained("/kaggle/working/sft_adapter")
    print("adapter saved:", os.listdir("/kaggle/working/sft_adapter"))
"""

C_VAL = """\
if TRAIN:
    # val loss adapter ON vs OFF — in-kernel proof the adapter changes the model
    # (July failure = a LoRA that never served; the serving gate proper is in the RUNBOOK)
    def val_loss():
        model.train(False); tot = n = 0
        with torch.no_grad():
            for r in val_rows:
                feats, _ = encode_with_mask(proc, r["messages"], r["target"], TOOLS, MAX_LEN)
                if "pixel_values" in feats: feats["pixel_values"] = feats["pixel_values"].to(torch.bfloat16)
                labels = feats.pop("labels")
                n_tgt = int((labels != -100).sum())
                feats = {k: v.to(model.device) for k, v in feats.items()}
                out = model(**feats, logits_to_keep=n_tgt + 1, use_cache=False)
                loss = F.cross_entropy(out.logits[:, :-1, :].reshape(-1, out.logits.size(-1)).float(),
                                       labels[:, -n_tgt:].to(model.device).reshape(-1))
                tot += float(loss); n += 1
        return tot / n
    tuned = val_loss()
    with model.disable_adapter():
        base = val_loss()
    print(f"VAL target-loss  base={base:.4f}  tuned={tuned:.4f}  (tuned must differ AND be lower)")
    assert abs(base - tuned) > 1e-4, "adapter changed nothing — investigate before serving"
"""

C_SUBMIT = """\
import pandas as pd
pd.DataFrame([["1_0", "1", True, 0]],
             columns=["row_id", "game_id", "end_of_game", "score"]).to_parquet(
    "/kaggle/working/submission.parquet", index=False)
print("done; outputs:", os.listdir("/kaggle/working"))
"""


def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "cells": [cell("markdown", MD_HEADER), cell("code", C_GUARD), cell("code", C_INPUTS),
                cell("code", C_MODEL), cell("code", C_TRAINER), cell("code", C_VAL),
                cell("code", C_SUBMIT)]}
(HERE / "sft-k3.ipynb").write_text(json.dumps(nb, indent=1))

meta = {
    "id": "ahmedmobasher86/arc-agi-3-sft-k3",
    "title": "arc-agi-3-sft-k3",
    "code_file": "sft-k3.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True,
    "enable_gpu": True, "enable_internet": False,
    "dataset_sources": ["ahmedmobasher86/arc3-sft-k3-corpus",
                        "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"],
    "competition_sources": ["arc-prize-2026-arc-agi-3"],
    # resume workflow: Kaggle REJECTS self-referencing kernel_sources (proven 07-26).
    # To resume a cap-killed run: download its sft_out/checkpoint-* via the API file_pattern
    # trick, push as a version of dataset ahmedmobasher86/arc3-sft-k3-ckpts, and add that
    # dataset to dataset_sources — the /kaggle/input glob picks it up.
    "kernel_sources": ["ahmedmobasher86/arc3-deps-prep"],
}
(HERE / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("wrote sft-k3.ipynb + kernel-metadata.json")
