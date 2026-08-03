"""build_sft_synth.py — emit sft-synth-v1.ipynb + kernel-metadata.json.

STAGE-2 PIPELINE VALIDATION of the synthetic-corpus SFT track (synthgen commit
0553b61): LoRA fine-tune Qwen3.6-27B-VL on corpus_synth_v1 train.jsonl (nav+click
families only) and measure paired base-vs-adapter NLL on val.jsonl (in-family) AND
holdout_transfer.jsonl (held-out push family). The paired comparison is the
deliverable; nothing here ships to a submission.

Adapted COPY of the proven submission/_sft_k3/build_sft_k3.py (kernel v8 = run 8,
the first valid adapter). Deltas from that lineage, everything else untouched:
  * corpus dataset -> ahmedmobasher86/arc3-corpus-synth-v1 (504 train / 99 val /
    185 holdout_transfer; byte-identical sft_common.py + tokenizer_bundle)
  * EPOCHS default 0.25 (~15 optim steps on 504 rows) — OOD peaks early
    (deep-research A1); save_steps=5 banks early/mid/final checkpoints (5/10/15)
  * post-train eval: paired per-row NLL (base via disable_adapter vs tuned) on
    val AND holdout_transfer slices, with per-family breakdown + RESULT json line
Kept verbatim: FP8 fix (fp8_scaled_linear_inplace + 256-linear assert), PRE-FLIGHT
base-NLL gate (<6.0 nats or abort), CPU-safe guard, TargetLossTrainer memory math.

Run: .venv/bin/python submission/_sft_synth/build_sft_synth.py   (build only)
Push: cd submission/_sft_synth && kaggle kernels push --accelerator NvidiaRtxPro6000
      (RTX Pro 6000 96GB needs competition attached + that flag + internet off —
       July-proven; the metadata machine_shape field does NOT select it.)
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

MD_HEADER = """\
# ARC-3 SFT synth-v1 — LoRA Qwen3.6-27B-VL on the SYNTHETIC corpus (pipeline validation)

Train on corpus_synth_v1 `train.jsonl` (nav+click, 504 rows), then paired base-vs-adapter
target-NLL on `val.jsonl` (in-family, 99 rows) and `holdout_transfer.jsonl` (held-out push
family, 185 rows). Recipe = the proven sft-k3 v8 lineage: fp8_scaled_linear_inplace,
pre-flight base-NLL gate, LoRA r=16 language-linears-only, logits_to_keep target loss.

Pre-registered readout (from the kernel log):
* pipeline VALIDATED = training completes + tuned val NLL < base val NLL
* TRANSFER signal   = tuned holdout NLL < base holdout NLL (bonus, not required)
* neither implies shipping anything.

Budget: ~15 optim steps (EPOCHS 0.25), checkpoints at 5/10/15, eval slices 48 val +
64 holdout x 2 arms. Target <= 5h GPU.
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
MAX_LEN = int(os.environ.get("SFT_MAX_LEN", 24576))  # proven-safe budget from the k3 lineage
EPOCHS = float(os.environ.get("SFT_EPOCHS", 0.25))  # 504 rows / accum 8 -> ~15 optim steps; OOD peaks early (A1); raise only if the paired eval says deeper helps
EVAL_VAL_N = int(os.environ.get("SFT_EVAL_VAL_N", 48))    # paired eval slice, val.jsonl
EVAL_HOLD_N = int(os.environ.get("SFT_EVAL_HOLD_N", 64))  # paired eval slice, holdout_transfer.jsonl
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
hold_rows = [json.loads(l) for l in open(os.path.join(CORPUS, "holdout_transfer.jsonl"))]
print(f"{len(train_rows)} train / {len(val_rows)} val / {len(hold_rows)} holdout_transfer samples")
fams = lambda rows: sorted({r["meta"].get("family") for r in rows})
print("families train:", fams(train_rows), "val:", fams(val_rows), "holdout:", fams(hold_rows))
assert "push" not in fams(train_rows) and fams(hold_rows) == ["push"], "holdout split integrity"
# smoke: one sample per split end-to-end through the REAL processor before touching the GPU
for name, rows in (("train", train_rows), ("val", val_rows), ("holdout", hold_rows)):
    _f, _i = encode_with_mask(proc, rows[0]["messages"], rows[0]["target"], TOOLS, MAX_LEN)
    assert _f is not None and _i["n_target"] > 0, (name, _i)
    print(f"encode smoke {name}:", _i)
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
    # 07-26 ROOT-CAUSE FIX (k3 runs 1-5 invalid): the VL snapshot keeps 256 Linears as
    # f8e4m3 STORAGE (w_true/scale); stripping the scales unapplied -> base NLL ~14.7.
    # Keep f8 storage, correct SCALED forward (bit-identical math, ~0.4GB transient).
    from sft_common import fp8_scaled_linear_inplace
    qstats = fp8_scaled_linear_inplace(model)
    print("fp8_scaled_linear_inplace:", qstats)
    assert qstats["fp8_scaled_linears"] == 256, qstats
    from collections import Counter
    dtypes = Counter(str(p.dtype) for p in model.parameters())
    print("param dtypes:", dtypes)
    assert dtypes.get("torch.float8_e4m3fn") == 256, "expected 256 f8-storage linears"
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
    print("PLAN: load bf16 27B-VL, fp8 scaled forward, LoRA r=16, train EPOCHS", EPOCHS)
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

    def row_nll(r):
        feats, _ = encode_with_mask(proc, r["messages"], r["target"], TOOLS, MAX_LEN)
        if "pixel_values" in feats: feats["pixel_values"] = feats["pixel_values"].to(torch.bfloat16)
        labels = feats.pop("labels")
        n_tgt = int((labels != -100).sum())
        feats = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in feats.items()}
        out = model(**feats, logits_to_keep=n_tgt + 1, use_cache=False)
        logits = out.logits[:, :-1, :]
        tgt = labels[:, -n_tgt:].to(logits.device)
        return float(F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), tgt.reshape(-1)))

    OUT = "/kaggle/working/sft_out"
    # fresh run expected (no ckpts dataset attached); resume path kept from the k3 lineage
    prev = sorted(glob.glob("/kaggle/input/**/sft_out/checkpoint-*", recursive=True))
    if prev and not os.path.exists(OUT):
        os.makedirs(OUT, exist_ok=True)
        for p in prev: shutil.copytree(p, os.path.join(OUT, os.path.basename(p)))
        print("resuming from copied checkpoints:", [os.path.basename(p) for p in prev])
    if not prev:
        # PRE-FLIGHT BASE-HEALTH GATE (lora_B=0 -> adapter is identity, so this measures
        # the BASE). k3 runs 1-5 would have failed here at ~14.7. Never train a broken base.
        model.train(False)
        with torch.no_grad():
            nlls = [row_nll(r) for r in val_rows[:2]]
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
        logging_steps=1, save_strategy="steps", save_steps=5, save_total_limit=8,
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

C_EVAL = """\
if TRAIN:
    # PAIRED base-vs-adapter eval — THE deliverable of this kernel.
    # Same rows, same greedy NLL, adapter arm = final adapter; base arm = disable_adapter().
    import time
    def eval_rows(rows):
        out = []
        with torch.no_grad():
            for r in rows:
                out.append((r["meta"].get("family", "?"), row_nll(r)))
        return out
    def spread(rows, n):
        # evenly-spaced deterministic slice — the jsonl is grouped by game, so a
        # head-slice would cover few games; this spans the whole file
        if len(rows) <= n: return rows
        return [rows[round(i * len(rows) / n)] for i in range(n)]
    model.train(False)
    t0 = time.time()
    slices = {"val": spread(val_rows, EVAL_VAL_N), "holdout_transfer": spread(hold_rows, EVAL_HOLD_N)}
    tuned = {k: eval_rows(v) for k, v in slices.items()}
    print(f"tuned eval done in {time.time()-t0:.0f}s", flush=True)
    with model.disable_adapter():
        base = {k: eval_rows(v) for k, v in slices.items()}
    print(f"both arms done in {time.time()-t0:.0f}s", flush=True)
    result = {"epochs": EPOCHS, "n_train": len(train_rows)}
    for k in slices:
        b = [x[1] for x in base[k]]; t = [x[1] for x in tuned[k]]
        wins = sum(1 for bb, tt in zip(b, t) if tt < bb)
        result[k] = {"n": len(b), "base_nll": sum(b) / len(b), "tuned_nll": sum(t) / len(t),
                     "delta": sum(t) / len(t) - sum(b) / len(b), "row_wins_tuned": wins}
        for fam in sorted({f for f, _ in base[k]}):
            fb = [x for f, x in base[k] if f == fam]; ft = [x for f, x in tuned[k] if f == fam]
            result[k][f"family_{fam}"] = {"n": len(fb), "base_nll": sum(fb) / len(fb),
                                          "tuned_nll": sum(ft) / len(ft)}
        print(f"EVAL {k}: base={result[k]['base_nll']:.4f} tuned={result[k]['tuned_nll']:.4f} "
              f"delta={result[k]['delta']:+.4f} row_wins={result[k]['row_wins_tuned']}/{len(b)}")
    assert abs(result["val"]["delta"]) > 1e-4, "adapter changed nothing — investigate before trusting this run"
    result["pipeline_validated"] = bool(result["val"]["delta"] < 0)
    result["transfer_signal"] = bool(result["holdout_transfer"]["delta"] < 0)
    print("RESULT " + json.dumps(result))
    json.dump(result, open("/kaggle/working/eval_result.json", "w"), indent=2)
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
                cell("code", C_MODEL), cell("code", C_TRAINER), cell("code", C_EVAL),
                cell("code", C_SUBMIT)]}
(HERE / "sft-synth-v1.ipynb").write_text(json.dumps(nb, indent=1))

meta = {
    "id": "ahmedmobasher86/arc-agi-3-sft-synth-v1",
    "title": "arc-agi-3-sft-synth-v1",
    "code_file": "sft-synth-v1.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True,
    "enable_gpu": True, "enable_internet": False,
    "dataset_sources": ["ahmedmobasher86/arc3-corpus-synth-v1",
                        "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"],
    "competition_sources": ["arc-prize-2026-arc-agi-3"],
    "kernel_sources": ["ahmedmobasher86/arc3-deps-prep"],
}
(HERE / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("wrote sft-synth-v1.ipynb + kernel-metadata.json")
