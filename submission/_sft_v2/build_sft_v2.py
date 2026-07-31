#!/usr/bin/env python3
"""build_sft_v2.py — emit sft-v2.ipynb + kernel-metadata.json for Zero-Leakage SFT-v2.

Offline LoRA fine-tune of Qwen3.6-27B-VL on teacher win-trajectories using the
zero-leakage episode-family corpus (ahmedmobasher86/arc3-sft-k3-corpus-v2).
"""
import json
from pathlib import Path

REPO = Path("/Users/ahmed/Documents/ArcAGI3")
HERE = Path(__file__).resolve().parent
OUT = HERE

MD_HEADER = """\
# ARC-3 SFT v2 — Zero-Leakage Episode-Isolated LoRA Fine-Tuning of Qwen3.6-27B-VL

Key Upgrades in SFT-v2:
1. **Zero-Leakage Dataset:** Uses `ahmedmobasher86/arc3-sft-k3-corpus-v2` with 0% train/val episode family overlap.
2. **In-Place FP8 Forward Math:** Uses `fp8_scaled_linear_inplace` for bit-exact forward passes without RAM OOM.
3. **Multi-Turn Target Loss:** Evaluates target-token NLL across full trajectory assistant outputs.
"""

C_GUARD = """\
import glob, json, os, subprocess, sys
deps = sorted(glob.glob("/kaggle/input/**/deps", recursive=True))
if deps: sys.path.insert(0, deps[0])
print("deps on path:", deps[:1])
import torch
DEV = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
TRAIN = "RTX PRO 6000" in DEV.upper() or os.environ.get("SFT_FORCE") == "1"
print(f"device: {DEV} | TRAIN={TRAIN}")
import transformers, peft
print("transformers", transformers.__version__, "| peft", peft.__version__)
MAX_LEN = int(os.environ.get("SFT_MAX_LEN", 24576))
EPOCHS = float(os.environ.get("SFT_EPOCHS", 0.50))
"""

C_INPUTS = """\
CORPUS = os.path.dirname(sorted(glob.glob("/kaggle/input/**/train.jsonl", recursive=True))[0])
sys.path.insert(0, CORPUS)
from sft_common import encode_with_mask
MODEL = next(os.path.dirname(p) for p in glob.glob("/kaggle/input/**/config.json", recursive=True)
             if "tokenizer_bundle" not in p and json.load(open(p)).get("model_type") == "qwen3_5")
print("corpus_v2:", CORPUS, "| model:", MODEL)
from transformers import AutoProcessor
try:
    proc = AutoProcessor.from_pretrained(MODEL)
except Exception as e:
    print("processor from snapshot failed:", e)
    proc = AutoProcessor.from_pretrained(os.path.join(CORPUS, "tokenizer_bundle"))
print("processor:", type(proc).__name__)
TOOLS = json.load(open(os.path.join(CORPUS, "tools.json")))
train_rows = [json.loads(l) for l in open(os.path.join(CORPUS, "train.jsonl"))]
val_rows = [json.loads(l) for l in open(os.path.join(CORPUS, "val.jsonl"))]
print(f"Zero-Leakage Corpus v2: {len(train_rows)} train / {len(val_rows)} val samples")
_f, _i = encode_with_mask(proc, train_rows[0]["messages"], train_rows[0]["target"], TOOLS, MAX_LEN)
assert _f is not None and _i["n_target"] > 0, _i
print("encode smoke test PASS:", _i)
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

    from sft_common import fp8_scaled_linear_inplace
    qstats = fp8_scaled_linear_inplace(model)
    print("fp8_scaled_linear_inplace:", qstats)
    assert qstats["fp8_scaled_linears"] == 256, qstats

    import torch.nn as nn
    from peft import LoraConfig, get_peft_model
    SKIP = ("visual", "lm_head", "mtp", "embed")
    targets = sorted({n for n, m in model.named_modules()
                      if isinstance(m, nn.Linear) and not any(s in n for s in SKIP)})
    print(f"{len(targets)} LoRA target linears")
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=targets, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
"""

cells = [
    {"cell_type": "markdown", "metadata": {}, "source": MD_HEADER.splitlines(keepends=True)},
    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": C_GUARD.splitlines(keepends=True)},
    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": C_INPUTS.splitlines(keepends=True)},
    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": C_MODEL.splitlines(keepends=True)},
]

nb = {
    "metadata": {
        "language_info": {"name": "python"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
    },
    "nbformat": 4, "nbformat_minor": 4,
    "cells": cells
}

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "sft-v2.ipynb").write_text(json.dumps(nb, indent=1))

meta = {
    "id": "ahmedmobasher86/arc-agi-3-sft-v2",
    "title": "arc-agi-3-sft-v2",
    "code_file": "sft-v2.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": False,
    "machine_shape": "NvidiaRtxPro6000",
    "dataset_sources": [
        "driessmit1/arc3-vllm-h100-wheelhouse-v3",
        "ahmedmobasher86/arc3-sft-k3-corpus-v2",
        "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"
    ],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": []
}

(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print(f"wrote {OUT/'sft-v2.ipynb'} ({len(cells)} cells) and kernel-metadata.json successfully!")
