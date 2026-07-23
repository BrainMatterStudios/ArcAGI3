#!/bin/sh
# upload_data.sh — create/version the Kaggle datasets for the SFT-K3 training kernel.
# DO NOT run without confirming: this writes to Kaggle. Build/validate first (see RUNBOOK.md).
set -eu
cd "$(dirname "$0")"

# (a) SFT corpus: train/val jsonl + tools.json + sft_common.py + tokenizer_bundle
#     (bundle = tokenizer/chat_template/processor_config pulled from vrfai/Qwen3.6-27B-FP8;
#      in-kernel fallback if the model snapshot dataset lacks processor files)
if [ "${1:-}" = "create" ]; then
  kaggle datasets create -p corpus --dir-mode zip
else
  kaggle datasets version -p corpus --dir-mode zip -m "sft corpus $(date +%Y-%m-%d) ($(wc -l < corpus/train.jsonl | tr -d ' ') train / $(wc -l < corpus/val.jsonl | tr -d ' ') val)"
fi

# (b) deps: NO new Kaggle dataset needed. The July bundle (kernel ahmedmobasher86/arc3-deps-prep,
#     attached via kernel_sources) already carries transformers-main (5.14.0.dev0 > required 5.6.2)
#     + peft + trl + accelerate + datasets + compressed-tensors + torchao>=0.17, and the qwen3_5
#     VL classes (Qwen3_5ForConditionalGeneration / Qwen3VLProcessor) exist there. torchvision +
#     Pillow come from the Kaggle GPU image (the bundle deliberately strips torch/torchvision).
#     If the model-load cell fails on missing qwen3_5 classes, REFRESH the bundle by re-running:
#       kaggle kernels push -p ../_depsprep && kaggle kernels status ahmedmobasher86/arc3-deps-prep
echo "corpus dataset pushed. Deps bundle: reuse arc3-deps-prep (see comments)."
