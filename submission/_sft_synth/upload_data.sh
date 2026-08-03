#!/bin/sh
# upload_data.sh — package tools/synthgen/out/corpus_synth_v1 as the Kaggle dataset
# ahmedmobasher86/arc3-corpus-synth-v1 (train/val/holdout_transfer jsonl + tools.json
# + sft_common.py + tokenizer_bundle + prep_stats.json).
# DO NOT run without confirming: this writes to Kaggle.
# First time: sh upload_data.sh create     Later versions: sh upload_data.sh
set -eu
cd "$(dirname "$0")/../.."
CORPUS=tools/synthgen/out/corpus_synth_v1

# dataset-metadata.json lives in the (gitignored) corpus output dir; (re)write it here
# so the packaging is reproducible from the repo.
cat > "$CORPUS/dataset-metadata.json" <<'EOF'
{
  "title": "arc3-corpus-synth-v1",
  "id": "ahmedmobasher86/arc3-corpus-synth-v1",
  "licenses": [
    {
      "name": "CC0-1.0"
    }
  ]
}
EOF

if [ "${1:-}" = "create" ]; then
  kaggle datasets create -p "$CORPUS" --dir-mode zip
else
  kaggle datasets version -p "$CORPUS" --dir-mode zip -m "corpus_synth_v1 $(date +%Y-%m-%d) ($(wc -l < "$CORPUS/train.jsonl" | tr -d ' ') train / $(wc -l < "$CORPUS/val.jsonl" | tr -d ' ') val / $(wc -l < "$CORPUS/holdout_transfer.jsonl" | tr -d ' ') holdout)"
fi
echo "pushed. Verify: kaggle datasets files ahmedmobasher86/arc3-corpus-synth-v1"
