#!/bin/sh
# One-shot: download Qwen/Qwen3.8-27B-FP8 (official FP8, published 2026-08-14
# 14:44Z, apache-2.0) and create the Kaggle dataset for the duck-38 arm.
# Pattern proven by upload_redhatai_snapshot.sh (2026-08-14). Portable sh
# (bash 3.2-safe). Logs to stdout; run under nohup.
set -e
REPO="/Users/ahmed/Documents/ArcAGI3"
DEST="$HOME/models/qwen3.8-27b-fp8"
MIN_BYTES=25000000000

echo "[qwen38-upload] $(date -u) step 1: download to $DEST"
mkdir -p "$DEST"
export SSL_CERT_FILE=$("$REPO/.venv/bin/python" -m certifi)
"$REPO/.venv/bin/python" -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.8-27B-FP8', local_dir='$DEST', max_workers=4)
print('snapshot_download complete')
"

FILES=$(find "$DEST" -type f -not -path "*/.cache/*" -not -name "dataset-metadata.json" | wc -l | tr -d ' ')
BYTES=$(find "$DEST" -type f -not -path "*/.cache/*" -not -name "dataset-metadata.json" -exec stat -f "%z" {} + | awk '{s+=$1} END {print s}')
echo "[qwen38-upload] $(date -u) downloaded: $FILES files, $BYTES bytes (guard: >= $MIN_BYTES)"
if [ "$BYTES" -lt "$MIN_BYTES" ]; then
    echo "[qwen38-upload] ABORT: byte count too low — incomplete download"
    exit 1
fi
# hub cache metadata must not go into the dataset
rm -rf "$DEST/.cache"

echo "[qwen38-upload] $(date -u) step 2: kaggle dataset create (tar mode)"
cd "$DEST"
cat > dataset-metadata.json <<'EOF'
{
  "title": "Qwen3.8 27B FP8 Hugging Face Snapshot",
  "id": "ahmedmobasher86/qwen3-8-27b-fp8-hf-snapshot",
  "licenses": [{"name": "apache-2.0"}]
}
EOF
if ! python3 -m kaggle datasets create -p . --dir-mode tar; then
    echo "[qwen38-upload] tar-mode create failed — retrying per-file mode"
    python3 -m kaggle datasets create -p .
fi
echo "[qwen38-upload] $(date -u) DONE"
