#!/bin/sh
# One-shot: download RedHatAI/Qwen3.6-27B-FP8 (~30.89 GB, 80 files) and create
# the Kaggle dataset per submission/_quant_probe/UPLOAD_PLAN.md.
# Portable sh (bash 3.2-safe). Logs to stdout; run under nohup.
set -e
REPO="/Users/ahmed/Documents/ArcAGI3"
DEST="$HOME/models/redhatai-qwen3.6-27b-fp8"
EXPECTED_BYTES=30890041440

echo "[quant-upload] $(date -u) step 1: download to $DEST"
mkdir -p "$DEST"
"$REPO/.venv/bin/python" -c "
from huggingface_hub import snapshot_download
snapshot_download('RedHatAI/Qwen3.6-27B-FP8', local_dir='$DEST', max_workers=4)
print('snapshot_download complete')
"

FILES=$(find "$DEST" -type f -not -path "*/.cache/*" -not -name "dataset-metadata.json" | wc -l | tr -d ' ')
BYTES=$(find "$DEST" -type f -not -path "*/.cache/*" -not -name "dataset-metadata.json" -exec stat -f "%z" {} + | awk '{s+=$1} END {print s}')
echo "[quant-upload] $(date -u) downloaded: $FILES files, $BYTES bytes (expect ~80 files, $EXPECTED_BYTES bytes)"
if [ "$BYTES" -lt 30000000000 ]; then
    echo "[quant-upload] ABORT: byte count too low — incomplete download"
    exit 1
fi
# hub cache metadata must not go into the dataset
rm -rf "$DEST/.cache"

echo "[quant-upload] $(date -u) step 2: kaggle dataset create (tar mode)"
cd "$DEST"
cat > dataset-metadata.json <<'EOF'
{
  "title": "RedHatAI Qwen3.6 27B FP8 Hugging Face Snapshot",
  "id": "ahmedmobasher86/redhatai-qwen3-6-27b-fp8-hf-snapshot",
  "licenses": [{"name": "apache-2.0"}]
}
EOF
if ! python3 -m kaggle datasets create -p . --dir-mode tar; then
    echo "[quant-upload] tar-mode create failed — retrying per-file mode"
    python3 -m kaggle datasets create -p .
fi
echo "[quant-upload] $(date -u) create submitted; polling status"
n=0
while [ $n -lt 120 ]; do
    STATUS=$(python3 -m kaggle datasets status ahmedmobasher86/redhatai-qwen3-6-27b-fp8-hf-snapshot 2>&1 || true)
    echo "[quant-upload] $(date -u) status: $STATUS"
    case "$STATUS" in
        *ready*|*complete*) echo "[quant-upload] DONE"; exit 0 ;;
        *error*|*failed*) echo "[quant-upload] SERVER-SIDE FAILURE"; exit 1 ;;
    esac
    n=$((n+1))
    sleep 300
done
echo "[quant-upload] gave up polling after 10h (upload may still be processing)"
exit 2
