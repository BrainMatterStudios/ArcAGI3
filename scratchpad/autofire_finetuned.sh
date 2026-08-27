#!/bin/bash
cd /Users/ahmed/Documents/ArcAGI3
# STAGE 1: retry push until the rolling GPU quota frees enough for the commit run
while true; do
  OUT=$(cd submission/_finetuned && kaggle kernels push -p . --accelerator NvidiaRtxPro6000 2>&1)
  echo "[$(date -u +%H:%MZ)] push: $(echo "$OUT" | grep -iE 'pushed|quota|error' | tail -1)"
  echo "$OUT" | grep -qi "successfully pushed" && { echo "PUSHED (quota freed)"; break; }
  echo "$OUT" | grep -qi "quota" || { echo "UNEXPECTED push error -> stop"; exit 1; }
  sleep 1800
done
# STAGE 2: wait for the commit run (LoRA serving + 5 held-out games) to finish
while true; do
  S=$(kaggle kernels status ahmedmobasher86/arc-agi-3-duck-finetuned 2>/dev/null | grep -oE "COMPLETE|ERROR")
  [ -n "$S" ] && break; sleep 120
done
echo "commit -> $S"
[ "$S" != "COMPLETE" ] && { echo "commit ERRORed -> LoRA serving likely failed; NOT submitting, review needed"; exit 0; }
# STAGE 3: commit COMPLETE = LoRA served -> auto-submit at the next open slot
MSG="Fine-tuned 27B: pure Tufa duck + our RFT LoRA adapter (base Qwen3.6-27B + trained adapter, analyzer=policy). Commit validated LoRA serving on held-out. Tests fine-tune vs 1.10 pure-duck baseline."
for i in $(seq 1 80); do
  kaggle competitions submit arc-prize-2026-arc-agi-3 -k ahmedmobasher86/arc-agi-3-duck-finetuned -v 1 -f submission.parquet -m "$MSG" >/dev/null 2>&1
  sleep 25
  kaggle competitions submissions arc-prize-2026-arc-agi-3 2>/dev/null | grep -qi "Fine-tuned 27B" && { echo "FINETUNED_SUBMITTED_OK"; break; }
  sleep 275
done
kaggle competitions submissions arc-prize-2026-arc-agi-3 2>/dev/null | grep -vE "Warning" | head -3
