#!/bin/bash
# Submit duck-sft v4 (run-8 adapter, serving path FIXED) in the 2026-08-01 00:00 UTC slot.
#
# Why this and not duck-effects: the independent research sweep (2026-07-31, 23 agents,
# no access to campaign docs) found the binding constraint on this benchmark is BASE
# MODEL CAPABILITY, not scaffolding — same scaffold, GPT-5.4 -> 8 games solved vs
# GPT-5.5 -> 15; public board spans Opus 4.8 1.5% to Opus 5 30.2%, a ~20x spread against
# the +/-0.1-0.3 that prompt arms argue over. It also undercut B1 specifically:
# board_changed has NO control-flow consumers (9 call sites, one advisory sentence) and
# the prompt already exposes previous_frame/transitions/segmentation.
#
# The adapter is the only capability lever we hold, and it has NEVER been served:
#   serve-verify-k3 v5-v8, checkpoint-8, 5536 target tokens, full 32768 ctx:
#     base 0.8497 | attached 0.7329 | merged 0.7350
#     RETAINED 98.2% | merged relative gain 13.51% (A1 §1 bar: >= 2%) -> MERGE GATE PASS
# duck-sft's earlier 0.95 was a base draw: the vLLM model swap targeted a string absent
# from setup_commands.json, so base was served. Fixed in 7bc3cfc (anchored on the real
# MODEL_PATH assignment) with a hard assert that REFUSES to run if the merged tree is
# missing — a failed merge now errors instead of silently scoring base.
#
# ONE change vs base (the adapter). Do not stack.
cd /Users/ahmed/Documents/ArcAGI3 || exit 1

MARKER="SFT-run8-ckpt8-first-served"
MSG="Duck-SFT v4: run-8 LoRA (checkpoint-8) merged in-kernel and served. The prior 0.95 was a BASE draw — the vLLM model-path swap targeted a string absent from setup_commands.json, so base was served; now anchored on the real MODEL_PATH assignment and hard-asserted (a missing merge errors instead of silently scoring base). Gate 0 (serve-verify-k3 v9) evidence: merged target-NLL gain 13.51%, 98.2% of the attached gain retained through the merge (base 0.8497 / attached 0.7329 / merged 0.7350, 5536 target tokens); in-kernel merge+save rc=0 in 390s; vLLM reached 'Application startup complete' on the merged tree under the duck's exact serve line. NOT verified: well-formed generation from the merged model (gate probe hit a payload-schema 400 before generating, and weekly GPU quota is exhausted). [$MARKER] Single-variable experiment: adapter only, no prompt, budget or concurrency changes."

while true; do
  H=$(date -u +%H); M=$(date -u +%M)
  if [ "$H" = "00" ] && [ "$M" -ge 5 ]; then break; fi
  sleep 60
done
echo "[sft] slot window reached at $(date -u '+%F %H:%M:%S UTC')"

# Distinguish "cannot read the status" from "status is bad". The 2026-08-01 run lost
# 11 hours because a transient empty reply from `kaggle kernels status` was treated as
# not-COMPLETE and aborted on a single reading. An unknown is a retry; only a status
# that positively reads as ERROR/CANCEL is a reason not to spend the slot.
STATUS=""
for attempt in $(seq 1 10); do
  STATUS=$(kaggle kernels status ahmedmobasher86/arc-agi-3-duck-sft 2>/dev/null | tail -1)
  echo "[sft] status attempt $attempt: '${STATUS:-<empty>}'"
  case "$STATUS" in
    *COMPLETE*) break;;
    *ERROR*|*CANCEL*)
      echo "[sft] ABORT — kernel reports a terminal failure, refusing to spend the slot"
      exit 1;;
  esac
  sleep 60
done
case "$STATUS" in
  *COMPLETE*) echo "[sft] kernel COMPLETE — proceeding";;
  *) echo "[sft] status never readable after 10 attempts; submitting anyway —"
     echo "[sft] an unconfirmed-but-committed kernel is worth more than a forfeited slot,"
     echo "[sft] and the marker check below still verifies the submission landed.";;
esac

OK=0
for i in $(seq 1 40); do
  kaggle competitions submit arc-prize-2026-arc-agi-3 \
    -k ahmedmobasher86/arc-agi-3-duck-sft -v 4 \
    -f submission.parquet -m "$MSG" >/dev/null 2>&1
  sleep 30
  if kaggle competitions submissions arc-prize-2026-arc-agi-3 2>/dev/null | grep -q "$MARKER"; then
    echo "[sft] SUBMITTED_OK at $(date -u '+%F %H:%M:%S UTC') (attempt $i)"
    OK=1; break
  fi
  echo "[sft] attempt $i not confirmed ($(date -u '+%H:%M:%S UTC')) — retry in 5m"
  sleep 270
done
[ "$OK" = "1" ] || echo "[FAIL] duck-sft never confirmed in the 2026-08-01 slot"
echo "[sft] done at $(date -u '+%F %H:%M:%S UTC')"
