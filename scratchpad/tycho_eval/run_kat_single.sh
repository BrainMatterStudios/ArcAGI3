#!/bin/zsh
# Track-1 FINAL probe: KAT-Coder-V2.5-Dev (W4A16) through Tycho SINGLE mode.
# Usage: ./run_kat_single.sh <game_id> [out_subdir]
# Deltas vs run_single.sh: KAT base URL, KAT sampling env (card opinion
# 1.0/0.95/20 instead of qwen 0.6/0.95/20), 75-min wall cap per the
# pre-registration ("Cap ~75 min wall"; 300-call cap is the primary limit).
set -e
GAME="$1"
OUT="${2:-results/${GAME}_kat_single}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/tycho"

export SSL_CERT_FILE="$($HERE/.venv/bin/python -m certifi)"
export LLM_BASE_URL="https://a-m-mobasher--arc3-kat-serve-serve.modal.run/v1"
export LLM_API_KEY="$(cat ~/.arc3_vllm_token | tr -d '[:space:]')"
export TYCHO_EVAL_KAT_SAMPLING=1               # card opinion: temp 1.0 / top_p 0.95 / top_k 20
export TYCHO_PER_GAME_WALLCLOCK_BUDGET_SECONDS=4500   # 75 min hard wall cap
export TYCHO_DIAGNOSTICS_DIR="$HERE/diagnostics"
mkdir -p "$TYCHO_DIAGNOSTICS_DIR"

exec "$HERE/.venv/bin/python" -m tycho.harness.run_parallel \
  --approach tycho \
  --games "$GAME" \
  --out-dir "$OUT" \
  --config "$HERE/kat_single.yaml" \
  --operation-mode offline \
  --max-workers 1 \
  --viz
