#!/bin/zsh
# PART-1 falsifier: ONE game through Tycho SINGLE mode (actor writes world_model.py itself)
# against our Modal vLLM 27B. Usage: ./run_single.sh <game_id> [out_subdir]
set -e
GAME="$1"
OUT="${2:-results/${GAME}_single}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/tycho"

export SSL_CERT_FILE="$($HERE/.venv/bin/python -m certifi)"
export LLM_BASE_URL="https://a-m-mobasher--arc3-vllm-serve.modal.run/v1"
export LLM_API_KEY="$(cat ~/.arc3_vllm_token | tr -d '[:space:]')"
export TYCHO_EVAL_QWEN_SAMPLING=1              # local patch: temp 0.6 / top_p 0.95 / top_k 20
# 300 calls at the measured 15.1 s/call (sb26_v2) ~= 75 min; give 2h so the call cap,
# not the clock, is the binding limit of this falsifier.
export TYCHO_PER_GAME_WALLCLOCK_BUDGET_SECONDS=7200
export TYCHO_DIAGNOSTICS_DIR="$HERE/diagnostics"
mkdir -p "$TYCHO_DIAGNOSTICS_DIR"

exec "$HERE/.venv/bin/python" -m tycho.harness.run_parallel \
  --approach tycho \
  --games "$GAME" \
  --out-dir "$OUT" \
  --config "$HERE/qwen27b_single.yaml" \
  --operation-mode offline \
  --max-workers 1 \
  --viz
