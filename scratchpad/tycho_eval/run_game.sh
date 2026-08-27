#!/bin/zsh
# Kill-or-scale rig: run ONE public game through Tycho against our Modal vLLM 27B.
# Usage: ./run_game.sh <game_id> [out_subdir]
set -e
GAME="$1"
OUT="${2:-results/$GAME}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/tycho"

export SSL_CERT_FILE="$($HERE/.venv/bin/python -m certifi)"
export LLM_BASE_URL="https://a-m-mobasher--arc3-vllm-serve.modal.run/v1"
export LLM_API_KEY="$(cat ~/.arc3_vllm_token | tr -d '[:space:]')"
export TYCHO_EVAL_QWEN_SAMPLING=1              # local patch: temp 0.6 / top_p 0.95 / top_k 20
export TYCHO_PER_GAME_WALLCLOCK_BUDGET_SECONDS=3600
export TYCHO_DIAGNOSTICS_DIR="$HERE/diagnostics"
mkdir -p "$TYCHO_DIAGNOSTICS_DIR"

exec "$HERE/.venv/bin/python" -m tycho.harness.run_parallel \
  --approach tycho \
  --games "$GAME" \
  --out-dir "$OUT" \
  --config "$HERE/qwen27b_orchestrator.yaml" \
  --operation-mode offline \
  --max-workers 1 \
  --viz
