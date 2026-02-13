#!/bin/bash
set -e

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$PIPELINE_DIR/.." && pwd)"
INFRA_DIR="$(cd "$ROOT_DIR/../model_infra" && pwd)"
# shellcheck disable=SC1090
source "$INFRA_DIR/pipeline/common.sh"

export BACKEND=openai
export MODEL_ID="${MODEL_ID:-qwen3}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:5001/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy-key}"

# Start Qwen3 LLM server if not already running
if ! curl -s "http://localhost:5001/v1/models" > /dev/null 2>&1; then
  log "Starting Qwen3 vLLM server..."
  "${INFRA_DIR}/qwen3/start_qwen3_vllm.sh" > /dev/null 2>&1 &
  wait_for_service "http://localhost:5001/v1/models" "Qwen3 vLLM"
fi

exec "$PIPELINE_DIR/run_pipeline.sh"
