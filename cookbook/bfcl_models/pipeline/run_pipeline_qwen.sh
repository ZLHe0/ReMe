#!/bin/bash
set -e

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$PIPELINE_DIR/.." && pwd)"

export BACKEND=openai
export MODEL_ID="${MODEL_ID:-qwen3}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:5001/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy-key}"

# Start Qwen3 LLM server if not already running
if ! curl -s "http://localhost:5001/v1/models" > /dev/null 2>&1; then
  echo "Starting Qwen3 vLLM server..."
  "${ROOT_DIR}/qwen3/start_qwen3_vllm.sh" > /dev/null 2>&1 &
  sleep 5
fi

exec "$PIPELINE_DIR/run_pipeline.sh"
