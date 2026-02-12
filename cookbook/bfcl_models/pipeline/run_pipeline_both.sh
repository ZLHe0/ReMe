#!/bin/bash
set -e

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$PIPELINE_DIR/.." && pwd)"
LOG_ROOT="$ROOT_DIR/logs"

mkdir -p "$LOG_ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
CLAUDE_LOG="$LOG_ROOT/claude_pipeline_$STAMP.log"
QWEN_LOG="$LOG_ROOT/qwen_pipeline_$STAMP.log"

echo "Logs:"
echo "  Claude: $CLAUDE_LOG"
echo "  Qwen:   $QWEN_LOG"

echo "=== Running Claude pipeline ==="
BACKEND=bedrock \
MODEL_ID="${MODEL_ID:-us.anthropic.claude-sonnet-4-20250514-v1:0}" \
AWS_REGION="${AWS_REGION:-us-west-2}" \
EMBEDDING_MODEL_NAME="${EMBEDDING_MODEL_NAME:-Qwen/Qwen3-Embedding-0.6B}" \
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://localhost:5002/v1}" \
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-1}" \
EMBEDDING_PORT="${EMBEDDING_PORT:-5002}" \
EMBEDDING_START="${EMBEDDING_START:-true}" \
"$PIPELINE_DIR/run_pipeline.sh" > "$CLAUDE_LOG" 2>&1

echo "=== Running Qwen pipeline ==="
BACKEND=openai \
MODEL_ID="${MODEL_ID_QWEN:-qwen3}" \
OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:5001/v1}" \
OPENAI_API_KEY="${OPENAI_API_KEY:-dummy-key}" \
EMBEDDING_MODEL_NAME="${EMBEDDING_MODEL_NAME:-Qwen/Qwen3-Embedding-0.6B}" \
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://localhost:5002/v1}" \
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-1}" \
EMBEDDING_PORT="${EMBEDDING_PORT:-5002}" \
EMBEDDING_START="${EMBEDDING_START:-true}" \
"$PIPELINE_DIR/run_pipeline.sh" > "$QWEN_LOG" 2>&1

echo "Done."
