#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EMBEDDING_CONDA_ENV="${EMBEDDING_CONDA_ENV:-qwen3-vllm}"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  echo "ERROR: Cannot find conda.sh. Do you need to run 'conda init'?"
  exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "$EMBEDDING_CONDA_ENV"

EMBEDDING_MODEL_NAME="${EMBEDDING_MODEL_NAME:-Qwen/Qwen3-Embedding-0.6B}"
EMBEDDING_PORT="${EMBEDDING_PORT:-5002}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-1}"
EMBEDDING_TP="${EMBEDDING_TP:-1}"
EMBEDDING_PID_FILE="${EMBEDDING_PID_FILE:-/tmp/qwen3_embedding_${EMBEDDING_PORT}.pid}"
EMBEDDING_LOG="${EMBEDDING_LOG:-/tmp/qwen3_embedding_${EMBEDDING_PORT}.log}"

echo "Starting Qwen3 Embedding vLLM..."
echo "  MODEL_NAME: $EMBEDDING_MODEL_NAME"
echo "  PORT:       $EMBEDDING_PORT"
echo "  DEVICE:     $EMBEDDING_DEVICE"
echo "  TP:         $EMBEDDING_TP"
echo "  LOG:        $EMBEDDING_LOG"

python "$SCRIPT_DIR/qwen3_embedding_server.py" \
  --model_path "$EMBEDDING_MODEL_NAME" \
  --port "$EMBEDDING_PORT" \
  --device "$EMBEDDING_DEVICE" \
  --tensor_parallel_size "$EMBEDDING_TP" \
  --pid_file "$EMBEDDING_PID_FILE" \
  --log_file "$EMBEDDING_LOG" \
  --initial_wait 120 \
  > /dev/null 2>&1 &

echo $! > "$EMBEDDING_PID_FILE"
echo "Started PID $(cat "$EMBEDDING_PID_FILE")"
