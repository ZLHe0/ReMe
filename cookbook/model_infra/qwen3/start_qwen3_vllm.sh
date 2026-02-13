#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

QWEN3_CONDA_ENV="${QWEN3_CONDA_ENV:-qwen3-vllm}"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  echo "ERROR: Cannot find conda.sh. Do you need to run 'conda init'?"
  exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "$QWEN3_CONDA_ENV"

QWEN3_MODEL_PATH="${QWEN3_MODEL_PATH:-Qwen/Qwen3-8B}"
QWEN3_PORT="${QWEN3_PORT:-5001}"
QWEN3_DEVICE="${QWEN3_DEVICE:-0,2,3,4,5,6,7}"
QWEN3_DP="${QWEN3_DP:-7}"
QWEN3_TP="${QWEN3_TP:-1}"
QWEN3_CONTEXT_LEN="${QWEN3_CONTEXT_LEN:-32768}"
QWEN3_PID_FILE="${QWEN3_PID_FILE:-/tmp/qwen3_vllm_${QWEN3_PORT}.pid}"
QWEN3_LOG="${QWEN3_LOG:-/tmp/qwen3_vllm_${QWEN3_PORT}.log}"

echo "Starting Qwen3 vLLM..."
echo "  MODEL_PATH: $QWEN3_MODEL_PATH"
echo "  PORT:       $QWEN3_PORT"
echo "  DEVICE:     $QWEN3_DEVICE"
echo "  DP/TP:      $QWEN3_DP / $QWEN3_TP"
echo "  LOG:        $QWEN3_LOG"

python "$SCRIPT_DIR/qwen3_vllm_server.py" \
  --model_path "$QWEN3_MODEL_PATH" \
  --port "$QWEN3_PORT" \
  --device "$QWEN3_DEVICE" \
  --data_parallel_size "$QWEN3_DP" \
  --tensor_parallel_size "$QWEN3_TP" \
  --context_length "$QWEN3_CONTEXT_LEN" \
  --pid_file "$QWEN3_PID_FILE" \
  --log_file "$QWEN3_LOG" \
  --initial_wait 120 \
  > /dev/null 2>&1 &

echo $! > "$QWEN3_PID_FILE"
echo "Started PID $(cat "$QWEN3_PID_FILE")"
