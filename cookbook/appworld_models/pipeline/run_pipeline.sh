#!/bin/bash
#
# AppWorld pipeline (Claude/Qwen via OpenAI-compatible backend) using modular scripts.
#
set -e

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$PIPELINE_DIR/.." && pwd)"
INFRA_DIR="$(cd "$ROOT_DIR/../model_infra" && pwd)"
cd "$ROOT_DIR"
# shellcheck disable=SC1090
source "$INFRA_DIR/pipeline/common.sh"

# ==========================================================================
# Configuration
# ==========================================================================
CONDA_ENV="${CONDA_ENV:-appworld}"
CONDA_PATH="${CONDA_PATH:-$HOME/miniconda}"

MODEL_ID="${MODEL_ID:-us.anthropic.claude-sonnet-4-20250514-v1:0}"
AWS_REGION="${AWS_REGION:-us-west-2}"
BACKEND="${BACKEND:-bedrock}" # bedrock | openai
OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:5001/v1}"
OPENAI_API_KEY="${OPENAI_API_KEY:-dummy-key}"

EMBEDDING_MODEL_NAME="${EMBEDDING_MODEL_NAME:-Qwen/Qwen3-Embedding-0.6B}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://localhost:5002/v1}"
EMBEDDING_START="${EMBEDDING_START:-true}"
EMBEDDING_PORT="${EMBEDDING_PORT:-5002}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-1}"

DATASET_NAME="${DATASET_NAME:-test_normal}"
EXPERIMENT_SUFFIX="${EXPERIMENT_SUFFIX:-with-memory}"
MAX_WORKERS="${MAX_WORKERS:-8}"
NUM_TRIALS="${NUM_TRIALS:-2}"
BATCH_SIZE="${BATCH_SIZE:-8}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_DIR="$ROOT_DIR/exp_result/$TIMESTAMP"
LOG_DIR="$EXP_DIR/logs"

LITELLM_PORT=${LITELLM_PORT:-4000}
REME_PORT=${REME_PORT:-8002}
REME_WORKSPACE="${REME_WORKSPACE:-appworld_${BACKEND}_${TIMESTAMP}}"

RESET_WORKSPACE="${RESET_WORKSPACE:-true}"
LOAD_MEMORY="${LOAD_MEMORY:-false}"
MEMORY_PATH="${MEMORY_PATH:-docs/library}"

USE_MEMORY="${USE_MEMORY:-true}"
USE_MEMORY_ADDITION="${USE_MEMORY_ADDITION:-true}"
USE_MEMORY_DELETION="${USE_MEMORY_DELETION:-true}"
DELETE_FREQ="${DELETE_FREQ:-5}"
FREQ_THRESHOLD="${FREQ_THRESHOLD:-5}"
UTILITY_THRESHOLD="${UTILITY_THRESHOLD:-0.5}"

# ==========================================================================
# Pipeline Steps
# ==========================================================================
setup() {
    mkdir -p "$EXP_DIR" "$LOG_DIR"
    activate_conda
    cleanup_services
}

start_services() {
    log "Step 1: Start services"
    activate_conda

    if [ "$BACKEND" = "bedrock" ]; then
        log "Starting LiteLLM proxy on port $LITELLM_PORT..."
        nohup litellm --model "bedrock/$MODEL_ID" --port "$LITELLM_PORT" \
            > "$LOG_DIR/litellm.log" 2>&1 &
        echo $! > "$LOG_DIR/litellm.pid"
        wait_for_service "http://localhost:$LITELLM_PORT/health" "LiteLLM"
    else
        log "Skipping LiteLLM (BACKEND=$BACKEND)"
    fi

    if [ "$EMBEDDING_START" = true ]; then
        log "Starting Qwen3 embedding server on port $EMBEDDING_PORT (GPU $EMBEDDING_DEVICE)..."
        nohup bash "$INFRA_DIR/qwen3/start_qwen3_embedding.sh" \
            > "$LOG_DIR/qwen3_embedding.log" 2>&1 &
        wait_for_service "http://localhost:$EMBEDDING_PORT/v1/models" "Qwen3 Embedding"
    else
        log "Skipping embedding server (EMBEDDING_START=$EMBEDDING_START)"
    fi

    log "Creating ReMe config..."
    if [ "$BACKEND" = "bedrock" ]; then
        LLM_BASE_URL="http://localhost:$LITELLM_PORT/v1"
        LLM_MODEL_NAME="bedrock/$MODEL_ID"
    else
        LLM_BASE_URL="$OPENAI_BASE_URL"
        LLM_MODEL_NAME="$MODEL_ID"
    fi

    cat > "$EXP_DIR/reme_config.yaml" << EOF
backend: http
http:
  host: "0.0.0.0"
  port: $REME_PORT
  timeout_keep_alive: 600
llm:
  default:
    backend: openai_compatible
    base_url: ${LLM_BASE_URL}
    model_name: ${LLM_MODEL_NAME}
    params:
      temperature: 0.6
embedding_model:
  default:
    backend: openai_compatible
    base_url: ${EMBEDDING_BASE_URL}
    model_name: ${EMBEDDING_MODEL_NAME}
vector_store:
  default:
    backend: local
    path: $EXP_DIR/vector_store
EOF

    log "Starting ReMe on port $REME_PORT..."
    nohup reme --config "$EXP_DIR/reme_config.yaml" \
        > "$LOG_DIR/reme.log" 2>&1 &
    echo $! > "$LOG_DIR/reme.pid"
    wait_for_service "http://localhost:$REME_PORT/health" "ReMe"
}

run_appworld() {
    log "Step 2: Run AppWorld"
    activate_conda

    if [ "$BACKEND" = "bedrock" ]; then
        APPWORLD_BASE_URL="http://localhost:$LITELLM_PORT/v1"
    else
        APPWORLD_BASE_URL="$OPENAI_BASE_URL"
    fi

    EXTRA_ARGS=()
    if [ "$USE_MEMORY" = true ]; then
        EXTRA_ARGS+=("--use-memory")
    fi
    if [ "$USE_MEMORY_ADDITION" = true ]; then
        EXTRA_ARGS+=("--use-memory-addition")
    fi
    if [ "$USE_MEMORY_DELETION" = true ]; then
        EXTRA_ARGS+=("--use-memory-deletion")
    fi
    if [ "$RESET_WORKSPACE" = true ]; then
        EXTRA_ARGS+=("--reset-workspace")
    fi
    if [ "$LOAD_MEMORY" = true ]; then
        EXTRA_ARGS+=("--load-memory")
    fi

    python scripts/run_appworld.py \
        --model-name "$MODEL_ID" \
        --base-url "$APPWORLD_BASE_URL" \
        --api-key "$OPENAI_API_KEY" \
        --dataset-name "$DATASET_NAME" \
        --experiment-suffix "$EXPERIMENT_SUFFIX" \
        --max-workers "$MAX_WORKERS" \
        --num-trials "$NUM_TRIALS" \
        --batch-size "$BATCH_SIZE" \
        --workspace-id "$REME_WORKSPACE" \
        --api-url "http://localhost:$REME_PORT/" \
        --delete-freq "$DELETE_FREQ" \
        --freq-threshold "$FREQ_THRESHOLD" \
        --utility-threshold "$UTILITY_THRESHOLD" \
        --memory-path "$MEMORY_PATH" \
        "${EXTRA_ARGS[@]}"
}

main() {
    setup
    start_services
    run_appworld
    log "Pipeline complete. Logs: $LOG_DIR"
}

main "$@"
