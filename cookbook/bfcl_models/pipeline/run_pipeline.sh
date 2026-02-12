#!/bin/bash
#
# BFCL v3 pipeline (Claude + ReMe) using modular Python scripts.
#
# Steps:
# 1) Setup data
# 2) Start services (LiteLLM, ReMe)
# 3) Collect training trajectories
# 4) Build memory from successful trajectories
# 5) Run validation with memory
# 6) Run baseline without memory
# 7) Evaluate results
#

set -e

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$PIPELINE_DIR/.." && pwd)"
cd "$ROOT_DIR"

# ============================================================================
# Configuration
# ============================================================================
CONDA_ENV="bfcl"
CONDA_PATH="$HOME/miniconda"

MODEL_ID="us.anthropic.claude-sonnet-4-20250514-v1:0"
AWS_REGION="us-west-2"
BACKEND="${BACKEND:-bedrock}" # bedrock | openai
OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:5001/v1}"
OPENAI_API_KEY="${OPENAI_API_KEY:-dummy-key}"
EMBEDDING_MODEL_NAME="${EMBEDDING_MODEL_NAME:-Qwen/Qwen3-Embedding-0.6B}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://localhost:5002/v1}"
EMBEDDING_START="${EMBEDDING_START:-true}"
EMBEDDING_PORT="${EMBEDDING_PORT:-5002}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-1}"

DATA_DIR="$ROOT_DIR/data"
TRAIN_DATA="$DATA_DIR/multiturn_data_base_train.jsonl"
VAL_DATA="$DATA_DIR/multiturn_data_base_val.jsonl"
ANSWER_DIR="$DATA_DIR/possible_answer"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_DIR="$ROOT_DIR/exp_result/$TIMESTAMP"
LOG_DIR="$EXP_DIR/logs"

LITELLM_PORT=4000
REME_PORT=8002
REME_WORKSPACE="bfcl_v3_${BACKEND}_$TIMESTAMP"

MAX_WORKERS=10
NUM_TRIALS=1

# ============================================================================
# Helpers
# ============================================================================
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg"
    if [ -d "$LOG_DIR" ]; then
        echo "$msg" >> "$LOG_DIR/pipeline.log"
    fi
}

activate_conda() {
    source "$CONDA_PATH/bin/activate" "$CONDA_ENV"
}

wait_for_service() {
    local url=$1
    local name=$2
    local max_attempts=30
    local attempt=0

    log "Waiting for $name to be ready..."
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "$url" > /dev/null 2>&1; then
            log "$name is ready"
            return 0
        fi
        sleep 2
        attempt=$((attempt + 1))
    done
    echo "ERROR: $name failed to start after $max_attempts attempts"
    exit 1
}

cleanup_services() {
    log "Cleaning up services..."
    pkill -f "litellm.*--port $LITELLM_PORT" 2>/dev/null || true
    pkill -f "reme.*port.*$REME_PORT" 2>/dev/null || true
    pkill -f "vllm.*--port $EMBEDDING_PORT" 2>/dev/null || true
    sleep 2
}

# ============================================================================
# Pipeline Steps
# ============================================================================
setup() {
    mkdir -p "$EXP_DIR" "$LOG_DIR" "$DATA_DIR"
    activate_conda
    cleanup_services
}

setup_data() {
    log "Step 1: Setup BFCL data"
    activate_conda
    python setup_bfcl_data.py --data-dir "$DATA_DIR"
}

start_services() {
    log "Step 2: Start services"
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
        nohup bash "$ROOT_DIR/qwen3/start_qwen3_embedding.sh" \
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

run_training() {
    log "Step 3: Collect trajectories (training)"
    activate_conda
    python scripts/run_bfcl.py \
        --dataset-name bfcl-multi-turn-base \
        --experiment-suffix train \
        --max-workers "$MAX_WORKERS" \
        --num-trials "$NUM_TRIALS" \
        --model-id "$MODEL_ID" \
        --data-path "$TRAIN_DATA" \
        --answer-path "$ANSWER_DIR" \
        --backend "$BACKEND" \
        --base-url "$OPENAI_BASE_URL" \
        --api-key "$OPENAI_API_KEY" \
        --region-name "$AWS_REGION" \
        --output-dir "$EXP_DIR"
}

build_memory() {
    log "Step 4: Build memory"
    activate_conda
    python scripts/build_memory_from_results.py \
        --results-path "$EXP_DIR/bfcl-multi-turn-base_train.jsonl" \
        --reme-url "http://localhost:$REME_PORT" \
        --workspace-id "$REME_WORKSPACE"
}

run_validation_with_memory() {
    log "Step 5: Validation with memory"
    activate_conda
    python scripts/run_bfcl.py \
        --dataset-name bfcl-multi-turn-base \
        --experiment-suffix val_with_memory \
        --max-workers "$MAX_WORKERS" \
        --num-trials "$NUM_TRIALS" \
        --model-id "$MODEL_ID" \
        --data-path "$VAL_DATA" \
        --answer-path "$ANSWER_DIR" \
        --backend "$BACKEND" \
        --base-url "$OPENAI_BASE_URL" \
        --api-key "$OPENAI_API_KEY" \
        --use-memory \
        --memory-base-url "http://localhost:$REME_PORT/" \
        --memory-workspace-id "$REME_WORKSPACE" \
        --region-name "$AWS_REGION" \
        --output-dir "$EXP_DIR"
}

run_baseline() {
    log "Step 6: Validation baseline"
    activate_conda
    python scripts/run_bfcl.py \
        --dataset-name bfcl-multi-turn-base \
        --experiment-suffix val_baseline \
        --max-workers "$MAX_WORKERS" \
        --num-trials "$NUM_TRIALS" \
        --model-id "$MODEL_ID" \
        --data-path "$VAL_DATA" \
        --answer-path "$ANSWER_DIR" \
        --backend "$BACKEND" \
        --base-url "$OPENAI_BASE_URL" \
        --api-key "$OPENAI_API_KEY" \
        --region-name "$AWS_REGION" \
        --output-dir "$EXP_DIR"
}

evaluate() {
    log "Step 7: Evaluate results"
    activate_conda
    python scripts/evaluate_results.py \
        --train "$EXP_DIR/bfcl-multi-turn-base_train.jsonl" \
        --baseline "$EXP_DIR/bfcl-multi-turn-base_val_baseline.jsonl" \
        --memory "$EXP_DIR/bfcl-multi-turn-base_val_with_memory.jsonl" \
        --output "$EXP_DIR/summary.json"
}

cleanup() {
    cleanup_services
}

# ============================================================================
# Main
# ============================================================================
trap cleanup EXIT

setup
setup_data
start_services
run_training
build_memory
run_validation_with_memory
run_baseline
evaluate

log "Pipeline complete"
log "Results: $EXP_DIR"
