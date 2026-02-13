#!/bin/bash
set -e

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg"
    if [ -n "${LOG_DIR:-}" ] && [ -d "$LOG_DIR" ]; then
        echo "$msg" >> "$LOG_DIR/pipeline.log"
    fi
}

activate_conda() {
    if [ -n "${CONDA_PATH:-}" ] && [ -n "${CONDA_ENV:-}" ]; then
        # shellcheck disable=SC1090
        source "$CONDA_PATH/bin/activate" "$CONDA_ENV"
    fi
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
    if [ -n "${LITELLM_PORT:-}" ]; then
        pkill -f "litellm.*--port $LITELLM_PORT" 2>/dev/null || true
    fi
    if [ -n "${REME_PORT:-}" ]; then
        pkill -f "reme.*port.*$REME_PORT" 2>/dev/null || true
    fi
    if [ -n "${EMBEDDING_PORT:-}" ]; then
        pkill -f "vllm.*--port $EMBEDDING_PORT" 2>/dev/null || true
    fi
    sleep 2
}
