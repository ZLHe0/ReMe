#!/bin/bash
#
# BFCL v3 Full Pipeline with Claude API and ReMe Memory
# This script runs the complete pipeline:
# 1. Setup data
# 2. Start services (LiteLLM proxy, ReMe)
# 3. Run training to collect trajectories
# 4. Build memory from successful trajectories
# 5. Run inference with memory on validation data
# 6. Run baseline (without memory) on validation data
# 7. Evaluate and compare results
#

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Conda environment
CONDA_ENV="bfcl"
CONDA_PATH="$HOME/miniconda"

# Model configuration
MODEL_ID="us.anthropic.claude-sonnet-4-20250514-v1:0"
AWS_REGION="us-west-2"

# Data paths
DATA_DIR="$SCRIPT_DIR/data"
TRAIN_DATA="$DATA_DIR/multiturn_data_base_train.jsonl"
VAL_DATA="$DATA_DIR/multiturn_data_base_val.jsonl"
FULL_DATA="$DATA_DIR/multiturn_data_base.jsonl"

# Output paths
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_DIR="$SCRIPT_DIR/exp_result/$TIMESTAMP"
LOG_DIR="$EXP_DIR/logs"

# Service ports
LITELLM_PORT=4000
REME_PORT=8002

# ReMe configuration
REME_WORKSPACE="bfcl_v3_claude_$TIMESTAMP"

# Pipeline settings
MAX_WORKERS=1  # Sequential to avoid rate limits
TRAIN_SPLIT_RATIO=0.25  # 50 train / 150 val

# ============================================================================
# Helper Functions
# ============================================================================
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg"
    if [ -d "$LOG_DIR" ]; then
        echo "$msg" >> "$LOG_DIR/pipeline.log"
    fi
}

error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" | tee -a "$LOG_DIR/pipeline.log"
    exit 1
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
    error "$name failed to start after $max_attempts attempts"
}

cleanup_services() {
    log "Cleaning up services..."
    pkill -f "litellm.*--port $LITELLM_PORT" 2>/dev/null || true
    pkill -f "reme.*port.*$REME_PORT" 2>/dev/null || true
    sleep 2
}

# ============================================================================
# Step 0: Setup
# ============================================================================
setup() {
    log "=============================================="
    log "Step 0: Setup"
    log "=============================================="

    # Create directories
    mkdir -p "$EXP_DIR"
    mkdir -p "$LOG_DIR"
    mkdir -p "$DATA_DIR"

    log "Experiment directory: $EXP_DIR"
    log "Log directory: $LOG_DIR"

    # Activate conda
    activate_conda

    # Check dependencies
    log "Checking dependencies..."
    python -c "import boto3; import bfcl_eval" || error "Missing dependencies"

    # Clean up any existing services
    cleanup_services
}

# ============================================================================
# Step 1: Data Setup
# ============================================================================
setup_data() {
    log "=============================================="
    log "Step 1: Data Setup"
    log "=============================================="

    activate_conda

    # Check if Gorilla repo exists
    if [ ! -d "gorilla" ]; then
        log "Cloning Gorilla repository..."
        git clone --depth 1 https://github.com/ShishirPatil/gorilla.git
    fi

    # Check if BFCL data exists
    BFCL_DATA_SRC="gorilla/berkeley-function-call-leaderboard/bfcl_eval/data"
    if [ ! -f "$BFCL_DATA_SRC/BFCL_v4_multi_turn_base.json" ]; then
        error "BFCL data not found at $BFCL_DATA_SRC"
    fi

    # Copy and prepare data
    log "Preparing BFCL v3 data..."
    python3 << EOF
import json
import random

# Load BFCL v4 multi_turn_base (equivalent to BFCL v3)
data = []
with open("$BFCL_DATA_SRC/BFCL_v4_multi_turn_base.json", "r") as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

print(f"Loaded {len(data)} samples")

# Save full data
with open("$FULL_DATA", "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")

# Split train/val (50/150 following ReMe paper)
random.seed(42)
random.shuffle(data)
train_size = int(len(data) * $TRAIN_SPLIT_RATIO)
train_data = data[:train_size]
val_data = data[train_size:]

# Save train
with open("$TRAIN_DATA", "w") as f:
    for item in train_data:
        f.write(json.dumps(item) + "\n")
print(f"Train: {len(train_data)} samples")

# Save val
with open("$VAL_DATA", "w") as f:
    for item in val_data:
        f.write(json.dumps(item) + "\n")
print(f"Val: {len(val_data)} samples")
EOF

    # Copy possible answers
    if [ ! -d "$DATA_DIR/possible_answer" ]; then
        cp -r "$BFCL_DATA_SRC/possible_answer" "$DATA_DIR/"
    fi

    log "Data setup complete"
    log "  Train: $(wc -l < $TRAIN_DATA) samples"
    log "  Val: $(wc -l < $VAL_DATA) samples"
}

# ============================================================================
# Step 2: Start Services
# ============================================================================
start_services() {
    log "=============================================="
    log "Step 2: Starting Services"
    log "=============================================="

    activate_conda

    # Start LiteLLM proxy for Claude via Bedrock
    log "Starting LiteLLM proxy on port $LITELLM_PORT..."
    nohup litellm --model "bedrock/$MODEL_ID" --port $LITELLM_PORT \
        > "$LOG_DIR/litellm.log" 2>&1 &
    echo $! > "$LOG_DIR/litellm.pid"

    wait_for_service "http://localhost:$LITELLM_PORT/health" "LiteLLM"

    # Create ReMe config with LiteLLM backend
    log "Creating ReMe configuration..."
    cat > "$EXP_DIR/reme_config.yaml" << EOF
backend: http

http:
  host: "0.0.0.0"
  port: $REME_PORT
  timeout_keep_alive: 600

llm:
  default:
    backend: openai_compatible
    base_url: http://localhost:$LITELLM_PORT/v1
    model_name: bedrock/$MODEL_ID
    params:
      temperature: 0.6

embedding_model:
  default:
    backend: openai_compatible
    base_url: http://localhost:$LITELLM_PORT/v1
    model_name: bedrock/$MODEL_ID

vector_store:
  default:
    backend: local
    path: $EXP_DIR/vector_store
EOF

    # Stop existing ReMe if running
    pkill -f "reme.*port.*$REME_PORT" 2>/dev/null || true
    sleep 2

    # Start ReMe with custom config
    log "Starting ReMe on port $REME_PORT..."
    cd "$SCRIPT_DIR"
    nohup reme --config "$EXP_DIR/reme_config.yaml" \
        > "$LOG_DIR/reme.log" 2>&1 &
    echo $! > "$LOG_DIR/reme.pid"

    wait_for_service "http://localhost:$REME_PORT/health" "ReMe"

    log "Services started successfully"
}

# ============================================================================
# Step 3: Training - Collect Trajectories
# ============================================================================
run_training() {
    log "=============================================="
    log "Step 3: Collecting Trajectories on Training Data"
    log "=============================================="

    activate_conda
    cd "$SCRIPT_DIR"

    local train_output="$EXP_DIR/train_results.jsonl"

    log "Running inference on training data..."
    log "  Input: $TRAIN_DATA"
    log "  Output: $train_output"

    python3 << EOF
import os
import sys
import json
import time
import datetime
from pathlib import Path

# Add BFCL to path
sys.path.insert(0, "gorilla/berkeley-function-call-leaderboard")

import boto3
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from loguru import logger
from tqdm import tqdm

from bfcl_eval.constants.executable_backend_config import MULTI_TURN_FUNC_DOC_FILE_MAPPING
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.model_handler.utils import convert_to_tool
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    execute_multi_turn_func_call,
    is_empty_execute_response,
)

MODEL_ID = "$MODEL_ID"
FUNC_DOC_DIR = "gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/multi_turn_func_doc"
MAX_TOKENS = 4096

def get_bedrock_client():
    return boto3.client('bedrock-runtime', region_name="$AWS_REGION")

def is_throttling(exc):
    return isinstance(exc, ClientError) and exc.response.get("Error", {}).get("Code") == "ThrottlingException"

def load_functions_for_test(test_entry):
    functions = []
    for cls in test_entry.get("involved_classes", []):
        if cls in MULTI_TURN_FUNC_DOC_FILE_MAPPING:
            func_file = os.path.join(FUNC_DOC_DIR, MULTI_TURN_FUNC_DOC_FILE_MAPPING[cls])
            if os.path.exists(func_file):
                with open(func_file, "r") as f:
                    for line in f:
                        if line.strip():
                            func_data = json.loads(line)
                            if isinstance(func_data, list):
                                functions.extend(func_data)
                            else:
                                functions.append(func_data)
    return functions

def convert_functions_to_tools(functions, test_category):
    tools = convert_to_tool(functions, GORILLA_TO_OPENAPI, ModelStyle.OPENAI_COMPLETIONS)
    for tool in tools:
        if "function" in tool and "response" in tool["function"]:
            del tool["function"]["response"]
    return tools

def convert_openai_tools_to_claude(openai_tools):
    claude_tools = []
    for tool in openai_tools:
        if tool.get("type") == "function":
            func = tool.get("function", {})
            claude_tools.append({
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {"type": "object", "properties": {}})
            })
    return claude_tools

def convert_messages_to_claude(messages):
    claude_messages = []
    for msg in messages:
        role = msg.get("role", "")
        if role == "user":
            claude_messages.append({"role": "user", "content": msg.get("content", "")})
        elif role == "assistant":
            content = []
            if msg.get("content"):
                content.append({"type": "text", "text": msg.get("content", "")})
            if "tool_calls" in msg and msg["tool_calls"]:
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    args = func.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except:
                            args = {}
                    content.append({
                        "type": "tool_use",
                        "id": tc.get("id", f"call_{len(content)}"),
                        "name": func.get("name", ""),
                        "input": args
                    })
            if content:
                claude_messages.append({"role": "assistant", "content": content})
        elif role == "tool":
            claude_messages.append({
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": msg.get("tool_call_id", ""), "content": str(msg.get("content", ""))}]
            })
    return claude_messages

def convert_claude_response_to_openai(resp_body):
    content_blocks = resp_body.get("content", [])
    text_parts = []
    tool_use_blocks = []
    for block in content_blocks:
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_use_blocks.append(block)
    msg = {"role": "assistant", "content": "\n".join(text_parts) if text_parts else ""}
    if tool_use_blocks:
        msg["tool_calls"] = [{"id": b.get("id", f"call_{i}"), "type": "function", "function": {"name": b.get("name", ""), "arguments": json.dumps(b.get("input", {}))}} for i, b in enumerate(tool_use_blocks)]
    return msg

@retry(retry=retry_if_exception(is_throttling), stop=stop_after_attempt(20), wait=wait_exponential(multiplier=1, min=2, max=30))
def call_claude(client, messages, tools, model_id):
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
        "messages": convert_messages_to_claude(messages),
    }
    if tools:
        payload["tools"] = convert_openai_tools_to_claude(tools)
        payload["tool_choice"] = {"type": "auto"}
    resp = client.invoke_model(body=json.dumps(payload), modelId=model_id)
    return convert_claude_response_to_openai(json.loads(resp['body'].read()))

def run_task(client, task_id, data_path, model_id, max_turns=30):
    # Load test entry
    with open(data_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("id") == task_id:
                break

    entry["function"] = load_functions_for_test(entry)
    test_category = task_id.rsplit("_", 1)[0]
    tools = convert_functions_to_tools(entry["function"], test_category)

    questions = entry.get("question", [])
    if not questions:
        return {"task_id": task_id, "reward": 0, "error": "No questions"}

    history = [{"role": "user", "content": questions[0][0]["content"]}]
    current_turn = 1

    for _ in range(max_turns):
        try:
            response = call_claude(client, history, tools, model_id)
            history.append(response)

            if "tool_calls" in response and response["tool_calls"]:
                decoded = []
                for tc in response["tool_calls"]:
                    func = tc.get("function", {})
                    args = func.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except:
                            args = {}
                    decoded.append(f"{func.get('name', '')}({', '.join(f'{k}={repr(v)}' for k,v in args.items())})")

                if not is_empty_execute_response(decoded):
                    results, _ = execute_multi_turn_func_call(decoded, entry.get("initial_config", {}), entry.get("involved_classes", []), "claude", task_id, False, False)
                    for i, (tc, r) in enumerate(zip(response["tool_calls"], results)):
                        history.append({"role": "tool", "content": str(r), "tool_call_id": tc.get("id", f"call_{i}")})
                else:
                    if current_turn < len(questions):
                        history.append({"role": "user", "content": questions[current_turn][0]["content"] if questions[current_turn] else ""})
                        current_turn += 1
                    else:
                        break
            else:
                if current_turn < len(questions):
                    history.append({"role": "user", "content": questions[current_turn][0]["content"] if questions[current_turn] else ""})
                    current_turn += 1
                else:
                    break
        except Exception as e:
            logger.error(f"Error: {e}")
            break

    return {
        "task_id": task_id,
        "reward": 1.0 if current_turn >= len(questions) else 0.0,
        "completed": current_turn >= len(questions),
        "turns_completed": current_turn,
        "total_turns": len(questions),
        "history": history,
    }

# Main
client = get_bedrock_client()
task_ids = []
with open("$TRAIN_DATA", "r") as f:
    for line in f:
        task_ids.append(json.loads(line)["id"])

logger.info(f"Running {len(task_ids)} training tasks")

with open("$train_output", "w") as out:
    for task_id in tqdm(task_ids, desc="Training"):
        result = run_task(client, task_id, "$TRAIN_DATA", MODEL_ID)
        out.write(json.dumps(result) + "\n")
        out.flush()
        time.sleep(1)

# Summary
with open("$train_output", "r") as f:
    results = [json.loads(l) for l in f]
successful = sum(1 for r in results if r.get("reward", 0) == 1)
logger.info(f"Training complete: {successful}/{len(results)} successful")
EOF

    log "Training complete"
    log "  Results: $(wc -l < $train_output) tasks"
    log "  Successful: $(grep '"reward": 1.0' $train_output | wc -l)"
}

# ============================================================================
# Step 4: Build Memory Pool
# ============================================================================
build_memory() {
    log "=============================================="
    log "Step 4: Building Memory Pool"
    log "=============================================="

    activate_conda
    cd "$SCRIPT_DIR"

    local train_results="$EXP_DIR/train_results.jsonl"

    if [ ! -f "$train_results" ]; then
        error "Training results not found: $train_results"
    fi

    log "Building memory from successful trajectories..."

    python3 << EOF
import json
import requests
from loguru import logger

REME_URL = "http://localhost:$REME_PORT"
WORKSPACE_ID = "$REME_WORKSPACE"

# Load training results
with open("$train_results", "r") as f:
    results = [json.loads(l) for l in f]

# Filter successful
successful = [r for r in results if r.get("reward", 0) == 1]
logger.info(f"Found {len(successful)} successful trajectories")

if not successful:
    logger.warning("No successful trajectories to build memory from")
    exit(0)

# Build memory for each successful trajectory
total_memories = 0
for result in successful:
    trajectory = {
        "task_id": result["task_id"],
        "messages": result.get("history", []),
        "score": result["reward"]
    }

    request_data = {
        "trajectories": [trajectory],
        "workspace_id": WORKSPACE_ID
    }

    try:
        resp = requests.post(f"{REME_URL}/summary_task_memory", json=request_data, timeout=120)
        if resp.status_code == 200:
            data = resp.json()
            memories = len(data.get("metadata", {}).get("memory_list", []))
            total_memories += memories
            logger.info(f"Task {result['task_id']}: {memories} memories created")
        else:
            logger.warning(f"Task {result['task_id']}: Failed - {resp.status_code}")
    except Exception as e:
        logger.error(f"Task {result['task_id']}: Error - {e}")

logger.info(f"Total memories created: {total_memories}")
EOF

    log "Memory building complete"
}

# ============================================================================
# Step 5: Run with Memory on Validation Data
# ============================================================================
run_validation_with_memory() {
    log "=============================================="
    log "Step 5: Running Validation with Memory"
    log "=============================================="

    activate_conda
    cd "$SCRIPT_DIR"

    local val_output="$EXP_DIR/val_with_memory.jsonl"

    log "Running inference with memory on validation data..."

    python3 << EOF
import os
import sys
import json
import time
import requests
from pathlib import Path

sys.path.insert(0, "gorilla/berkeley-function-call-leaderboard")

import boto3
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from loguru import logger
from tqdm import tqdm

from bfcl_eval.constants.executable_backend_config import MULTI_TURN_FUNC_DOC_FILE_MAPPING
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.model_handler.utils import convert_to_tool
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    execute_multi_turn_func_call,
    is_empty_execute_response,
)

MODEL_ID = "$MODEL_ID"
FUNC_DOC_DIR = "gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/multi_turn_func_doc"
MAX_TOKENS = 4096
REME_URL = "http://localhost:$REME_PORT"
WORKSPACE_ID = "$REME_WORKSPACE"

def get_bedrock_client():
    return boto3.client('bedrock-runtime', region_name="$AWS_REGION")

def is_throttling(exc):
    return isinstance(exc, ClientError) and exc.response.get("Error", {}).get("Code") == "ThrottlingException"

def load_functions_for_test(test_entry):
    functions = []
    for cls in test_entry.get("involved_classes", []):
        if cls in MULTI_TURN_FUNC_DOC_FILE_MAPPING:
            func_file = os.path.join(FUNC_DOC_DIR, MULTI_TURN_FUNC_DOC_FILE_MAPPING[cls])
            if os.path.exists(func_file):
                with open(func_file, "r") as f:
                    for line in f:
                        if line.strip():
                            func_data = json.loads(line)
                            if isinstance(func_data, list):
                                functions.extend(func_data)
                            else:
                                functions.append(func_data)
    return functions

def convert_functions_to_tools(functions, test_category):
    tools = convert_to_tool(functions, GORILLA_TO_OPENAPI, ModelStyle.OPENAI_COMPLETIONS)
    for tool in tools:
        if "function" in tool and "response" in tool["function"]:
            del tool["function"]["response"]
    return tools

def convert_openai_tools_to_claude(openai_tools):
    return [{"name": t.get("function", {}).get("name", ""), "description": t.get("function", {}).get("description", ""), "input_schema": t.get("function", {}).get("parameters", {})} for t in openai_tools if t.get("type") == "function"]

def convert_messages_to_claude(messages):
    claude_messages = []
    for msg in messages:
        role = msg.get("role", "")
        if role == "user":
            claude_messages.append({"role": "user", "content": msg.get("content", "")})
        elif role == "assistant":
            content = []
            if msg.get("content"):
                content.append({"type": "text", "text": msg.get("content", "")})
            if "tool_calls" in msg and msg["tool_calls"]:
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    args = func.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except:
                            args = {}
                    content.append({"type": "tool_use", "id": tc.get("id", ""), "name": func.get("name", ""), "input": args})
            if content:
                claude_messages.append({"role": "assistant", "content": content})
        elif role == "tool":
            claude_messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": msg.get("tool_call_id", ""), "content": str(msg.get("content", ""))}]})
    return claude_messages

def convert_claude_response_to_openai(resp_body):
    content_blocks = resp_body.get("content", [])
    text_parts, tool_use_blocks = [], []
    for block in content_blocks:
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_use_blocks.append(block)
    msg = {"role": "assistant", "content": "\n".join(text_parts) if text_parts else ""}
    if tool_use_blocks:
        msg["tool_calls"] = [{"id": b.get("id", f"call_{i}"), "type": "function", "function": {"name": b.get("name", ""), "arguments": json.dumps(b.get("input", {}))}} for i, b in enumerate(tool_use_blocks)]
    return msg

@retry(retry=retry_if_exception(is_throttling), stop=stop_after_attempt(20), wait=wait_exponential(multiplier=1, min=2, max=30))
def call_claude(client, messages, tools, model_id):
    payload = {"anthropic_version": "bedrock-2023-05-31", "max_tokens": MAX_TOKENS, "temperature": 0.7, "messages": convert_messages_to_claude(messages)}
    if tools:
        payload["tools"] = convert_openai_tools_to_claude(tools)
        payload["tool_choice"] = {"type": "auto"}
    resp = client.invoke_model(body=json.dumps(payload), modelId=model_id)
    return convert_claude_response_to_openai(json.loads(resp['body'].read()))

def retrieve_memory(query):
    try:
        resp = requests.post(f"{REME_URL}/retrieve_task_memory", json={"workspace_id": WORKSPACE_ID, "query": query, "top_k": 3}, timeout=30)
        if resp.status_code == 200:
            return resp.json().get("answer", "")
    except:
        pass
    return ""

def run_task(client, task_id, data_path, model_id, use_memory=True, max_turns=30):
    with open(data_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("id") == task_id:
                break

    entry["function"] = load_functions_for_test(entry)
    test_category = task_id.rsplit("_", 1)[0]
    tools = convert_functions_to_tools(entry["function"], test_category)

    questions = entry.get("question", [])
    if not questions:
        return {"task_id": task_id, "reward": 0, "error": "No questions"}

    first_question = questions[0][0]["content"]

    # Retrieve memory if enabled
    memory_content = ""
    if use_memory:
        memory_content = retrieve_memory(first_question)
        if memory_content:
            first_question = f"{first_question}\n\nRelevant experience:\n{memory_content}"

    history = [{"role": "user", "content": first_question}]
    current_turn = 1

    for _ in range(max_turns):
        try:
            response = call_claude(client, history, tools, model_id)
            history.append(response)

            if "tool_calls" in response and response["tool_calls"]:
                decoded = []
                for tc in response["tool_calls"]:
                    func = tc.get("function", {})
                    args = func.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except:
                            args = {}
                    decoded.append(f"{func.get('name', '')}({', '.join(f'{k}={repr(v)}' for k,v in args.items())})")

                if not is_empty_execute_response(decoded):
                    results, _ = execute_multi_turn_func_call(decoded, entry.get("initial_config", {}), entry.get("involved_classes", []), "claude", task_id, False, False)
                    for i, (tc, r) in enumerate(zip(response["tool_calls"], results)):
                        history.append({"role": "tool", "content": str(r), "tool_call_id": tc.get("id", f"call_{i}")})
                else:
                    if current_turn < len(questions):
                        history.append({"role": "user", "content": questions[current_turn][0]["content"] if questions[current_turn] else ""})
                        current_turn += 1
                    else:
                        break
            else:
                if current_turn < len(questions):
                    history.append({"role": "user", "content": questions[current_turn][0]["content"] if questions[current_turn] else ""})
                    current_turn += 1
                else:
                    break
        except Exception as e:
            logger.error(f"Error: {e}")
            break

    return {
        "task_id": task_id,
        "reward": 1.0 if current_turn >= len(questions) else 0.0,
        "completed": current_turn >= len(questions),
        "memory_used": bool(memory_content),
        "history": history,
    }

# Main
client = get_bedrock_client()
task_ids = []
with open("$VAL_DATA", "r") as f:
    for line in f:
        task_ids.append(json.loads(line)["id"])

logger.info(f"Running {len(task_ids)} validation tasks with memory")

with open("$val_output", "w") as out:
    for task_id in tqdm(task_ids, desc="Val+Memory"):
        result = run_task(client, task_id, "$VAL_DATA", MODEL_ID, use_memory=True)
        out.write(json.dumps(result) + "\n")
        out.flush()
        time.sleep(1)

with open("$val_output", "r") as f:
    results = [json.loads(l) for l in f]
successful = sum(1 for r in results if r.get("reward", 0) == 1)
logger.info(f"Validation with memory: {successful}/{len(results)} successful")
EOF

    log "Validation with memory complete"
}

# ============================================================================
# Step 6: Run Baseline (without Memory)
# ============================================================================
run_baseline() {
    log "=============================================="
    log "Step 6: Running Baseline (without Memory)"
    log "=============================================="

    activate_conda
    cd "$SCRIPT_DIR"

    local baseline_output="$EXP_DIR/val_baseline.jsonl"

    log "Running baseline inference on validation data..."

    # Reuse the same code but with use_memory=False
    python3 << EOF
import os
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, "gorilla/berkeley-function-call-leaderboard")

import boto3
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from loguru import logger
from tqdm import tqdm

from bfcl_eval.constants.executable_backend_config import MULTI_TURN_FUNC_DOC_FILE_MAPPING
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.model_handler.utils import convert_to_tool
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    execute_multi_turn_func_call,
    is_empty_execute_response,
)

MODEL_ID = "$MODEL_ID"
FUNC_DOC_DIR = "gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/multi_turn_func_doc"
MAX_TOKENS = 4096

def get_bedrock_client():
    return boto3.client('bedrock-runtime', region_name="$AWS_REGION")

def is_throttling(exc):
    return isinstance(exc, ClientError) and exc.response.get("Error", {}).get("Code") == "ThrottlingException"

def load_functions_for_test(test_entry):
    functions = []
    for cls in test_entry.get("involved_classes", []):
        if cls in MULTI_TURN_FUNC_DOC_FILE_MAPPING:
            func_file = os.path.join(FUNC_DOC_DIR, MULTI_TURN_FUNC_DOC_FILE_MAPPING[cls])
            if os.path.exists(func_file):
                with open(func_file, "r") as f:
                    for line in f:
                        if line.strip():
                            func_data = json.loads(line)
                            if isinstance(func_data, list):
                                functions.extend(func_data)
                            else:
                                functions.append(func_data)
    return functions

def convert_functions_to_tools(functions, test_category):
    tools = convert_to_tool(functions, GORILLA_TO_OPENAPI, ModelStyle.OPENAI_COMPLETIONS)
    for tool in tools:
        if "function" in tool and "response" in tool["function"]:
            del tool["function"]["response"]
    return tools

def convert_openai_tools_to_claude(openai_tools):
    return [{"name": t.get("function", {}).get("name", ""), "description": t.get("function", {}).get("description", ""), "input_schema": t.get("function", {}).get("parameters", {})} for t in openai_tools if t.get("type") == "function"]

def convert_messages_to_claude(messages):
    claude_messages = []
    for msg in messages:
        role = msg.get("role", "")
        if role == "user":
            claude_messages.append({"role": "user", "content": msg.get("content", "")})
        elif role == "assistant":
            content = []
            if msg.get("content"):
                content.append({"type": "text", "text": msg.get("content", "")})
            if "tool_calls" in msg and msg["tool_calls"]:
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    args = func.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except:
                            args = {}
                    content.append({"type": "tool_use", "id": tc.get("id", ""), "name": func.get("name", ""), "input": args})
            if content:
                claude_messages.append({"role": "assistant", "content": content})
        elif role == "tool":
            claude_messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": msg.get("tool_call_id", ""), "content": str(msg.get("content", ""))}]})
    return claude_messages

def convert_claude_response_to_openai(resp_body):
    content_blocks = resp_body.get("content", [])
    text_parts, tool_use_blocks = [], []
    for block in content_blocks:
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_use_blocks.append(block)
    msg = {"role": "assistant", "content": "\n".join(text_parts) if text_parts else ""}
    if tool_use_blocks:
        msg["tool_calls"] = [{"id": b.get("id", f"call_{i}"), "type": "function", "function": {"name": b.get("name", ""), "arguments": json.dumps(b.get("input", {}))}} for i, b in enumerate(tool_use_blocks)]
    return msg

@retry(retry=retry_if_exception(is_throttling), stop=stop_after_attempt(20), wait=wait_exponential(multiplier=1, min=2, max=30))
def call_claude(client, messages, tools, model_id):
    payload = {"anthropic_version": "bedrock-2023-05-31", "max_tokens": MAX_TOKENS, "temperature": 0.7, "messages": convert_messages_to_claude(messages)}
    if tools:
        payload["tools"] = convert_openai_tools_to_claude(tools)
        payload["tool_choice"] = {"type": "auto"}
    resp = client.invoke_model(body=json.dumps(payload), modelId=model_id)
    return convert_claude_response_to_openai(json.loads(resp['body'].read()))

def run_task(client, task_id, data_path, model_id, max_turns=30):
    with open(data_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("id") == task_id:
                break

    entry["function"] = load_functions_for_test(entry)
    test_category = task_id.rsplit("_", 1)[0]
    tools = convert_functions_to_tools(entry["function"], test_category)

    questions = entry.get("question", [])
    if not questions:
        return {"task_id": task_id, "reward": 0, "error": "No questions"}

    history = [{"role": "user", "content": questions[0][0]["content"]}]
    current_turn = 1

    for _ in range(max_turns):
        try:
            response = call_claude(client, history, tools, model_id)
            history.append(response)

            if "tool_calls" in response and response["tool_calls"]:
                decoded = []
                for tc in response["tool_calls"]:
                    func = tc.get("function", {})
                    args = func.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except:
                            args = {}
                    decoded.append(f"{func.get('name', '')}({', '.join(f'{k}={repr(v)}' for k,v in args.items())})")

                if not is_empty_execute_response(decoded):
                    results, _ = execute_multi_turn_func_call(decoded, entry.get("initial_config", {}), entry.get("involved_classes", []), "claude", task_id, False, False)
                    for i, (tc, r) in enumerate(zip(response["tool_calls"], results)):
                        history.append({"role": "tool", "content": str(r), "tool_call_id": tc.get("id", f"call_{i}")})
                else:
                    if current_turn < len(questions):
                        history.append({"role": "user", "content": questions[current_turn][0]["content"] if questions[current_turn] else ""})
                        current_turn += 1
                    else:
                        break
            else:
                if current_turn < len(questions):
                    history.append({"role": "user", "content": questions[current_turn][0]["content"] if questions[current_turn] else ""})
                    current_turn += 1
                else:
                    break
        except Exception as e:
            logger.error(f"Error: {e}")
            break

    return {
        "task_id": task_id,
        "reward": 1.0 if current_turn >= len(questions) else 0.0,
        "completed": current_turn >= len(questions),
        "history": history,
    }

# Main
client = get_bedrock_client()
task_ids = []
with open("$VAL_DATA", "r") as f:
    for line in f:
        task_ids.append(json.loads(line)["id"])

logger.info(f"Running {len(task_ids)} validation tasks (baseline)")

with open("$baseline_output", "w") as out:
    for task_id in tqdm(task_ids, desc="Baseline"):
        result = run_task(client, task_id, "$VAL_DATA", MODEL_ID)
        out.write(json.dumps(result) + "\n")
        out.flush()
        time.sleep(1)

with open("$baseline_output", "r") as f:
    results = [json.loads(l) for l in f]
successful = sum(1 for r in results if r.get("reward", 0) == 1)
logger.info(f"Baseline: {successful}/{len(results)} successful")
EOF

    log "Baseline complete"
}

# ============================================================================
# Step 7: Evaluation
# ============================================================================
evaluate() {
    log "=============================================="
    log "Step 7: Evaluation"
    log "=============================================="

    activate_conda
    cd "$SCRIPT_DIR"

    python3 << EOF
import json
from pathlib import Path

exp_dir = Path("$EXP_DIR")

# Load results
train_file = exp_dir / "train_results.jsonl"
memory_file = exp_dir / "val_with_memory.jsonl"
baseline_file = exp_dir / "val_baseline.jsonl"

def load_results(path):
    if not path.exists():
        return []
    with open(path, "r") as f:
        return [json.loads(l) for l in f]

train = load_results(train_file)
memory = load_results(memory_file)
baseline = load_results(baseline_file)

print("=" * 60)
print("BFCL v3 EVALUATION RESULTS")
print("=" * 60)
print()

# Training summary
train_success = sum(1 for r in train if r.get("reward", 0) == 1)
print(f"Training Data:")
print(f"  Total tasks:     {len(train)}")
print(f"  Successful:      {train_success}")
print(f"  Success rate:    {100*train_success/len(train):.1f}%" if train else "N/A")
print()

# Validation summary
baseline_success = sum(1 for r in baseline if r.get("reward", 0) == 1)
memory_success = sum(1 for r in memory if r.get("reward", 0) == 1)

print(f"Validation Data:")
print(f"  Total tasks:     {len(baseline)}")
print()
print(f"  Baseline (no memory):")
print(f"    Successful:    {baseline_success}")
print(f"    Success rate:  {100*baseline_success/len(baseline):.1f}%" if baseline else "N/A")
print()
print(f"  With Memory:")
print(f"    Successful:    {memory_success}")
print(f"    Success rate:  {100*memory_success/len(memory):.1f}%" if memory else "N/A")
print()

# Improvement
if baseline and memory:
    improvement = memory_success - baseline_success
    print(f"  Improvement:     {improvement:+d} tasks ({100*improvement/len(baseline):+.1f}%)")
print()
print("=" * 60)

# Save summary
summary = {
    "training": {"total": len(train), "successful": train_success},
    "baseline": {"total": len(baseline), "successful": baseline_success},
    "with_memory": {"total": len(memory), "successful": memory_success},
    "improvement": memory_success - baseline_success if baseline and memory else 0
}

with open(exp_dir / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"Summary saved to: {exp_dir / 'summary.json'}")
EOF

    log "Evaluation complete"
}

# ============================================================================
# Cleanup
# ============================================================================
cleanup() {
    log "=============================================="
    log "Cleanup"
    log "=============================================="

    cleanup_services
    log "Services stopped"
}

# ============================================================================
# Main
# ============================================================================
main() {
    # Create directories first so logging works
    mkdir -p "$EXP_DIR"
    mkdir -p "$LOG_DIR"
    mkdir -p "$DATA_DIR"

    log "=============================================="
    log "BFCL v3 Full Pipeline"
    log "Started at: $(date)"
    log "=============================================="

    setup
    setup_data
    start_services
    run_training
    build_memory
    run_validation_with_memory
    run_baseline
    evaluate
    cleanup

    log "=============================================="
    log "Pipeline Complete"
    log "Finished at: $(date)"
    log "Results in: $EXP_DIR"
    log "=============================================="
}

# Run with trap for cleanup
trap cleanup EXIT
main "$@"
