# BFCL v3 with Pluggable Backends (Bedrock + OpenAI-Compatible)

This directory contains the implementation of BFCL v3 (Berkeley Function Call Leaderboard) benchmark with pluggable model backends.

## Overview

BFCL v3 focuses on multi-turn function calling scenarios. This implementation supports:
- **Bedrock Claude** via AWS Bedrock
- **OpenAI-compatible** endpoints (e.g., vLLM for Qwen3)

### BFCL v3 vs v4

- **BFCL v3**: Uses `multi_turn_base` category (200 test cases)
- **BFCL v4**: Adds `multi_turn_miss_param` and `multi_turn_miss_func` categories

This implementation targets BFCL v3 (`multi_turn_base`). If using BFCL v4 data, ensure you select only the `multi_turn_base` subset.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install BFCL Evaluation Package

```bash
# Clone Gorilla repo
git clone https://github.com/ShishirPatil/gorilla.git

# Install BFCL package
cd gorilla/berkeley-function-call-leaderboard
pip install -e .
pip install -r requirements.txt
```

### 3. Setup BFCL Data

```bash
# This will clone Gorilla, copy data, and split into train/val
python setup_bfcl_data.py
```

### 4. Configure AWS Credentials (Bedrock only)

Ensure your AWS credentials are configured for Bedrock access:

```bash
# Option 1: Environment variables
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-west-2

# Option 2: AWS CLI
aws configure
```

## Usage

### Basic Run (Bedrock Claude)

```bash
python scripts/run_bfcl.py \
    --data-path data/multiturn_data_base_val.jsonl \
    --model-id us.anthropic.claude-sonnet-4-20250514-v1:0 \
    --backend bedrock
```

### With ReMe Memory

First, start the ReMe service:
```bash
reme \
    backend=http \
    http.port=8002 \
    llm.default.model_name=your-model \
    embedding_model.default.model_name=text-embedding-v4 \
    vector_store.default.backend=local
```

Then run with memory enabled:
```bash
python scripts/run_bfcl.py \
    --data-path data/multiturn_data_base_val.jsonl \
    --use-memory \
    --memory-base-url http://0.0.0.0:8002/ \
    --backend bedrock
```

### With Claude Thinking Mode

```bash
python scripts/run_bfcl.py \
    --data-path data/multiturn_data_base_val.jsonl \
    --enable-thinking \
    --backend bedrock

### Qwen3 via vLLM (OpenAI-Compatible)

Start a vLLM server (OpenAI-compatible) for Qwen3. With 8x L40s, reserve GPU 1 for embeddings and use data parallel across the remaining 7 GPUs:

```bash
QWEN3_DEVICE=0,2,3,4,5,6,7 \
QWEN3_DP=7 \
QWEN3_TP=1 \
../model_infra/qwen3/start_qwen3_vllm.sh
```

Start a vLLM embedding server for `Qwen/Qwen3-Embedding-0.6B` (separate GPU):

```bash
EMBEDDING_MODEL_NAME=Qwen/Qwen3-Embedding-0.6B \
EMBEDDING_PORT=5002 \
EMBEDDING_DEVICE=1 \
../model_infra/qwen3/start_qwen3_embedding.sh
```

Run BFCL with the OpenAI-compatible backend:

```bash
python scripts/run_bfcl.py \
  --data-path data/multiturn_data_base_val.jsonl \
  --model-id qwen3 \
  --backend openai \
  --base-url http://localhost:5001/v1 \
  --api-key dummy-key
```

### Full Pipeline (Bash)

Edit `pipeline/run_pipeline.sh` to set defaults (or override via env):
- `BACKEND` (`bedrock` or `openai`)
- `MODEL_ID` (e.g., `qwen3` for vLLM)
- `OPENAI_BASE_URL` (vLLM endpoint)
- `EMBEDDING_MODEL_NAME` / `EMBEDDING_BASE_URL` (embedding provider)
- `EMBEDDING_DEVICE` (GPU for embeddings)

Then run:

```bash
./pipeline/run_pipeline.sh
```

### Sequential Pipelines (Claude → Qwen)

Run Claude first:

```bash
./pipeline/run_pipeline_claude.sh
```

Then run Qwen3:

```bash
./pipeline/run_pipeline_qwen.sh
```

If you want to override defaults, set env vars before running. Example:

```bash
BACKEND=openai MODEL_ID=qwen3 OPENAI_BASE_URL=http://localhost:5001/v1 ./pipeline/run_pipeline.sh
```

### One-Click Sequential Run (Claude → Qwen)

This runs both backends in sequence and writes logs to `bfcl_models/logs/`.

```bash
./pipeline/run_pipeline_both.sh
```
```

## Available Claude Models on Bedrock

Common model IDs:
- `us.anthropic.claude-sonnet-4-20250514-v1:0` (Claude Sonnet 4)
- `us.anthropic.claude-3-5-sonnet-20241022-v2:0` (Claude 3.5 Sonnet)
- `us.anthropic.claude-3-7-sonnet-20250219-v1:0` (Claude 3.7 Sonnet)
- `us.anthropic.claude-3-opus-20240229-v1:0` (Claude 3 Opus)

## File Structure

```
bfcl_models/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── setup_bfcl_data.py        # Data setup script
├── scripts/
│   ├── run_bfcl.py           # Main runner (bedrock/openai)
│   ├── bfcl_agent.py         # Agent implementation
│   ├── build_memory_from_results.py
│   └── evaluate_results.py
├── pipeline/
│   └── run_pipeline.sh
│   └── run_pipeline_claude.sh
│   └── run_pipeline_qwen.sh
├── model_infra/qwen3/
│   ├── start_qwen3_vllm.sh
│   ├── qwen3_vllm_server.py
│   ├── start_qwen3_embedding.sh
│   └── qwen3_embedding_server.py
├── deprecated/
│   └── run_full_pipeline_DEPRECATED.sh
└── data/                     # BFCL data (after setup)
    ├── multiturn_data_base.jsonl
    ├── multiturn_data_base_train.jsonl
    ├── multiturn_data_base_val.jsonl
    └── possible_answer/
```

## Experimental Setup (Following ReMe Paper)

The ReMe paper uses the following setup for BFCL v3:
- 50 training samples (for building experience pool)
- 150 validation samples (for testing)
- Random split with seed 42

The `setup_bfcl_data.py` script creates this split by default.

## Results Format

Results are saved as JSONL files with the following structure:
```json
{
    "run_id": 0,
    "task_id": "multi_turn_base_0",
    "experiment_name": "bfcl-multi-turn-base_run",
    "task_completed": true,
    "reward": 1.0,
    "task_history": [...],
    "task_start_time": "2025-02-12 00:00:00"
}
```

## Comparison with Original ReMe BFCL

| Feature | Original (Qwen) | This (Bedrock/OpenAI) |
|---------|-----------------|-----------------------|
| API | OpenAI-compatible | Bedrock or OpenAI-compatible |
| Parallelization | Ray | ThreadPoolExecutor |
| Tool Format | OpenAI | OpenAI or Claude (converted) |
| Thinking | Extra param | Bedrock thinking or none |
