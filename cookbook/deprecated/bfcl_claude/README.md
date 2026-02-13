# BFCL v3 with Claude API (AWS Bedrock)

This directory contains the implementation of BFCL v3 (Berkeley Function Call Leaderboard) benchmark using Claude API via AWS Bedrock.

## Overview

BFCL v3 focuses on multi-turn function calling scenarios. This implementation adapts the original ReMe BFCL benchmark to use Claude models through AWS Bedrock instead of OpenAI-compatible APIs.

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

### 4. Configure AWS Credentials

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

### Basic Run

```bash
python run_bfcl_claude.py \
    --data-path data/multiturn_data_base_val.jsonl \
    --model-id us.anthropic.claude-sonnet-4-20250514-v1:0
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
python run_bfcl_claude.py \
    --data-path data/multiturn_data_base_val.jsonl \
    --use-memory \
    --memory-base-url http://0.0.0.0:8002/
```

### With Claude Thinking Mode

```bash
python run_bfcl_claude.py \
    --data-path data/multiturn_data_base_val.jsonl \
    --enable-thinking
```

## Available Claude Models on Bedrock

Common model IDs:
- `us.anthropic.claude-sonnet-4-20250514-v1:0` (Claude Sonnet 4)
- `us.anthropic.claude-3-5-sonnet-20241022-v2:0` (Claude 3.5 Sonnet)
- `us.anthropic.claude-3-7-sonnet-20250219-v1:0` (Claude 3.7 Sonnet)
- `us.anthropic.claude-3-opus-20240229-v1:0` (Claude 3 Opus)

## File Structure

```
bfcl_claude/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── setup_bfcl_data.py        # Data setup script
├── run_bfcl_claude.py        # Main runner script
├── claude_bfcl_agent.py      # Claude BFCL agent implementation
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
    "experiment_name": "bfcl-multi-turn-base_claude",
    "task_completed": true,
    "reward": 1.0,
    "task_history": [...],
    "task_start_time": "2025-02-12 00:00:00"
}
```

## Comparison with Original ReMe BFCL

| Feature | Original (Qwen) | This (Claude) |
|---------|-----------------|---------------|
| API | OpenAI-compatible | AWS Bedrock |
| Parallelization | Ray | ThreadPoolExecutor |
| Tool Format | OpenAI | Claude (converted) |
| Thinking | Extra param | Native support |
