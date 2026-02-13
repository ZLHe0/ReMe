# AppWorld Models

This folder mirrors the `bfcl_models` layout but for AppWorld tasks. Shared model infrastructure
(Qwen3 vLLM + embedding servers and pipeline helpers) lives under:

- `../model_infra/`

## Quick Start

```bash
cd ReMe_bfcl_claude/cookbook/appworld_models
BACKEND=bedrock MODEL_ID=us.anthropic.claude-sonnet-4-20250514-v1:0 ./pipeline/run_pipeline.sh
```

Run a single backend:

```bash
./pipeline/run_pipeline_claude.sh
```

```bash
./pipeline/run_pipeline_qwen.sh
```

To run against a local Qwen3 vLLM server:

```bash
# Terminal 1 (LLM)
cd ReMe_bfcl_claude/cookbook/model_infra/qwen3
QWEN3_DEVICE=0,2,3,4,5,6,7 QWEN3_DP=7 QWEN3_TP=1 ./start_qwen3_vllm.sh

# Terminal 2 (AppWorld pipeline)
cd ReMe_bfcl_claude/cookbook/appworld_models
BACKEND=openai MODEL_ID=qwen3 OPENAI_BASE_URL=http://localhost:5001/v1 ./pipeline/run_pipeline.sh
```

Run Claude → Qwen sequentially with logs:

```bash
./pipeline/run_pipeline_both.sh
```

## Structure

- `scripts/` — AppWorld runner and agent logic
- `pipeline/` — orchestration script
- `deprecated/` — reserved for legacy scripts
