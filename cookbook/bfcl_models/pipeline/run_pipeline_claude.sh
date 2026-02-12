#!/bin/bash
set -e

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export BACKEND=bedrock
export MODEL_ID="${MODEL_ID:-us.anthropic.claude-sonnet-4-20250514-v1:0}"
export AWS_REGION="${AWS_REGION:-us-west-2}"

exec "$PIPELINE_DIR/run_pipeline.sh"
