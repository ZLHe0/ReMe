#!/usr/bin/env python3
"""
Complete BFCL v3 Pipeline with Claude API and ReMe Memory.

This script runs the full pipeline:
1. Run inference without memory on training data to collect trajectories
2. Build the memory pool from successful trajectories using ReMe
3. Run inference with memory on validation data
4. Compare results
"""

import os
import sys
import json
import time
import random
import argparse
import datetime
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from loguru import logger
from tqdm import tqdm

# Add BFCL to path
sys.path.insert(0, "gorilla/berkeley-function-call-leaderboard")

from bfcl_eval.constants.executable_backend_config import MULTI_TURN_FUNC_DOC_FILE_MAPPING
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.model_handler.utils import convert_to_tool
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    execute_multi_turn_func_call,
    is_empty_execute_response,
)

# Configuration
DEFAULT_MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"
MAX_TOKENS = 4096
FUNC_DOC_DIR = "gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/multi_turn_func_doc"


def get_bedrock_client(region_name: str = "us-west-2"):
    """Initialize and return a boto3 Bedrock client."""
    session = boto3.Session()
    return session.client('bedrock-runtime', region_name=region_name)


def is_throttling(exc):
    """Check if the exception is due to throttling."""
    return (
        isinstance(exc, ClientError) and
        exc.response.get("Error", {}).get("Code") == "ThrottlingException"
    )


def load_functions_for_test(test_entry: dict) -> List[dict]:
    """Load function definitions for a test entry."""
    functions = []
    involved_classes = test_entry.get("involved_classes", [])

    for cls in involved_classes:
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


def load_test_entry(data_path: str, task_id: str) -> dict:
    """Load a test entry by ID and add function definitions."""
    with open(data_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("id") == task_id:
                # Load functions
                entry["function"] = load_functions_for_test(entry)
                return entry
    raise ValueError(f"Task {task_id} not found in {data_path}")


def convert_functions_to_tools(functions: List[dict], test_category: str) -> List[dict]:
    """Convert function definitions to OpenAI tool format."""
    tools = convert_to_tool(functions, GORILLA_TO_OPENAPI, ModelStyle.OPENAI_COMPLETIONS)
    # Remove response field from tools
    for tool in tools:
        if "function" in tool and "response" in tool["function"]:
            del tool["function"]["response"]
    return tools


def convert_openai_tools_to_claude(openai_tools: List[dict]) -> List[dict]:
    """Convert OpenAI-format tools to Claude/Bedrock format."""
    claude_tools = []
    for tool in openai_tools:
        if tool.get("type") == "function":
            func = tool.get("function", {})
            claude_tool = {
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {"type": "object", "properties": {}})
            }
            claude_tools.append(claude_tool)
    return claude_tools


def convert_messages_to_claude(messages: List[dict]) -> List[dict]:
    """Convert OpenAI format messages to Claude format."""
    claude_messages = []

    for msg in messages:
        role = msg.get("role", "")

        if role == "system":
            continue
        elif role == "user":
            claude_messages.append({
                "role": "user",
                "content": msg.get("content", "")
            })
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
            tool_call_id = msg.get("tool_call_id", "")
            claude_messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": str(msg.get("content", ""))
                }]
            })

    return claude_messages


def convert_claude_response_to_openai(resp_body: dict) -> dict:
    """Convert Claude response to OpenAI format."""
    content_blocks = resp_body.get("content", [])

    text_parts = []
    tool_use_blocks = []

    for block in content_blocks:
        block_type = block.get("type", "")
        if block_type == "text":
            text_parts.append(block.get("text", ""))
        elif block_type == "tool_use":
            tool_use_blocks.append(block)

    msg = {
        "role": "assistant",
        "content": "\n".join(text_parts) if text_parts else "",
    }

    if tool_use_blocks:
        msg["tool_calls"] = []
        for i, block in enumerate(tool_use_blocks):
            msg["tool_calls"].append({
                "id": block.get("id", f"call_{i}"),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {}))
                }
            })

    return msg


@retry(
    retry=retry_if_exception(is_throttling),
    stop=stop_after_attempt(20),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def call_claude(client, messages: List[dict], tools: List[dict], model_id: str, temperature: float = 0.7) -> dict:
    """Call Claude API with tools."""
    claude_messages = convert_messages_to_claude(messages)
    claude_tools = convert_openai_tools_to_claude(tools)

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": MAX_TOKENS,
        "temperature": temperature,
        "messages": claude_messages,
    }

    if claude_tools:
        payload["tools"] = claude_tools
        payload["tool_choice"] = {"type": "auto"}

    resp = client.invoke_model(body=json.dumps(payload), modelId=model_id)
    resp_body = json.loads(resp['body'].read())

    return convert_claude_response_to_openai(resp_body)


def execute_tool_calls(tool_calls: List[dict], test_entry: dict) -> List[dict]:
    """Execute tool calls and return results."""
    # Convert tool calls to execution format
    decoded_calls = []
    for tc in tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "")
        args = func.get("arguments", "{}")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except:
                args = {}
        args_str = ", ".join([f"{k}={repr(v)}" for k, v in args.items()])
        decoded_calls.append(f"{name}({args_str})")

    if is_empty_execute_response(decoded_calls):
        return []

    # Execute
    results, _ = execute_multi_turn_func_call(
        func_call_list=decoded_calls,
        initial_config=test_entry.get("initial_config", {}),
        involved_classes=test_entry.get("involved_classes", []),
        model_name="claude_agent",
        test_entry_id=test_entry.get("id", ""),
        long_context=False,
        is_evaL_run=False,
    )

    return results


def run_single_task(
    client,
    task_id: str,
    data_path: str,
    model_id: str,
    max_turns: int = 30,
    memory_content: str = None
) -> dict:
    """Run a single BFCL task and return the result."""

    # Load test entry
    test_entry = load_test_entry(data_path, task_id)
    test_category = task_id.rsplit("_", 1)[0]

    # Convert functions to tools
    tools = convert_functions_to_tools(test_entry["function"], test_category)

    # Initialize conversation with first user message
    questions = test_entry.get("question", [])
    if not questions:
        return {"task_id": task_id, "reward": 0, "error": "No questions"}

    # Build initial message with optional memory
    first_question = questions[0][0]["content"] if questions[0] else ""
    if memory_content:
        first_question = f"{first_question}\n\nHere is some relevant experience that might help:\n{memory_content}"

    history = [{"role": "user", "content": first_question}]
    current_turn = 1

    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for iteration in range(max_turns):
        try:
            # Call Claude
            response = call_claude(client, history, tools, model_id)
            history.append(response)

            # Check for tool calls
            if "tool_calls" in response and response["tool_calls"]:
                # Execute tool calls
                results = execute_tool_calls(response["tool_calls"], test_entry)

                if results:
                    for i, (tc, result) in enumerate(zip(response["tool_calls"], results)):
                        history.append({
                            "role": "tool",
                            "content": str(result),
                            "tool_call_id": tc.get("id", f"call_{i}")
                        })
                else:
                    # No results, move to next turn
                    if current_turn < len(questions):
                        next_msg = questions[current_turn][0]["content"] if questions[current_turn] else ""
                        history.append({"role": "user", "content": next_msg})
                        current_turn += 1
                    else:
                        history.append({"role": "user", "content": "[CONVERSATION_COMPLETED]"})
                        break
            else:
                # No tool calls, move to next turn
                if current_turn < len(questions):
                    next_msg = questions[current_turn][0]["content"] if questions[current_turn] else ""
                    history.append({"role": "user", "content": next_msg})
                    current_turn += 1
                else:
                    history.append({"role": "user", "content": "[CONVERSATION_COMPLETED]"})
                    break

            # Check for completion
            if history[-1].get("content") == "[CONVERSATION_COMPLETED]":
                break

        except Exception as e:
            logger.error(f"Error in task {task_id} iteration {iteration}: {e}")
            break

    # Simple reward: 1 if completed all turns, 0 otherwise
    completed = current_turn >= len(questions)
    reward = 1.0 if completed else 0.0

    return {
        "task_id": task_id,
        "reward": reward,
        "completed": completed,
        "turns_completed": current_turn,
        "total_turns": len(questions),
        "history": history,
        "start_time": start_time
    }


def run_without_memory(
    data_path: str,
    output_path: Path,
    model_id: str,
    max_tasks: int = None,
    region_name: str = "us-west-2"
) -> List[dict]:
    """Run inference without memory on training data."""
    logger.info(f"Running inference without memory on {data_path}")

    client = get_bedrock_client(region_name)

    # Load task IDs
    task_ids = []
    with open(data_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            task_ids.append(entry["id"])

    if max_tasks:
        task_ids = task_ids[:max_tasks]

    logger.info(f"Running {len(task_ids)} tasks")

    results = []
    for task_id in tqdm(task_ids, desc="Without memory"):
        result = run_single_task(client, task_id, data_path, model_id)
        results.append(result)

        # Save incrementally
        with open(output_path / "train_results.jsonl", "a") as f:
            f.write(json.dumps(result) + "\n")

        time.sleep(0.5)  # Rate limiting

    # Summary
    successful = sum(1 for r in results if r.get("reward", 0) == 1)
    logger.info(f"Training results: {successful}/{len(results)} successful")

    return results


def build_memory_pool(
    results: List[dict],
    memory_base_url: str,
    workspace_id: str
) -> bool:
    """Build memory pool from successful trajectories using ReMe."""
    logger.info("Building memory pool from successful trajectories")

    # Filter successful trajectories
    successful = [r for r in results if r.get("reward", 0) == 1]
    logger.info(f"Found {len(successful)} successful trajectories")

    if not successful:
        logger.warning("No successful trajectories to build memory from")
        return False

    # Format trajectories for ReMe
    trajectories = []
    for result in successful:
        traj = {
            "task_id": result["task_id"],
            "messages": result.get("history", []),
            "score": result["reward"]
        }
        trajectories.append(traj)

    # Send to ReMe
    try:
        response = requests.post(
            f"{memory_base_url}summary_task_memory",
            json={
                "workspace_id": workspace_id,
                "trajectories": trajectories
            },
            timeout=300
        )

        if response.status_code == 200:
            result = response.json()
            memory_count = len(result.get("metadata", {}).get("memory_list", []))
            logger.info(f"Successfully built memory pool with {memory_count} memories")
            return True
        else:
            logger.error(f"Failed to build memory: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        logger.error(f"Error building memory pool: {e}")
        return False


def retrieve_memory(query: str, memory_base_url: str, workspace_id: str, top_k: int = 3) -> str:
    """Retrieve relevant memory for a query."""
    try:
        response = requests.post(
            f"{memory_base_url}retrieve_task_memory",
            json={
                "workspace_id": workspace_id,
                "query": query,
                "top_k": top_k
            },
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            return result.get("answer", "")
        return ""
    except Exception as e:
        logger.warning(f"Error retrieving memory: {e}")
        return ""


def run_with_memory(
    data_path: str,
    output_path: Path,
    model_id: str,
    memory_base_url: str,
    workspace_id: str,
    max_tasks: int = None,
    region_name: str = "us-west-2"
) -> List[dict]:
    """Run inference with memory on validation data."""
    logger.info(f"Running inference with memory on {data_path}")

    client = get_bedrock_client(region_name)

    # Load task IDs
    task_ids = []
    with open(data_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            task_ids.append(entry["id"])

    if max_tasks:
        task_ids = task_ids[:max_tasks]

    logger.info(f"Running {len(task_ids)} tasks with memory")

    results = []
    for task_id in tqdm(task_ids, desc="With memory"):
        # Load test entry to get first question for memory retrieval
        test_entry = load_test_entry(data_path, task_id)
        questions = test_entry.get("question", [])
        first_question = questions[0][0]["content"] if questions and questions[0] else ""

        # Retrieve memory
        memory_content = retrieve_memory(first_question, memory_base_url, workspace_id)

        result = run_single_task(client, task_id, data_path, model_id, memory_content=memory_content)
        result["memory_used"] = bool(memory_content)
        results.append(result)

        # Save incrementally
        with open(output_path / "val_results_with_memory.jsonl", "a") as f:
            f.write(json.dumps(result) + "\n")

        time.sleep(0.5)  # Rate limiting

    # Summary
    successful = sum(1 for r in results if r.get("reward", 0) == 1)
    logger.info(f"Validation with memory: {successful}/{len(results)} successful")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run BFCL v3 Pipeline with Claude and ReMe")

    parser.add_argument("--train-path", type=str, default="data/multiturn_data_base_train.jsonl")
    parser.add_argument("--val-path", type=str, default="data/multiturn_data_base_val.jsonl")
    parser.add_argument("--output-dir", type=str, default="exp_result")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--memory-url", type=str, default="http://localhost:8002/")
    parser.add_argument("--workspace-id", type=str, default="bfcl_v3_claude")
    parser.add_argument("--max-train-tasks", type=int, default=None)
    parser.add_argument("--max-val-tasks", type=int, default=None)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-memory-build", action="store_true")
    parser.add_argument("--region", type=str, default="us-west-2")

    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("BFCL v3 Pipeline with Claude API and ReMe Memory")
    logger.info("=" * 60)

    # Step 1: Run without memory on training data
    if not args.skip_training:
        logger.info("\n=== Step 1: Collecting trajectories on training data ===")
        train_results = run_without_memory(
            args.train_path,
            output_path,
            args.model_id,
            args.max_train_tasks,
            args.region
        )
    else:
        # Load existing results
        train_results = []
        train_file = output_path / "train_results.jsonl"
        if train_file.exists():
            with open(train_file, "r") as f:
                for line in f:
                    train_results.append(json.loads(line))
        logger.info(f"Loaded {len(train_results)} existing training results")

    # Step 2: Build memory pool
    if not args.skip_memory_build and train_results:
        logger.info("\n=== Step 2: Building memory pool ===")
        build_memory_pool(train_results, args.memory_url, args.workspace_id)

    # Step 3: Run with memory on validation data
    logger.info("\n=== Step 3: Running with memory on validation data ===")
    val_results_with_memory = run_with_memory(
        args.val_path,
        output_path,
        args.model_id,
        args.memory_url,
        args.workspace_id,
        args.max_val_tasks,
        args.region
    )

    # Step 4: Run without memory on validation data for comparison
    logger.info("\n=== Step 4: Running without memory on validation data (baseline) ===")
    val_results_baseline = []
    client = get_bedrock_client(args.region)

    task_ids = []
    with open(args.val_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            task_ids.append(entry["id"])

    if args.max_val_tasks:
        task_ids = task_ids[:args.max_val_tasks]

    for task_id in tqdm(task_ids, desc="Baseline"):
        result = run_single_task(client, task_id, args.val_path, args.model_id)
        val_results_baseline.append(result)

        with open(output_path / "val_results_baseline.jsonl", "a") as f:
            f.write(json.dumps(result) + "\n")

        time.sleep(0.5)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)

    baseline_success = sum(1 for r in val_results_baseline if r.get("reward", 0) == 1)
    memory_success = sum(1 for r in val_results_with_memory if r.get("reward", 0) == 1)
    total = len(val_results_baseline)

    logger.info(f"Validation without memory: {baseline_success}/{total} ({100*baseline_success/total:.1f}%)")
    logger.info(f"Validation with memory:    {memory_success}/{total} ({100*memory_success/total:.1f}%)")
    logger.info(f"Improvement:               {memory_success - baseline_success} tasks ({100*(memory_success-baseline_success)/total:.1f}%)")


if __name__ == "__main__":
    main()
