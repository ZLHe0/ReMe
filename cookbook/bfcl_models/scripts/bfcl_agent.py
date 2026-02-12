#!/usr/bin/env python3
"""
BFCL v3 Agent with pluggable backends (Bedrock Claude or OpenAI-compatible).
"""

import os
import re
import json
import time
import warnings
import tempfile
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import boto3
from botocore.exceptions import ClientError
from openai import OpenAI
from loguru import logger
from tqdm import tqdm

# Import BFCL utilities from parent directory
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "bfcl"))

from bfcl_utils import (
    load_test_case,
    handle_user_turn,
    handle_tool_calls,
    extract_tool_schema,
    extract_single_turn_response,
    extract_multi_turn_responses,
    capture_and_print_score_files,
    create_error_response,
)

try:
    from bfcl_eval.model_handler.api_inference.qwen import QwenAPIHandler
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
        is_empty_execute_response,
    )
    from bfcl_eval.eval_checker.eval_runner import (
        multi_turn_runner,
        ast_file_runner,
    )
    from bfcl_eval.eval_checker.eval_runner_helper import record_cost_latency
    from bfcl_eval.utils import (
        is_multi_turn,
        is_relevance_or_irrelevance,
        find_file_with_suffix,
        load_file,
    )
    BFCL_EVAL_AVAILABLE = True
except ImportError:
    logger.warning("bfcl_eval not available. Evaluation features will be disabled.")
    BFCL_EVAL_AVAILABLE = False

# ─── CONFIGURATION ────────────────────────────────────────────────────────────────
DEFAULT_MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"
MAX_TOKENS = 4096
THINKING_BUDGET = 2048

def get_bedrock_client(region_name: str = "us-west-2"):
    """
    Initialize and return a boto3 Bedrock client.

    Args:
        region_name: AWS region name

    Returns:
        Boto3 Bedrock client
    """
    session = boto3.Session()
    return session.client('bedrock-runtime', region_name=region_name)


def is_throttling(exc):
    """Check if the exception is due to throttling."""
    return (
        isinstance(exc, ClientError) and
        exc.response.get("Error", {}).get("Code") == "ThrottlingException"
    )


def convert_openai_tools_to_claude(openai_tools: List[Dict]) -> List[Dict]:
    """
    Convert OpenAI-format tools to Claude/Bedrock format.

    Args:
        openai_tools: List of tools in OpenAI format

    Returns:
        List of tools in Claude/Bedrock format
    """
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


def convert_claude_tool_use_to_openai(tool_use_blocks: List[Dict]) -> List[Dict]:
    """
    Convert Claude tool_use response blocks to OpenAI tool_calls format.

    Args:
        tool_use_blocks: List of Claude tool_use blocks

    Returns:
        List of tool calls in OpenAI format
    """
    tool_calls = []
    for i, block in enumerate(tool_use_blocks):
        if block.get("type") == "tool_use":
            tool_call = {
                "id": block.get("id", f"call_{i}"),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {}))
                }
            }
            tool_calls.append(tool_call)
    return tool_calls


class BFCLAgent:
    """A BFCL Agent with Bedrock Claude or OpenAI-compatible backends."""

    def __init__(
        self,
        index: int,
        task_ids: List[str],
        experiment_name: str,
        data_path: str = "data/multiturn_data_base_val.jsonl",
        answer_path: Path = Path("data/possible_answer"),
        model_id: str = DEFAULT_MODEL_ID,
        backend: str = "bedrock",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_interactions: int = 30,
        max_response_size: int = 2000,
        num_trials: int = 1,
        enable_thinking: bool = False,
        use_memory: bool = False,
        use_memory_addition: bool = False,
        use_memory_deletion: bool = False,
        delete_freq: int = 10,
        freq_threshold: int = 5,
        utility_threshold: float = 0.5,
        memory_base_url: str = "http://0.0.0.0:8002/",
        memory_workspace_id: str = "bfcl_v3",
        region_name: str = "us-west-2",
    ):
        self.index: int = index
        self.task_ids: List[str] = task_ids
        self.categories: List[str] = [
            task_id.rsplit("_", 1)[0] if "_" in task_id else task_id
            for task_id in task_ids
        ]
        self.experiment_name: str = experiment_name
        self.data_path: str = data_path
        self.answer_path: Path = answer_path
        self.model_id: str = model_id
        self.backend: str = backend
        self.base_url: Optional[str] = base_url
        self.api_key: Optional[str] = api_key
        self.temperature: float = temperature
        self.max_interactions: int = max_interactions
        self.max_response_size: int = max_response_size
        self.num_trials: int = num_trials
        self.enable_thinking: bool = enable_thinking
        self.use_memory: bool = use_memory
        self.use_memory_addition: bool = use_memory_addition if use_memory else False
        self.use_memory_deletion: bool = use_memory_deletion if use_memory else False
        self.delete_freq: int = delete_freq
        self.freq_threshold: int = freq_threshold
        self.utility_threshold: float = utility_threshold
        self.memory_base_url: str = memory_base_url
        self.memory_workspace_id: str = memory_workspace_id

        # Initialize backend client
        if self.backend == "bedrock":
            self.client = get_bedrock_client(region_name)
            self.openai_client = None
        elif self.backend == "openai":
            if not self.base_url:
                raise ValueError("base_url is required for backend='openai'")
            self.openai_client = OpenAI(base_url=self.base_url, api_key=self.api_key or "dummy-key")
            self.client = None
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        # Initialize state
        self.history: List[List[List[dict]]] = [[] for _ in range(num_trials)]
        self.retrieved_memory_list: List[List[List[Any]]] = [[] for _ in range(num_trials)]
        self.test_entry: List[List[Dict[str, Any]]] = [[] for _ in range(num_trials)]
        self.original_test_entry: List[List[Dict[str, Any]]] = [[] for _ in range(num_trials)]
        self.tool_schema: List[List[List[dict]]] = [[] for _ in range(num_trials)]
        self.current_turn = [[0 for _ in range(len(task_ids))] for _ in range(num_trials)]

        for run_id in range(num_trials):
            for task_index in range(len(task_ids)):
                self.init_state(run_id, task_index)

    def init_state(self, run_id: int, i: int) -> Dict[str, Any]:
        """Initialize state for a task."""
        self.test_entry[run_id].append(load_test_case(self.data_path, self.task_ids[i]))
        self.original_test_entry[run_id].append(self.test_entry[run_id][i].get("extra", {}))
        self.tool_schema[run_id].append(
            extract_tool_schema(self.test_entry[run_id][i].get("tools", [{}]))
        )

        msg = self.test_entry[run_id][i].get("messages", [])
        self.history[run_id].append(msg)
        self.retrieved_memory_list[run_id].append([])
        self.current_turn[run_id][i] = 1

    def call_llm(self, messages: List[dict], tool_schemas: List[dict]) -> dict:
        """
        Call the configured LLM backend with tool support.

        Args:
            messages: List of conversation messages in OpenAI format
            tool_schemas: List of tools in OpenAI format

        Returns:
            Assistant message in OpenAI format
        """
        if self.backend == "bedrock":
            return self._call_bedrock(messages, tool_schemas)
        return self._call_openai(messages, tool_schemas)

    def _call_bedrock(self, messages: List[dict], tool_schemas: List[dict]) -> dict:
        # Convert messages from OpenAI format to Claude format
        claude_messages = self._convert_messages_to_claude(messages)
        claude_tools = convert_openai_tools_to_claude(tool_schemas)

        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": MAX_TOKENS,
            "temperature": 1.0 if self.enable_thinking else self.temperature,
            "messages": claude_messages,
        }

        if claude_tools:
            payload["tools"] = claude_tools
            payload["tool_choice"] = {"type": "auto"}

        if self.enable_thinking:
            payload["thinking"] = {"type": "enabled", "budget_tokens": THINKING_BUDGET}

        for attempt in range(100):
            try:
                resp = self.client.invoke_model(
                    body=json.dumps(payload),
                    modelId=self.model_id,
                )
                resp_body = json.loads(resp["body"].read())
                return self._convert_claude_response_to_openai(resp_body)
            except ClientError as e:
                if is_throttling(e):
                    logger.warning(f"Throttling, attempt {attempt + 1}/100")
                    time.sleep(1 + attempt * 2)
                else:
                    raise
            except Exception as e:
                logger.exception(f"Error calling Bedrock Claude: {e}")
                time.sleep(1 + attempt * 10)

        return {"role": "assistant", "content": "Error: Failed to get response from Bedrock"}

    def _call_openai(self, messages: List[dict], tool_schemas: List[dict]) -> dict:
        for attempt in range(20):
            try:
                resp = self.openai_client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    tools=tool_schemas if tool_schemas else None,
                    tool_choice="auto" if tool_schemas else None,
                    temperature=self.temperature,
                )
                msg = resp.choices[0].message
                out = {"role": "assistant", "content": msg.content or ""}
                if getattr(msg, "tool_calls", None):
                    out["tool_calls"] = []
                    for tc in msg.tool_calls:
                        out["tool_calls"].append(
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                        )
                return out
            except Exception as e:
                logger.exception(f"Error calling OpenAI-compatible backend: {e}")
                time.sleep(1 + attempt * 2)

        return {"role": "assistant", "content": "Error: Failed to get response from OpenAI backend"}

    def _convert_messages_to_claude(self, messages: List[dict]) -> List[dict]:
        """
        Convert OpenAI format messages to Claude format.

        Args:
            messages: Messages in OpenAI format

        Returns:
            Messages in Claude format
        """
        claude_messages = []

        for msg in messages:
            role = msg.get("role", "")

            if role == "system":
                # Claude handles system separately, skip here
                continue
            elif role == "user":
                claude_messages.append({
                    "role": "user",
                    "content": msg.get("content", "")
                })
            elif role == "assistant":
                content = []

                # Add text content if present
                if msg.get("content"):
                    content.append({
                        "type": "text",
                        "text": msg.get("content", "")
                    })

                # Add tool calls if present
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
                    claude_messages.append({
                        "role": "assistant",
                        "content": content
                    })
            elif role == "tool":
                # Find the corresponding tool_use_id
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

    def _convert_claude_response_to_openai(self, resp_body: dict) -> dict:
        """
        Convert Claude response to OpenAI format.

        Args:
            resp_body: Claude API response body

        Returns:
            Message in OpenAI format
        """
        content_blocks = resp_body.get("content", [])

        text_parts = []
        tool_use_blocks = []
        reasoning_content = ""

        for block in content_blocks:
            block_type = block.get("type", "")

            if block_type == "text":
                text_parts.append(block.get("text", ""))
            elif block_type == "tool_use":
                tool_use_blocks.append(block)
            elif block_type == "thinking":
                reasoning_content = block.get("thinking", "")

        # Build OpenAI-format message
        msg = {
            "role": "assistant",
            "content": "\n".join(text_parts) if text_parts else "",
        }

        if reasoning_content:
            msg["reasoning_content"] = reasoning_content

        if tool_use_blocks:
            msg["tool_calls"] = convert_claude_tool_use_to_openai(tool_use_blocks)

        return msg

    def env_step(self, run_id: int, index: int, messages: List[dict]) -> dict:
        """
        Process one step in the conversation.

        Args:
            run_id: Current run ID
            index: Task index
            messages: List of conversation messages

        Returns:
            Dict containing next message and tools if applicable
        """
        try:
            if not messages:
                return handle_user_turn(
                    self.original_test_entry[run_id][index],
                    self.current_turn[run_id][index]
                )

            if messages[-1]["role"] != "assistant":
                return create_error_response("Last message must be from assistant")

            if "tool_calls" in messages[-1] and len(messages[-1]["tool_calls"]) > 0:
                try:
                    tool_calls = messages[-1]["tool_calls"]
                    decoded_calls = self._convert_tool_calls_to_execution_format(tool_calls)

                    logger.debug(f"decoded_calls: {decoded_calls}")

                    if BFCL_EVAL_AVAILABLE and is_empty_execute_response(decoded_calls):
                        warnings.warn(f"Empty execute response: {decoded_calls}")
                        return handle_user_turn(
                            self.original_test_entry[run_id][index],
                            self.current_turn[run_id][index],
                        )

                    return handle_tool_calls(
                        tool_calls,
                        decoded_calls,
                        self.original_test_entry[run_id][index],
                        self.current_turn[run_id][index],
                    )
                except Exception as e:
                    warnings.warn(f"Error during tool invocation: {str(e)}")
                    return handle_user_turn(
                        self.original_test_entry[run_id][index],
                        self.current_turn[run_id][index]
                    )
            else:
                return handle_user_turn(
                    self.original_test_entry[run_id][index],
                    self.current_turn[run_id][index]
                )

        except Exception as e:
            return create_error_response(f"Failed to process request: {str(e)}")

    def _convert_tool_calls_to_execution_format(self, tool_calls: List[Dict]) -> List[str]:
        """
        Convert tool calls to execution format.

        Args:
            tool_calls: List of tool calls in OpenAI format

        Returns:
            List of function calls in string format
        """
        execution_list = []

        for tool_call in tool_calls:
            function = tool_call.get("function", {})
            function_name = function.get("name", "")

            try:
                arguments = function.get("arguments", "{}")
                if isinstance(arguments, str):
                    args_dict = json.loads(arguments)
                else:
                    args_dict = arguments

                args_str = ", ".join([f"{k}={repr(v)}" for k, v in args_dict.items()])
                execution_list.append(f"{function_name}({args_str})")

            except Exception:
                execution_list.append(f"{function_name}()")

        return execution_list

    def get_reward(self, run_id: int, index: int) -> float:
        """
        Get reward/accuracy score for the task.

        Args:
            run_id: Current run ID
            index: Task index

        Returns:
            Accuracy score (0.0 or 1.0)
        """
        if not BFCL_EVAL_AVAILABLE:
            logger.warning("bfcl_eval not available, returning 0.0")
            return 0.0

        try:
            if not self.history[run_id][index] or not self.original_test_entry[run_id][index]:
                return 0.0

            model_name = "bfcl_agent"
            handler = QwenAPIHandler(model_name, temperature=1.0)

            model_result_data = self._convert_conversation_to_eval_format(run_id, index)
            prompt_data = [self.original_test_entry[run_id][index]]

            state = {"leaderboard_table": {}}
            record_cost_latency(
                state["leaderboard_table"],
                model_name,
                [model_result_data],
            )

            if is_relevance_or_irrelevance(self.categories[index]):
                accuracy, _ = self._eval_relevance_test(
                    handler,
                    model_result_data,
                    prompt_data,
                    model_name,
                    self.categories[index],
                )
            else:
                possible_answer_file = find_file_with_suffix(
                    self.answer_path,
                    self.categories[index],
                )
                possible_answer = load_file(possible_answer_file, sort_by_id=True)
                possible_answer = [
                    item for item in possible_answer
                    if item["id"] == self.task_ids[index]
                ]

                if is_multi_turn(self.categories[index]):
                    accuracy, _ = self._eval_multi_turn_test(
                        handler,
                        model_result_data,
                        prompt_data,
                        possible_answer,
                        model_name,
                        self.categories[index],
                    )
                else:
                    accuracy, _ = self._eval_single_turn_test(
                        handler,
                        model_result_data,
                        prompt_data,
                        possible_answer,
                        model_name,
                        self.categories[index],
                    )

            return accuracy

        except Exception as e:
            import traceback
            traceback.print_exc()
            return 0.0

    def _convert_conversation_to_eval_format(self, run_id: int, index: int) -> Dict[str, Any]:
        """Convert conversation history to evaluation format."""
        if is_multi_turn(self.categories[index]):
            turns_data = extract_multi_turn_responses(self.history[run_id][index])
        else:
            turns_data = extract_single_turn_response(self.history[run_id][index])

        return {
            "id": self.task_ids[index],
            "result": turns_data,
            "latency": 0,
            "input_token_count": 0,
            "output_token_count": 0,
        }

    def _eval_multi_turn_test(
        self,
        handler,
        model_result_data,
        prompt_data,
        possible_answer,
        model_name,
        test_category,
    ):
        """Evaluate multi-turn test."""
        with tempfile.TemporaryDirectory() as temp_dir:
            score_dir = Path(temp_dir)
            accuracy, total_count = multi_turn_runner(
                handler=handler,
                model_result=[model_result_data],
                prompt=prompt_data,
                possible_answer=possible_answer,
                model_name=model_name,
                test_category=test_category,
                score_dir=score_dir,
            )
            return accuracy, total_count

    def _eval_single_turn_test(
        self,
        handler,
        model_result_data,
        prompt_data,
        possible_answer,
        model_name,
        test_category,
    ):
        """Evaluate single-turn test."""
        language = "Python"
        if "java" in test_category.lower():
            language = "Java"
        elif "js" in test_category.lower() or "javascript" in test_category.lower():
            language = "JavaScript"

        with tempfile.TemporaryDirectory() as temp_dir:
            score_dir = Path(temp_dir)
            accuracy, total_count = ast_file_runner(
                handler=handler,
                model_result=[model_result_data],
                prompt=prompt_data,
                possible_answer=possible_answer,
                language=language,
                test_category=test_category,
                model_name=model_name,
                score_dir=score_dir,
            )
            return accuracy, total_count

    def task_completed(self, run_id: int, index: int) -> bool:
        """Check if task is completed."""
        return self.history[run_id][index][-1].get("content", "") == "[CONVERSATION_COMPLETED]"

    def execute(self) -> List[dict]:
        """
        Execute all tasks.

        Returns:
            List of result dictionaries
        """
        results = []

        for task_index, task_id in enumerate(tqdm(self.task_ids, desc=f"Agent {self.index}")):
            t_result = None

            for run_id in range(self.num_trials):
                try:
                    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    for i in range(self.max_interactions):
                        # Call LLM
                        llm_output = self.call_llm(
                            self.history[run_id][task_index],
                            self.tool_schema[run_id][task_index],
                        )
                        self.history[run_id][task_index].append(llm_output)

                        # Get environment response
                        env_output = self.env_step(
                            run_id, task_index,
                            self.history[run_id][task_index]
                        )

                        # Update tool schema if provided
                        if "tools" in env_output:
                            self.tool_schema[run_id][task_index] = extract_tool_schema(
                                env_output["tools"]
                            )

                        # Process environment output
                        new_tool_calls = []
                        new_tool_call_ids = []
                        next_user_msg = ""

                        for msg in env_output.get("messages", []):
                            if msg["role"] == "tool" and len(msg.get("content", "")) > 0:
                                new_tool_calls.append(msg.get("content", ""))
                                new_tool_call_ids.append(msg.get("tool_call_id", ""))
                            elif msg["role"] == "user":
                                next_user_msg = msg.get("content", "")
                                self.current_turn[run_id][task_index] += 1
                            else:
                                next_user_msg = msg.get("content", "")

                        if new_tool_calls:
                            for idx, call in enumerate(new_tool_calls):
                                self.history[run_id][task_index].append({
                                    "role": "tool",
                                    "content": str(call),
                                    "tool_call_id": new_tool_call_ids[idx]
                                })
                        else:
                            self.history[run_id][task_index].append({
                                "role": "user",
                                "content": next_user_msg
                            })

                        logger.info(f"Agent {self.index}, task {task_id}, iteration {i}")

                        if self.task_completed(run_id, task_index):
                            break

                    reward = self.get_reward(run_id, task_index)

                    t_result = {
                        "run_id": run_id,
                        "task_id": self.task_ids[task_index],
                        "experiment_name": self.experiment_name,
                        "task_completed": self.task_completed(run_id, task_index),
                        "reward": reward,
                        "task_history": self.history[run_id][task_index],
                        "task_start_time": start_time,
                    }

                    if reward == 1:
                        break

                except Exception as e:
                    logger.exception(f"Error in task {task_id}: {e}")
                    results.append({})

            results.append(t_result)

        return results


def main():
    """Run a simple test."""
    agent = BFCLAgent(
        index=0,
        task_ids=["multi_turn_base_0"],
        experiment_name="bfcl_agent_test",
        data_path="data/multiturn_data_base_val.jsonl",
        answer_path=Path("data/possible_answer"),
        backend="bedrock",
    )

    results = agent.execute()
    logger.info(f"Results: {json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()
