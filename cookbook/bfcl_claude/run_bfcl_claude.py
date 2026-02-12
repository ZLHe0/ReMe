#!/usr/bin/env python3
"""
BFCL v3 Runner using Claude API via AWS Bedrock.
This script runs the BFCL benchmark with Claude models.
"""

import time
import json
import argparse
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from claude_bfcl_agent import ClaudeBFCLAgent


def run_agent_task(
    worker_index: int,
    task_ids: list,
    experiment_name: str,
    data_path: str,
    answer_path: Path,
    model_id: str,
    num_trials: int,
    enable_thinking: bool,
    use_memory: bool,
    memory_base_url: str,
    memory_workspace_id: str,
    region_name: str,
) -> list:
    """
    Run agent for a subset of tasks.

    Args:
        worker_index: Index of this worker
        task_ids: List of task IDs to process
        Other args: Configuration parameters

    Returns:
        List of results
    """
    agent = ClaudeBFCLAgent(
        index=worker_index,
        task_ids=task_ids,
        experiment_name=experiment_name,
        data_path=data_path,
        answer_path=answer_path,
        model_id=model_id,
        num_trials=num_trials,
        enable_thinking=enable_thinking,
        use_memory=use_memory,
        memory_base_url=memory_base_url,
        memory_workspace_id=memory_workspace_id,
        region_name=region_name,
    )
    return agent.execute()


def run_bfcl_benchmark(
    dataset_name: str = "bfcl-multi-turn-base",
    experiment_suffix: str = "claude",
    max_workers: int = 4,
    num_trials: int = 1,
    model_id: str = "us.anthropic.claude-sonnet-4-20250514-v1:0",
    data_path: str = "data/multiturn_data_base_val.jsonl",
    answer_path: Path = Path("data/possible_answer"),
    use_memory: bool = False,
    enable_thinking: bool = False,
    memory_base_url: str = "http://0.0.0.0:8002/",
    memory_workspace_id: str = "bfcl_v3",
    region_name: str = "us-west-2",
):
    """
    Run the BFCL benchmark with Claude.

    Args:
        dataset_name: Name of the dataset
        experiment_suffix: Suffix for experiment name
        max_workers: Number of parallel workers
        num_trials: Number of trials per task
        model_id: Claude model ID
        data_path: Path to BFCL data
        answer_path: Path to answer files
        use_memory: Whether to use ReMe memory
        enable_thinking: Whether to enable Claude thinking
        memory_base_url: ReMe service URL
        memory_workspace_id: ReMe workspace ID
        region_name: AWS region
    """
    experiment_name = f"{dataset_name}_{experiment_suffix}"

    # Create output directory
    model_short = model_id.split("/")[-1].replace(":", "_")
    think_suffix = "with_think" if enable_thinking else "no_think"
    output_path = Path(f"./exp_result/{model_short}/{think_suffix}")
    output_path.mkdir(parents=True, exist_ok=True)

    # Load task IDs
    with open(data_path, "r", encoding="utf-8") as f:
        task_ids = [json.loads(line)["id"] for line in f]

    logger.info(f"Loaded {len(task_ids)} tasks from {data_path}")

    results = []

    if max_workers > 1:
        # Parallel execution
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            for i in range(max_workers):
                worker_task_ids = task_ids[i::max_workers]
                if not worker_task_ids:
                    continue

                future = executor.submit(
                    run_agent_task,
                    worker_index=i,
                    task_ids=worker_task_ids,
                    experiment_name=experiment_name,
                    data_path=data_path,
                    answer_path=answer_path,
                    model_id=model_id,
                    num_trials=num_trials,
                    enable_thinking=enable_thinking,
                    use_memory=use_memory,
                    memory_base_url=memory_base_url,
                    memory_workspace_id=memory_workspace_id,
                    region_name=region_name,
                )
                futures.append(future)
                time.sleep(1)  # Stagger starts

            logger.info(f"Submitted {len(futures)} workers")

            for i, future in enumerate(as_completed(futures)):
                try:
                    worker_results = future.result()
                    if worker_results:
                        results.extend(worker_results)
                    logger.info(f"Worker {i + 1}/{len(futures)} completed")
                except Exception as e:
                    logger.error(f"Worker failed: {e}")
    else:
        # Sequential execution
        agent = ClaudeBFCLAgent(
            index=0,
            task_ids=task_ids,
            experiment_name=experiment_name,
            data_path=data_path,
            answer_path=answer_path,
            model_id=model_id,
            num_trials=num_trials,
            enable_thinking=enable_thinking,
            use_memory=use_memory,
            memory_base_url=memory_base_url,
            memory_workspace_id=memory_workspace_id,
            region_name=region_name,
        )
        results = agent.execute()

    # Save results
    output_file = output_path / f"{experiment_name}.jsonl"
    with open(output_file, "w") as f:
        for result in results:
            if result:
                f.write(json.dumps(result) + "\n")

    logger.info(f"Saved {len(results)} results to {output_file}")

    # Print summary
    successful = sum(1 for r in results if r and r.get("reward", 0) == 1)
    total = len(results)
    logger.info(f"Success rate: {successful}/{total} ({100*successful/total:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Run BFCL v3 benchmark with Claude")

    parser.add_argument(
        "--dataset-name",
        type=str,
        default="bfcl-multi-turn-base",
        help="Dataset name"
    )
    parser.add_argument(
        "--experiment-suffix",
        type=str,
        default="claude",
        help="Experiment suffix"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=1,
        help="Number of trials per task"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="us.anthropic.claude-sonnet-4-20250514-v1:0",
        help="Claude model ID"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/multiturn_data_base_val.jsonl",
        help="Path to BFCL data"
    )
    parser.add_argument(
        "--answer-path",
        type=str,
        default="data/possible_answer",
        help="Path to answer files"
    )
    parser.add_argument(
        "--use-memory",
        action="store_true",
        help="Use ReMe memory"
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Claude thinking mode"
    )
    parser.add_argument(
        "--memory-base-url",
        type=str,
        default="http://0.0.0.0:8002/",
        help="ReMe service URL"
    )
    parser.add_argument(
        "--memory-workspace-id",
        type=str,
        default="bfcl_v3",
        help="ReMe workspace ID"
    )
    parser.add_argument(
        "--region-name",
        type=str,
        default="us-west-2",
        help="AWS region"
    )

    args = parser.parse_args()

    run_bfcl_benchmark(
        dataset_name=args.dataset_name,
        experiment_suffix=args.experiment_suffix,
        max_workers=args.max_workers,
        num_trials=args.num_trials,
        model_id=args.model_id,
        data_path=args.data_path,
        answer_path=Path(args.answer_path),
        use_memory=args.use_memory,
        enable_thinking=args.enable_thinking,
        memory_base_url=args.memory_base_url,
        memory_workspace_id=args.memory_workspace_id,
        region_name=args.region_name,
    )


if __name__ == "__main__":
    main()
