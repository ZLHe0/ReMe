# flake8: noqa: E402
import argparse
import os
import time

import ray
import requests
from ray import logger

os.environ["APPWORLD_ROOT"] = os.environ.get("APPWORLD_ROOT", "/work")
from dotenv import load_dotenv

load_dotenv("../../.env")

import json
from pathlib import Path

from appworld import load_task_ids

from appworld_react_agent import AppworldReactAgent


def handle_api_response(response: requests.Response):
    """Handle API response with proper error checking"""
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

    return response.json()


def delete_workspace(workspace_id: str, api_url: str = "http://0.0.0.0:8002/"):
    """Delete the current workspace from the vector store"""
    response = requests.post(
        url=f"{api_url}vector_store",
        json={
            "workspace_id": workspace_id,
            "action": "delete",
        },
    )

    result = handle_api_response(response)
    if result:
        print(f"Workspace '{workspace_id}' deleted successfully")


def dump_memory(workspace_id: str, path: str = "./", api_url: str = "http://0.0.0.0:8002/"):
    """Dump the vector store memories to disk"""
    response = requests.post(
        url=f"{api_url}vector_store",
        json={
            "workspace_id": workspace_id,
            "action": "dump",
            "path": path,
        },
    )

    result = handle_api_response(response)
    if result:
        print(f"Memory dumped to {path}")


def load_memory(workspace_id: str, path: str = "docs/library", api_url: str = "http://0.0.0.0:8002/"):
    """Load memories from disk into the vector store"""
    response = requests.post(
        url=f"{api_url}vector_store",
        json={
            "workspace_id": workspace_id,
            "action": "load",
            "path": path,
        },
    )

    result = handle_api_response(response)
    if result:
        print(f"Memory loaded from {path}")


def run_agent(
    model_name: str,
    base_url: str | None,
    api_key: str | None,
    dataset_name: str,
    experiment_suffix: str,
    max_workers: int,
    num_trials: int = 1,
    use_memory: bool = False,
    use_memory_addition: bool = False,
    use_memory_deletion: bool = False,
    delete_freq: int = 10,
    freq_threshold: int = 5,
    utility_threshold: float = 0.5,
    workspace_id: str = "appworld_v1",
    api_url: str = "http://0.0.0.0:8002/",
    batch_size: int = 4,
):
    experiment_name = dataset_name + "_" + experiment_suffix
    path: Path = Path(f"./exp_result/{model_name}")
    path.mkdir(parents=True, exist_ok=True)

    task_ids = load_task_ids(dataset_name)
    result: list = []

    def dump_file():
        with open(path / f"{experiment_name}.jsonl", "a") as f:
            for x in result:
                f.write(json.dumps(x) + "\n")

    if max_workers > 1:
        # Process tasks in batches
        total_tasks = len(task_ids)
        num_batches = (total_tasks + batch_size - 1) // batch_size  # Ceiling division

        logger.info(f"Total tasks: {total_tasks}, Batch size: {batch_size}, Number of batches: {num_batches}")

        for batch_idx in range(num_batches):
            # Initialize Ray for this batch
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_tasks)
            batch_task_ids = task_ids[start_idx:end_idx]

            logger.info(f"Starting batch {batch_idx + 1}/{num_batches} with {len(batch_task_ids)} tasks")

            # Initialize Ray with the number of CPUs needed for this batch
            ray.init(
                num_cpus=len(batch_task_ids),
                runtime_env={"env_vars": {"APPWORLD_ROOT": os.environ.get("APPWORLD_ROOT", "/work")}}
            )

            future_list: list = []
            for i, task_id in enumerate(batch_task_ids):
                actor = AppworldReactAgent.remote(
                    index=start_idx+i,
                    model_name=model_name,
                    base_url=base_url,
                    api_key=api_key,
                    task_ids=[task_id],
                    experiment_name=experiment_name,
                    num_trials=num_trials,
                    use_memory=use_memory,
                    use_memory_addition=use_memory_addition,
                    use_memory_deletion=use_memory_deletion,
                    delete_freq=delete_freq,
                    freq_threshold=freq_threshold,
                    utility_threshold=utility_threshold,
                    memory_workspace_id=workspace_id,
                    memory_base_url=api_url,
                )
                future = actor.execute.remote()
                future_list.append(future)
                time.sleep(1)

            logger.info(f"Batch {batch_idx + 1} submit complete, waiting for results...")

            # Collect results from this batch
            for i, (task_id, future) in enumerate(zip(batch_task_ids, future_list)):
                try:
                    t_result = ray.get(future)
                    if t_result:
                        if isinstance(t_result, list):
                            result.extend(t_result)
                        else:
                            result.append(t_result)
                except Exception as e:
                    logger.exception(f"run ray error with task_id={task_id}")

                logger.info(f"Batch {batch_idx + 1}: task {i + 1}/{len(batch_task_ids)} complete")

            # Shutdown Ray to free resources before next batch
            ray.shutdown()
            logger.info(f"Batch {batch_idx + 1}/{num_batches} complete, Ray resources released")

            # Optional: small delay between batches
            if batch_idx < num_batches - 1:
                time.sleep(2)

        dump_file()

    else:
        # Single worker mode - still use Ray but process sequentially
        ray.init(
            num_cpus=1,
            runtime_env={"env_vars": {"APPWORLD_ROOT": os.environ.get("APPWORLD_ROOT", "/work")}}
        )
        for index, task_id in enumerate(task_ids):
            agent = AppworldReactAgent.remote(
                index=index,
                model_name=model_name,
                base_url=base_url,
                api_key=api_key,
                task_ids=[task_id],
                experiment_name=experiment_name,
                num_trials=num_trials,
                use_memory=use_memory,
                use_memory_addition=use_memory_addition,
                use_memory_deletion=use_memory_deletion,
                delete_freq=delete_freq,
                freq_threshold=freq_threshold,
                utility_threshold=utility_threshold,
                memory_workspace_id=workspace_id,
                memory_base_url=api_url,
            )
            try:
                task_results = ray.get(agent.execute.remote())
                if isinstance(task_results, list):
                    result.extend(task_results)
                else:
                    result.append(task_results)
            except Exception as e:
                logger.exception(f"run ray error with task_id={task_id}")
        ray.shutdown()
        dump_file()

def main():
    parser = argparse.ArgumentParser(description="Run AppWorld with optional ReMe memory.")
    parser.add_argument("--model-name", type=str, default="qwen3-8b")
    parser.add_argument("--base-url", type=str, default=None, help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI-compatible API key")
    parser.add_argument("--dataset-name", type=str, default="test_normal")
    parser.add_argument("--experiment-suffix", type=str, default="with-memory")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--num-trials", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--use-memory", action="store_true")
    parser.add_argument("--use-memory-addition", action="store_true")
    parser.add_argument("--use-memory-deletion", action="store_true")
    parser.add_argument("--delete-freq", type=int, default=5)
    parser.add_argument("--freq-threshold", type=int, default=5)
    parser.add_argument("--utility-threshold", type=float, default=0.5)
    parser.add_argument("--workspace-id", type=str, default="appworld")
    parser.add_argument("--api-url", type=str, default="http://0.0.0.0:8002/")
    parser.add_argument("--reset-workspace", action="store_true")
    parser.add_argument("--load-memory", action="store_true")
    parser.add_argument("--memory-path", type=str, default="docs/library")
    args = parser.parse_args()

    if args.reset_workspace:
        logger.info("Deleting workspace...")
        delete_workspace(workspace_id=args.workspace_id, api_url=args.api_url)
        time.sleep(5)

    if args.load_memory:
        logger.info("Loading memory library...")
        load_memory(workspace_id=args.workspace_id, api_url=args.api_url, path=args.memory_path)

    run_agent(
        model_name=args.model_name,
        base_url=args.base_url,
        api_key=args.api_key,
        dataset_name=args.dataset_name,
        experiment_suffix=args.experiment_suffix,
        max_workers=args.max_workers,
        num_trials=args.num_trials,
        use_memory=args.use_memory,
        use_memory_addition=args.use_memory_addition,
        use_memory_deletion=args.use_memory_deletion,
        delete_freq=args.delete_freq,
        freq_threshold=args.freq_threshold,
        utility_threshold=args.utility_threshold,
        workspace_id=args.workspace_id,
        api_url=args.api_url,
        batch_size=args.batch_size,
    )

if __name__ == "__main__":
    main()
