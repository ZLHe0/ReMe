#!/usr/bin/env python3
"""
Build ReMe task memory from BFCL results JSONL.
"""

import argparse
import json
from typing import List, Dict, Any

import requests
from loguru import logger


def load_results(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_memory(
    results_path: str,
    reme_url: str,
    workspace_id: str,
    min_reward: float,
    timeout_s: int,
) -> int:
    results = load_results(results_path)
    successful = [r for r in results if r.get("reward", 0) >= min_reward]

    logger.info(f"Loaded {len(results)} results")
    logger.info(f"Using {len(successful)} trajectories with reward >= {min_reward}")

    total_memories = 0
    for result in successful:
        trajectory = {
            "task_id": result.get("task_id"),
            "messages": result.get("task_history", []),
            "score": result.get("reward", 0),
        }

        payload = {
            "trajectories": [trajectory],
            "workspace_id": workspace_id,
        }

        try:
            resp = requests.post(
                f"{reme_url.rstrip('/')}/summary_task_memory",
                json=payload,
                timeout=timeout_s,
            )
            if resp.status_code == 200:
                data = resp.json()
                memories = len(data.get("metadata", {}).get("memory_list", []))
                total_memories += memories
                logger.info(f"{result.get('task_id')}: {memories} memories created")
            else:
                logger.warning(
                    f"{result.get('task_id')}: failed ({resp.status_code}) {resp.text}"
                )
        except Exception as e:
            logger.error(f"{result.get('task_id')}: error {e}")

    logger.info(f"Total memories created: {total_memories}")
    return total_memories


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ReMe task memory from BFCL results JSONL"
    )
    parser.add_argument(
        "--results-path",
        type=str,
        required=True,
        help="Path to results JSONL (from scripts/run_bfcl.py)",
    )
    parser.add_argument(
        "--reme-url",
        type=str,
        default="http://0.0.0.0:8002/",
        help="ReMe service base URL",
    )
    parser.add_argument(
        "--workspace-id",
        type=str,
        required=True,
        help="ReMe workspace ID",
    )
    parser.add_argument(
        "--min-reward",
        type=float,
        default=1.0,
        help="Minimum reward to include trajectory (default: 1.0)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="HTTP timeout in seconds",
    )

    args = parser.parse_args()

    build_memory(
        results_path=args.results_path,
        reme_url=args.reme_url,
        workspace_id=args.workspace_id,
        min_reward=args.min_reward,
        timeout_s=args.timeout,
    )


if __name__ == "__main__":
    main()
