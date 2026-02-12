#!/usr/bin/env python3
"""
Summarize BFCL results and compare baseline vs memory runs.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any


def load_results(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def count_success(results: List[Dict[str, Any]]) -> int:
    return sum(1 for r in results if r.get("reward", 0) == 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BFCL run results")
    parser.add_argument("--train", type=str, required=True, help="Train results JSONL")
    parser.add_argument("--baseline", type=str, required=True, help="Baseline JSONL")
    parser.add_argument("--memory", type=str, required=True, help="Memory JSONL")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON summary output path",
    )
    args = parser.parse_args()

    train = load_results(Path(args.train))
    baseline = load_results(Path(args.baseline))
    memory = load_results(Path(args.memory))

    train_success = count_success(train)
    baseline_success = count_success(baseline)
    memory_success = count_success(memory)

    summary = {
        "training": {"total": len(train), "successful": train_success},
        "baseline": {"total": len(baseline), "successful": baseline_success},
        "with_memory": {"total": len(memory), "successful": memory_success},
        "improvement": memory_success - baseline_success,
    }

    print("=" * 60)
    print("BFCL v3 EVALUATION RESULTS")
    print("=" * 60)
    print()
    print("Training Data:")
    print(f"  Total tasks:     {len(train)}")
    if train:
        print(f"  Successful:      {train_success}")
        print(f"  Success rate:    {100*train_success/len(train):.1f}%")
    print()
    print("Validation Data:")
    print(f"  Total tasks:     {len(baseline)}")
    print()
    print("  Baseline (no memory):")
    if baseline:
        print(f"    Successful:    {baseline_success}")
        print(f"    Success rate:  {100*baseline_success/len(baseline):.1f}%")
    print()
    print("  With Memory:")
    if memory:
        print(f"    Successful:    {memory_success}")
        print(f"    Success rate:  {100*memory_success/len(memory):.1f}%")
    print()
    if baseline and memory:
        improvement = memory_success - baseline_success
        print(f"  Improvement:     {improvement:+d} tasks ({100*improvement/len(baseline):+.1f}%)")
    print()
    print("=" * 60)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {out_path}")


if __name__ == "__main__":
    main()
