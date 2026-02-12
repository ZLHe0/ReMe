#!/usr/bin/env python3
"""
Setup script for BFCL v3 data.
This script helps download and prepare the BFCL v3 dataset.
"""

import os
import json
import random
import subprocess
from pathlib import Path
from loguru import logger


def clone_gorilla_repo(target_dir: Path = Path("gorilla")):
    """Clone the Gorilla repository containing BFCL."""
    if target_dir.exists():
        logger.info(f"Gorilla repo already exists at {target_dir}")
        return target_dir

    logger.info("Cloning Gorilla repository...")
    subprocess.run(
        ["git", "clone", "https://github.com/ShishirPatil/gorilla.git", str(target_dir)],
        check=True
    )
    return target_dir


def copy_bfcl_data(gorilla_dir: Path, data_dir: Path = Path("data")):
    """
    Copy BFCL data from Gorilla repo to data directory.

    Args:
        gorilla_dir: Path to cloned Gorilla repo
        data_dir: Target data directory
    """
    bfcl_data_src = gorilla_dir / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data"

    if not bfcl_data_src.exists():
        logger.error(f"BFCL data not found at {bfcl_data_src}")
        return False

    data_dir.mkdir(parents=True, exist_ok=True)

    # Copy multi-turn base data (BFCL v3)
    multi_turn_src = bfcl_data_src / "multi_turn_base.json"
    if multi_turn_src.exists():
        # Convert to JSONL format
        with open(multi_turn_src, "r") as f:
            data = json.load(f)

        output_file = data_dir / "multiturn_data_base.jsonl"
        with open(output_file, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        logger.info(f"Converted {len(data)} items to {output_file}")
    else:
        # Try JSONL format
        jsonl_src = bfcl_data_src / "multi_turn_base.jsonl"
        if jsonl_src.exists():
            import shutil
            shutil.copy(jsonl_src, data_dir / "multiturn_data_base.jsonl")
            logger.info(f"Copied {jsonl_src} to data directory")

    # Copy possible answers
    possible_answer_src = bfcl_data_src / "possible_answer"
    if possible_answer_src.exists():
        import shutil
        dst = data_dir / "possible_answer"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(possible_answer_src, dst)
        logger.info(f"Copied possible_answer directory")

    return True


def split_train_val(
    data_file: Path,
    train_ratio: float = 0.25,
    seed: int = 42
):
    """
    Split BFCL data into train and validation sets.

    BFCL v3 is the multi_turn_base category. Following ReMe paper:
    - 50 samples for training (to build experience pool)
    - 150 samples for validation (to test)

    Args:
        data_file: Path to full JSONL data file
        train_ratio: Ratio for training set (default 0.25 = 50/200)
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    # Load data
    with open(data_file, "r") as f:
        data = [json.loads(line) for line in f]

    logger.info(f"Loaded {len(data)} samples from {data_file}")

    # Shuffle and split
    random.shuffle(data)

    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    val_data = data[train_size:]

    # Save train set
    train_file = data_file.parent / f"{data_file.stem}_train.jsonl"
    with open(train_file, "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")
    logger.info(f"Saved {len(train_data)} training samples to {train_file}")

    # Save validation set
    val_file = data_file.parent / f"{data_file.stem}_val.jsonl"
    with open(val_file, "w") as f:
        for item in val_data:
            f.write(json.dumps(item) + "\n")
    logger.info(f"Saved {len(val_data)} validation samples to {val_file}")

    return train_file, val_file


def list_bfcl_categories(gorilla_dir: Path):
    """
    List available BFCL categories (for reference).

    BFCL v3 Categories:
    - multi_turn_base: Multi-turn function calling (this is BFCL v3 main)
    - multi_turn_long_context: Multi-turn with long context
    - multi_turn_composite: Multi-turn composite tasks

    BFCL v4 added:
    - multi_turn_miss_param: Missing parameter scenarios
    - multi_turn_miss_func: Missing function scenarios
    """
    bfcl_data_src = gorilla_dir / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data"

    logger.info("\nBFCL Categories found:")
    for f in sorted(bfcl_data_src.glob("*.json*")):
        logger.info(f"  - {f.stem}")

    logger.info("\n=== BFCL v3 vs v4 Notes ===")
    logger.info("BFCL v3 (multi_turn_base):")
    logger.info("  - Standard multi-turn function calling benchmark")
    logger.info("  - 200 test cases")
    logger.info("  - Used in ReMe paper: 50 train / 150 val split")

    logger.info("\nBFCL v4 additions:")
    logger.info("  - multi_turn_miss_param: Tests handling of missing parameters")
    logger.info("  - multi_turn_miss_func: Tests handling of missing functions")
    logger.info("  - If using v4 data, select only multi_turn_base for v3 compatibility")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Setup BFCL v3 data")
    parser.add_argument(
        "--gorilla-dir",
        type=str,
        default="gorilla",
        help="Directory to clone/use Gorilla repo"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory for BFCL data"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.25,
        help="Ratio for training set (default 0.25 = 50/200)"
    )
    parser.add_argument(
        "--skip-clone",
        action="store_true",
        help="Skip cloning Gorilla repo"
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="List available BFCL categories"
    )

    args = parser.parse_args()

    gorilla_dir = Path(args.gorilla_dir)
    data_dir = Path(args.data_dir)

    # Clone Gorilla repo
    if not args.skip_clone:
        clone_gorilla_repo(gorilla_dir)

    # List categories if requested
    if args.list_categories:
        list_bfcl_categories(gorilla_dir)
        return

    # Copy BFCL data
    if copy_bfcl_data(gorilla_dir, data_dir):
        # Split into train/val
        data_file = data_dir / "multiturn_data_base.jsonl"
        if data_file.exists():
            split_train_val(data_file, args.train_ratio)
        else:
            logger.warning(f"Data file not found: {data_file}")

    logger.info("\n=== Setup Complete ===")
    logger.info("Next steps:")
    logger.info("1. Install BFCL evaluation package:")
    logger.info("   cd gorilla/berkeley-function-call-leaderboard && pip install -e .")
    logger.info("2. Run benchmark:")
    logger.info("   python scripts/run_bfcl.py --data-path data/multiturn_data_base_val.jsonl")


if __name__ == "__main__":
    main()
