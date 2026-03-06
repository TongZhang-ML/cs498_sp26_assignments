#!/usr/bin/env python3
"""
Generate local train/test preference datasets for RLHF assignments.

Run this script from the repository root:
- python gendata.py

By default it writes plain-text files into ./data:
- train_prefs.txt
- test_prefs.txt
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile

from datasets import Dataset, load_dataset


def _write_jsonl_txt(ds: Dataset, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def main() -> None:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    default_output_dir = os.path.join(repo_root, "data")
    parser = argparse.ArgumentParser(description="Generate local train/test data as plain text files.")
    parser.add_argument(
        "--output_dir",
        default=default_output_dir,
        help="Directory to save generated text files. Default: ./data",
    )
    parser.add_argument("--train_size", type=int, default=2500, help="Number of train examples to keep.")
    parser.add_argument(
        "--dataset_name",
        default="HuggingFaceH4/ultrafeedback_binarized",
        help="HF dataset repo id.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hf_cache_") as tmp_cache:
        dataset = load_dataset(args.dataset_name, cache_dir=tmp_cache)

    train_full = dataset["train_prefs"]
    test = dataset["test_prefs"]
    train_size = min(args.train_size, len(train_full))
    train = train_full.select(range(train_size))

    train_txt = os.path.join(args.output_dir, "train_prefs.txt")
    test_txt = os.path.join(args.output_dir, "test_prefs.txt")
    _write_jsonl_txt(train, train_txt)
    _write_jsonl_txt(test, test_txt)

    print("Saved dataset to:")
    print(f"- {train_txt}")
    print(f"- {test_txt}")
    print(f"train={len(train)} test={len(test)}")


if __name__ == "__main__":
    main()
