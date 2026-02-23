#!/usr/bin/env python
"""
Generate text files for GPT-2 sentiment classification demos.

Data source
-----------
Hugging Face dataset: amazon_polarity

Output
------
data/train.txt   (NUM_TRAIN lines)
data/val.txt     (NUM_VAL   lines)
data/test.txt    (NUM_TEST  lines)

Each line:  label <TAB> review_text
where label is 0 (negative) or 1 (positive).
"""

from __future__ import annotations

import argparse
import os
import random
import re
from typing import Tuple

import numpy as np
from datasets import load_dataset, Dataset

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

NUM_TRAIN = 500
NUM_VAL = 100
NUM_TEST = 100


def load_amazon_splits(
    n_train: int = NUM_TRAIN,
    n_val: int = NUM_VAL,
    n_test: int = NUM_TEST,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load amazon_polarity and create small train/val/test subsets.

    We sample (n_train + n_val) examples from the original training split
    and then split them into train and val.
    We sample n_test examples from the original test split.
    """
    ds = load_dataset("amazon_polarity")

    # Build train/val from the original train split
    tr_full = ds["train"].shuffle(seed=SEED).select(range(n_train + n_val))
    train_ds = tr_full.select(range(n_train))
    val_ds = tr_full.select(range(n_train, n_train + n_val))

    # Build test from the original test split
    test_ds = ds["test"].shuffle(seed=SEED).select(range(n_test))

    for split, name in ((train_ds, "Train"), (val_ds, "Val"), (test_ds, "Test")):
        pos = sum(1 for x in split if int(x["label"]) == 1)
        print(f"{name} subset: {len(split)} – {pos} pos / {len(split) - pos} neg")

    return train_ds, val_ds, test_ds


def clean(text: str) -> str:
    """Collapse newlines and excessive spaces so each example fits one row."""
    return re.sub(r"\s+", " ", text).strip()


def dump_split(split: Dataset, path: str) -> None:
    """Write one split to disk with format: label<TAB>text."""
    with open(path, "w", encoding="utf-8") as f:
        for ex in split:
            label = int(ex["label"])  # 0=neg, 1=pos
            content = clean(ex["content"])
            f.write(f"{label}\t{content}\n")


def main(out_dir: str = "./data") -> None:
    train_ds, val_ds, test_ds = load_amazon_splits()

    os.makedirs(out_dir, exist_ok=True)

    train_path = os.path.join(out_dir, "train.txt")
    val_path = os.path.join(out_dir, "val.txt")
    test_path = os.path.join(out_dir, "test.txt")

    dump_split(train_ds, train_path)
    dump_split(val_ds, val_path)
    dump_split(test_ds, test_path)

    print("\nSaved:")
    print(" ", train_path)
    print(" ", val_path)
    print(" ", test_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate txt files for sentiment classification.")
    parser.add_argument("--out_dir", default="./data", help="Directory to write the txt files.")
    parser.add_argument("--num_train", type=int, default=NUM_TRAIN)
    parser.add_argument("--num_val", type=int, default=NUM_VAL)
    parser.add_argument("--num_test", type=int, default=NUM_TEST)
    args = parser.parse_args()

    # Allow overriding counts from CLI while keeping the same data source.
    NUM_TRAIN = args.num_train
    NUM_VAL = args.num_val
    NUM_TEST = args.num_test

    main(args.out_dir)
