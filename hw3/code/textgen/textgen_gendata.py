#!/usr/bin/env python3
"""
textgen_gendata.py

Create a text-generation dataset split under ./data/:

  ./data/train.txt
  ./data/val.txt

If ./data/fables.txt does not exist, this script automatically downloads
"Aesop's Fables" from Project Gutenberg and saves it to ./data/fables.txt.

Then it:
- Reads all non-empty lines
- Shuffles with a seed
- Splits into train/val
- Writes train.txt and val.txt

Students do NOT implement anything in this file.
"""

from __future__ import annotations

import argparse
import os
import random
import urllib.request


GUTENBERG_URL = "https://www.gutenberg.org/files/21/21-0.txt"


def download_fables(path: str) -> None:
    print("Downloading Aesop's Fables from Project Gutenberg...")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    urllib.request.urlretrieve(GUTENBERG_URL, path)
    print(f"Saved to {path}")


def read_clean_lines(path: str):
    lines = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            if s:
                lines.append(s)
    return lines


def write_lines(path: str, lines):
    with open(path, "w", encoding="utf-8") as f:
        for s in lines:
            f.write(s + "\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_path", type=str, default="./data/fables.txt")
    p.add_argument("--out_dir", type=str, default="./data")
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.input_path):
        download_fables(args.input_path)

    lines = read_clean_lines(args.input_path)
    if len(lines) < 10:
        raise ValueError("Downloaded corpus seems too small.")

    print(f"Loaded {len(lines)} non-empty lines.")

    rnd = random.Random(args.seed)
    rnd.shuffle(lines)

    n_val = max(1, int(round(len(lines) * args.val_fraction)))
    n_val = min(n_val, len(lines) - 1)

    val_lines = lines[:n_val]
    train_lines = lines[n_val:]

    train_path = os.path.join(args.out_dir, "train.txt")
    val_path = os.path.join(args.out_dir, "val.txt")

    write_lines(train_path, train_lines)
    write_lines(val_path, val_lines)

    print(f"Wrote train: {train_path} ({len(train_lines)} lines)")
    print(f"Wrote val:   {val_path} ({len(val_lines)} lines)")


if __name__ == "__main__":
    main()
