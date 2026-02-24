#!/usr/bin/env python3
"""
textgen_gendata.py

Create a text-generation dataset split under ./data/:

  ./data/train.txt
  ./data/val.txt

If ./data/fables.txt does not exist, this script automatically downloads
"Aesop's Fables" from Project Gutenberg and saves it to ./data/fables.txt.

Then it:
- Reads all lines in original order (keeps empty lines)
- Splits sequentially into train/val (val is the last fraction)
- Adjusts the boundary to a blank line if possible (paragraph boundary)
- Writes train.txt and val.txt preserving blank lines

Students do NOT implement anything in this file.
"""

from __future__ import annotations

import argparse
import os
import urllib.request
from typing import List


GUTENBERG_URL = "https://www.gutenberg.org/files/21/21-0.txt"


def download_fables(path: str) -> None:
    print("Downloading Aesop's Fables from Project Gutenberg...")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    urllib.request.urlretrieve(GUTENBERG_URL, path)
    print(f"Saved to {path}")


def read_lines_keep_empty(path: str) -> List[str]:
    """
    Read file in-order, preserving empty lines.
    We strip only the trailing newline; we also strip trailing spaces.
    Blank lines become "".
    """
    out: List[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            s = raw.rstrip("\n").rstrip("\r")
            s = s.rstrip()  # remove trailing spaces/tabs, but keep emptiness
            out.append(s)
    # drop trailing empty lines so val does not start with a long blank tail
    while out and out[-1] == "":
        out.pop()
    return out


def choose_blank_boundary(lines: List[str], target_idx: int, window: int = 2000) -> int:
    """
    Move target_idx to a nearby blank line boundary if possible.
    We search for an index i such that lines[i] == "" and use i+1 as start of val,
    so that val begins right after a blank line.
    """
    n = len(lines)
    if n == 0:
        return 0
    target_idx = max(1, min(target_idx, n - 1))

    lo = max(0, target_idx - window)
    hi = min(n - 1, target_idx + window)

    # Prefer blank lines at/after target (so val is close to desired size)
    for i in range(target_idx, hi + 1):
        if lines[i] == "":
            return min(i + 1, n - 1)

    # Otherwise look backward
    for i in range(target_idx, lo - 1, -1):
        if lines[i] == "":
            return min(i + 1, n - 1)

    # No blank lines nearby; use raw target
    return target_idx


def write_lines_preserve_empty(path: str, lines: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for s in lines:
            f.write(s + "\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_path", type=str, default="./data/fables.txt")
    p.add_argument("--out_dir", type=str, default="./data")
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42, help="Unused now (kept for compatibility).")
    p.add_argument("--boundary_window", type=int, default=2000, help="Search window for blank-line boundary.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.input_path):
        download_fables(args.input_path)

    lines = read_lines_keep_empty(args.input_path)
    if len(lines) < 100:
        raise ValueError("Downloaded corpus seems too small.")

    n_total = len(lines)
    n_blank = sum(1 for s in lines if s == "")
    print(f"Loaded {n_total} lines (including {n_blank} blank lines).")

    # Sequential split: val is last fraction
    raw_val_start = int(round(n_total * (1.0 - args.val_fraction)))
    raw_val_start = max(1, min(raw_val_start, n_total - 1))

    val_start = choose_blank_boundary(lines, raw_val_start, window=args.boundary_window)

    train_lines = lines[:val_start]
    val_lines = lines[val_start:]

    train_path = os.path.join(args.out_dir, "train.txt")
    val_path = os.path.join(args.out_dir, "val.txt")

    write_lines_preserve_empty(train_path, train_lines)
    write_lines_preserve_empty(val_path, val_lines)

    print(f"Wrote train: {train_path} ({len(train_lines)} lines)")
    print(f"Wrote val:   {val_path} ({len(val_lines)} lines)")
    print(f"Split index: {val_start} (requested {raw_val_start})")


if __name__ == "__main__":
    main()
    
