#!/usr/bin/env python3
"""
Generate a lightweight synthetic controllable-generation dataset for Homework 6.
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List


ATTRIBUTES = [
    {"id": 0, "name": "red"},
    {"id": 1, "name": "green"},
    {"id": 2, "name": "blue"},
    {"id": 3, "name": "striped"},
]


def make_example(example_id: str, seed: int, attr_id: int) -> Dict[str, object]:
    return {
        "id": example_id,
        "seed": seed,
        "target_attr": int(attr_id),
        "target_name": ATTRIBUTES[attr_id]["name"],
    }


def make_split(n: int, offset: int) -> List[Dict[str, object]]:
    rng = random.Random(1000 + offset)
    rows: List[Dict[str, object]] = []
    for i in range(n):
        attr_id = rng.randrange(len(ATTRIBUTES))
        seed = 100000 + offset * 1000 + i
        rows.append(make_example(f"gan_{offset}_{i:04d}", seed, attr_id))
    return rows


def write_jsonl(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    base = os.path.join(os.path.dirname(__file__), "data")
    write_jsonl(os.path.join(base, "train.jsonl"), make_split(160, 0))
    write_jsonl(os.path.join(base, "val.jsonl"), make_split(48, 1))
    write_jsonl(os.path.join(base, "test.jsonl"), make_split(48, 2))


if __name__ == "__main__":
    main()
