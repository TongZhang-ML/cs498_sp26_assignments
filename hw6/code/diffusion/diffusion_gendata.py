#!/usr/bin/env python3
"""
Generate the visible prompt splits for the Homework 6 diffusion problem.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List


TRAIN_PROMPTS = [
    "a watercolor painting of a flying dog",
    "an impressionist painting of a flying cat",
    "a pastel painting of a flying horse",
    "a soft landscape painting with a flying sheep",
    "a dreamy artwork of a rabbit in flight",
    "a colorful painting of a flying cow",
    "a gentle brushstroke painting of a flying fox",
    "a sky scene with a bird flying above clouds",
    "an oil painting of a swan in flight",
    "a bright canvas showing a flying deer",
    "a French impressionist image of a flying elephant",
    "a lily pond painting with a duck in the air",
    "a storybook painting of a flying pig",
    "a Monet-inspired scene with a flying dog",
    "an impressionist artwork of a cat above the clouds",
    "a sunrise painting of a horse flying through the sky",
]

VAL_PROMPT = "an impressionist Monet style painting of a pig flying through the sky"
TEST_PROMPT = "a Monet style painting of a pig flying over bright clouds at sunrise"


def make_example(example_id: str, prompt: str, seed: int) -> Dict[str, object]:
    return {
        "id": example_id,
        "prompt": prompt,
        "seed": int(seed),
    }


def write_jsonl(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    base = os.path.join(os.path.dirname(__file__), "data")
    train_rows = [make_example(f"train_{idx:02d}", prompt, 1000 + 20 * idx) for idx, prompt in enumerate(TRAIN_PROMPTS)]
    val_rows = [make_example("val_00", VAL_PROMPT, 5000)]
    test_rows = [make_example("test_00", TEST_PROMPT, 7000)]
    write_jsonl(os.path.join(base, "train.jsonl"), train_rows)
    write_jsonl(os.path.join(base, "val.jsonl"), val_rows)
    write_jsonl(os.path.join(base, "test.jsonl"), test_rows)


if __name__ == "__main__":
    main()
