#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List


CLASS_SPECS = [
    ("red square", "red", "square"),
    ("blue circle", "blue", "circle"),
    ("green triangle", "green", "triangle"),
    ("yellow diamond", "yellow", "diamond"),
    ("purple plus", "purple", "plus"),
    ("orange x", "orange", "x"),
]

CAPTION_TEMPLATES = [
    "a photo of a {name}",
    "an image of a {name}",
    "a centered {name}",
    "the picture shows a {name}",
]


@dataclass(frozen=True)
class SplitSpec:
    name: str
    per_class: int
    seed: int


def make_record(class_index: int, example_index: int, split: str, rng: random.Random) -> Dict[str, object]:
    class_name, color, shape = CLASS_SPECS[class_index]
    caption = CAPTION_TEMPLATES[(example_index + class_index) % len(CAPTION_TEMPLATES)].format(name=class_name)
    return {
        "id": f"{split}_{class_index}_{example_index}",
        "split": split,
        "label": class_index,
        "class_name": class_name,
        "color": color,
        "shape": shape,
        "seed": rng.randint(0, 10**9),
        "caption": caption,
    }


def write_jsonl(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def build_split(spec: SplitSpec) -> List[Dict[str, object]]:
    rng = random.Random(spec.seed)
    rows: List[Dict[str, object]] = []
    for class_index in range(len(CLASS_SPECS)):
        for i in range(spec.per_class):
            rows.append(make_record(class_index, i, spec.name, rng))
    rng.shuffle(rows)
    return rows


def main() -> None:
    root = os.path.join(os.path.dirname(__file__), "data")
    train = build_split(SplitSpec("train", per_class=10, seed=21))
    val = build_split(SplitSpec("val", per_class=10, seed=34))
    test = build_split(SplitSpec("test", per_class=50, seed=55))

    write_jsonl(os.path.join(root, "train.jsonl"), train)
    write_jsonl(os.path.join(root, "val.jsonl"), val)
    write_jsonl(os.path.join(root, "test.jsonl"), test)

    print(f"wrote {len(train)} train examples")
    print(f"wrote {len(val)} val examples")
    print(f"wrote {len(test)} test examples")


if __name__ == "__main__":
    main()
