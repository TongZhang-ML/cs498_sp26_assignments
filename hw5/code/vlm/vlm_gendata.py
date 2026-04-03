#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw


COLORS = ["red", "blue", "green", "yellow"]
SHAPES = ["square", "circle", "triangle", "diamond"]
TRAIN_SCENES = 100
VAL_SCENES = 20
TEST_SCENES = 20
IMAGE_SIZE = 224


def make_scene(rng: random.Random, split: str, idx: int) -> Dict[str, object]:
    return {
        "scene_id": f"{split}_{idx}",
        "seed": rng.randint(0, 10**9),
        "left_color": rng.choice(COLORS),
        "left_shape": rng.choice(SHAPES),
        "right_color": rng.choice(COLORS),
        "right_shape": rng.choice(SHAPES),
    }


def scene_questions(scene: Dict[str, object]) -> List[Tuple[str, str, str]]:
    return [
        ("left_color", "what color is the object on the left ?", str(scene["left_color"])),
        ("right_color", "what color is the object on the right ?", str(scene["right_color"])),
        ("same_color", "do the two objects have the same color ?", "yes" if scene["left_color"] == scene["right_color"] else "no"),
        ("left_is_red", "is the object on the left red ?", "yes" if scene["left_color"] == "red" else "no"),
    ]


def write_jsonl(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _jittered_box(
    rng: random.Random,
    center_x: int,
    center_y: int,
    base_half_w: int,
    base_half_h: int,
    image_size: int,
) -> Tuple[int, int, int, int]:
    half_w = base_half_w + rng.randint(-4, 4)
    half_h = base_half_h + rng.randint(-8, 8)
    dx = rng.randint(-6, 6)
    dy = rng.randint(-6, 6)
    cx = max(24 + half_w, min(image_size - 24 - half_w, center_x + dx))
    cy = max(24 + half_h, min(image_size - 24 - half_h, center_y + dy))
    return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)


def _draw_shape(draw: ImageDraw.ImageDraw, shape: str, box: Tuple[int, int, int, int], fill: Tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = box
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    if shape == "square":
        draw.rectangle(box, fill=fill, outline=(40, 40, 40))
    elif shape == "circle":
        draw.ellipse(box, fill=fill, outline=(40, 40, 40))
    elif shape == "triangle":
        draw.polygon([(cx, y0), (x0, y1), (x1, y1)], fill=fill, outline=(40, 40, 40))
    elif shape == "diamond":
        draw.polygon([(cx, y0), (x0, cy), (cx, y1), (x1, cy)], fill=fill, outline=(40, 40, 40))


def render_scene_image(scene: Dict[str, object], image_size: int = IMAGE_SIZE) -> Image.Image:
    color_map = {
        "red": (230, 46, 38),
        "blue": (38, 89, 230),
        "green": (38, 179, 64),
        "yellow": (242, 204, 38),
    }
    rng = random.Random(int(scene["seed"]))
    bg = 236 + rng.randint(-6, 6)
    panel = 244 + rng.randint(-5, 5)
    outline = 206 + rng.randint(-6, 6)
    image = Image.new("RGB", (image_size, image_size), (bg, bg, bg))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((8, 8, image_size - 8, image_size - 8), radius=12, fill=(panel, panel, panel), outline=(outline, outline, outline))
    left_box = _jittered_box(rng, 56, 110, 30, 55, image_size)
    right_box = _jittered_box(rng, image_size - 56, 110, 30, 55, image_size)
    _draw_shape(draw, str(scene["left_shape"]), left_box, color_map[str(scene["left_color"])])
    _draw_shape(draw, str(scene["right_shape"]), right_box, color_map[str(scene["right_color"])])
    return image


def build_rows(split: str, num_scenes: int, seed: int, image_root: str) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    os.makedirs(os.path.join(image_root, split), exist_ok=True)
    rows: List[Dict[str, object]] = []
    for idx in range(num_scenes):
        scene = make_scene(rng, split, idx)
        image_rel_path = os.path.join("images", split, f"{scene['scene_id']}.png")
        image_abs_path = os.path.join(image_root, split, f"{scene['scene_id']}.png")
        render_scene_image(scene).save(image_abs_path)
        for question_type, question, answer in scene_questions(scene):
            rows.append(
                {
                    "id": f"{scene['scene_id']}_{question_type}",
                    "scene_id": scene["scene_id"],
                    "seed": scene["seed"],
                    "left_color": scene["left_color"],
                    "left_shape": scene["left_shape"],
                    "right_color": scene["right_color"],
                    "right_shape": scene["right_shape"],
                    "question_type": question_type,
                    "question": question,
                    "answer": answer,
                    "image_path": image_rel_path,
                }
            )
    rng.shuffle(rows)
    return rows


def main() -> None:
    root = os.path.dirname(__file__)
    data_root = os.path.join(root, "data")
    image_root = os.path.join(data_root, "images")
    train_rows = build_rows("train", TRAIN_SCENES, 5, image_root)
    val_rows = build_rows("val", VAL_SCENES, 9, image_root)
    test_rows = build_rows("test", TEST_SCENES, 17, image_root)

    print(f"writing data/train.jsonl from {TRAIN_SCENES} scenes -> {len(train_rows)} questions")
    write_jsonl(os.path.join(data_root, "train.jsonl"), train_rows)
    print(f"writing data/val.jsonl from {VAL_SCENES} scenes -> {len(val_rows)} questions")
    write_jsonl(os.path.join(data_root, "val.jsonl"), val_rows)
    print(f"writing data/test.jsonl from {TEST_SCENES} scenes -> {len(test_rows)} questions")
    write_jsonl(os.path.join(data_root, "test.jsonl"), test_rows)
    print("wrote train/val/test jsonl and image files")


if __name__ == "__main__":
    main()
