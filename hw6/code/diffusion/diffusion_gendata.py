#!/usr/bin/env python3
"""
Generate visible training data for the Homework 6 diffusion problem.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from os.path import expanduser
from typing import Any, Dict, List, Sequence, Tuple
from urllib.request import urlretrieve

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from PIL import Image
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]

_DIFFUSERS_IMPORT_ERROR: Exception | None = None
try:
    from diffusers import StableDiffusionPipeline
except Exception as e:
    _DIFFUSERS_IMPORT_ERROR = e
    StableDiffusionPipeline = None  # type: ignore[assignment]

try:
    import open_clip
except ImportError:
    open_clip = None  # type: ignore[assignment]


COLORS = [
    "red",
    "blue",
    "green",
    "yellow",
    "white",
    "black",
    "brown",
    "silver",
    "golden",
    "orange",
]
OBJECTS = [
    "bicycle",
    "school bus",
    "rabbit",
    "turtle",
    "sailing boat",
    "steam locomotive",
    "strawberry bowl",
    "wooden chair",
    "teapot",
    "teddy bear",
]
TRAIN_PROMPTS = [f"a {color} {obj}" for color in COLORS for obj in OBJECTS]
EVAL_PROMPTS = [
    "a red bicycle parked by a fence",
    "a yellow school bus on a street",
    "a fluffy rabbit in grass",
    "a green turtle on a rock",
    "a small sailing boat on a lake",
    "a steam locomotive at a station",
    "a bowl of ripe strawberries on a table",
    "a wooden chair by a window",
    "a white teapot on a shelf",
    "a brown teddy bear on a bed",
]
CANDIDATES_PER_EXAMPLE = 20
SD_MODEL_NAME = "runwayml/stable-diffusion-v1-5"
OPENCLIP_MODEL_NAME = "ViT-B-32"
OPENCLIP_PRETRAINED = "openai"
AESTHETIC_MODEL_NAME = "vit_b_32"
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_IMAGE_SIZE = 256
MIN_INFERENCE_STEPS = 20


def make_example(example_id: str, prompt: str, image_path: str) -> Dict[str, object]:
    return {
        "id": example_id,
        "prompt": prompt,
        "image_path": image_path,
    }


def write_jsonl(path: str, rows: Sequence[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _require_runtime() -> None:
    missing: List[str] = []
    if torch is None:
        missing.append("torch")
    if Image is None:
        missing.append("pillow")
    if StableDiffusionPipeline is None:
        missing.append("diffusers")
    if open_clip is None:
        missing.append("open_clip_torch")
    if nn is None:
        missing.append("torch.nn")
    if missing:
        pkg_list = ", ".join(missing)
        extra = ""
        if "diffusers" in missing and _DIFFUSERS_IMPORT_ERROR is not None:
            extra = (
                " Underlying diffusers import error:"
                f" {type(_DIFFUSERS_IMPORT_ERROR).__name__}: {_DIFFUSERS_IMPORT_ERROR}"
            )
        raise RuntimeError(
            f"Missing required packages: {pkg_list}. "
            "Install the diffusion dependencies first, for example: "
            "`pip install -r ../requirements.txt` from the code/diffusion directory."
            + extra
        )


def _generation_device(device: torch.device) -> str:
    return "cuda" if device.type == "cuda" else "cpu"


def _get_aesthetic_model(clip_model: str = AESTHETIC_MODEL_NAME) -> Any:
    cache_folder = os.path.join(expanduser("~"), ".cache", "emb_reader")
    path_to_model = os.path.join(cache_folder, f"sa_0_4_{clip_model}_linear.pth")
    if not os.path.exists(path_to_model):
        os.makedirs(cache_folder, exist_ok=True)
        url_model = f"https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_{clip_model}_linear.pth?raw=true"
        urlretrieve(url_model, path_to_model)
    if clip_model == "vit_b_32":
        model = nn.Linear(512, 1)
    else:
        raise ValueError(f"unsupported aesthetic clip model: {clip_model}")
    state_dict = torch.load(path_to_model, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_pipeline_and_reward_model(
    sd_model_name: str,
    clip_model_name: str,
    device: torch.device,
) -> Tuple[Any, Any, Any]:
    _require_runtime()
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        sd_model_name,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=True,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    if ":" in clip_model_name:
        openclip_model_name, openclip_pretrained = clip_model_name.split(":", 1)
    else:
        openclip_model_name, openclip_pretrained = OPENCLIP_MODEL_NAME, OPENCLIP_PRETRAINED
    reward_model, _, reward_processor = open_clip.create_model_and_transforms(
        model_name=openclip_model_name,
        pretrained=openclip_pretrained,
        device=device,
    )
    aesthetic_model = _get_aesthetic_model(AESTHETIC_MODEL_NAME).to(device)
    reward_model.eval()
    aesthetic_model.eval()
    return pipe, reward_processor, (reward_model, aesthetic_model)


@torch.no_grad()
def generate_images(
    pipe: Any,
    prompt: str,
    seeds: Sequence[int],
    device: torch.device,
    num_inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
) -> List[Image.Image]:
    step_count = max(int(num_inference_steps), MIN_INFERENCE_STEPS)
    generators = [torch.Generator(device=_generation_device(device)).manual_seed(int(seed)) for seed in seeds]
    result = pipe(
        prompt=[prompt] * len(seeds),
        generator=generators,
        num_inference_steps=step_count,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
    )
    return list(result.images)


@torch.no_grad()
def score_images(
    reward_processor: Any,
    reward_model: Any,
    images: Sequence[Image.Image],
    device: torch.device,
) -> torch.Tensor:
    clip_model, aesthetic_model = reward_model
    pixel_values = torch.stack([reward_processor(image.convert("RGB")) for image in images], dim=0).to(device)
    image_features = clip_model.encode_image(pixel_values)
    image_features = F.normalize(image_features, dim=-1)
    return aesthetic_model(image_features).squeeze(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=os.path.join(os.path.dirname(__file__), "data"))
    parser.add_argument("--sd_model_name", default=SD_MODEL_NAME)
    parser.add_argument("--clip_model_name", default=f"{OPENCLIP_MODEL_NAME}:{OPENCLIP_PRETRAINED}")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=DEFAULT_GUIDANCE_SCALE)
    parser.add_argument("--height", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--width", type=int, default=DEFAULT_IMAGE_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _require_runtime()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = args.data_dir
    image_dir = os.path.join(base, "finetune_images")
    os.makedirs(image_dir, exist_ok=True)
    pipe, reward_processor, reward_model = build_pipeline_and_reward_model(
        args.sd_model_name,
        args.clip_model_name,
        device,
    )

    train_rows: List[Dict[str, object]] = []
    for prompt_idx, prompt in enumerate(TRAIN_PROMPTS):
        base_seed = 1000 + 20 * prompt_idx
        candidate_seeds = [base_seed + offset for offset in range(CANDIDATES_PER_EXAMPLE)]
        images = generate_images(
            pipe=pipe,
            prompt=prompt,
            seeds=candidate_seeds,
            device=device,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
        )
        scores = score_images(reward_processor, reward_model, images, device)
        best_index = int(torch.argmax(scores).item())
        example_id = f"train_{prompt_idx:03d}"
        image_relpath = os.path.join("finetune_images", f"{example_id}.jpg")
        images[best_index].save(
            os.path.join(base, image_relpath),
            format="JPEG",
            quality=90,
            optimize=True,
        )
        train_rows.append(make_example(example_id, prompt, image_relpath))
        print(f"gendata-example: id={example_id} prompt={prompt} image_path={image_relpath}")

    write_jsonl(os.path.join(base, "train.jsonl"), train_rows)


if __name__ == "__main__":
    main()
