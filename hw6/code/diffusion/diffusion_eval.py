#!/usr/bin/env python3
"""
Score a saved diffusion generation manifest with the visible aesthetic model.
"""

from __future__ import annotations

import argparse
import json
import os
from os.path import expanduser
from typing import Any, Dict, List, Sequence, Tuple
from urllib.request import urlretrieve

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

try:
    import open_clip
except ImportError:
    open_clip = None  # type: ignore[assignment]


OPENCLIP_MODEL_NAME = "ViT-B-32"
OPENCLIP_PRETRAINED = "openai"
AESTHETIC_MODEL_NAME = "vit_b_32"


def _require_runtime() -> None:
    missing: List[str] = []
    if torch is None:
        missing.append("torch")
    if Image is None:
        missing.append("pillow")
    if open_clip is None:
        missing.append("open_clip_torch")
    if nn is None:
        missing.append("torch.nn")
    if missing:
        raise RuntimeError(
            f"Missing required packages: {', '.join(missing)}. "
            "Install the diffusion dependencies first, for example: "
            "`pip install -r ../requirements.txt` from the code/diffusion directory."
        )


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


def build_reward_model(
    clip_model_name: str,
    device: torch.device,
) -> Tuple[Any, Any]:
    _require_runtime()
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
    return reward_processor, (reward_model, aesthetic_model)


def load_manifest(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError("manifest must be a JSON list")
    return payload


def load_image_dir(path: str) -> List[Dict[str, Any]]:
    if not os.path.isdir(path):
        raise FileNotFoundError(f"image directory not found: {path}")
    names = sorted(
        name for name in os.listdir(path)
        if name.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    rows: List[Dict[str, Any]] = []
    for idx, name in enumerate(names):
        image_path = os.path.join(path, name)
        prompt = os.path.splitext(name)[0]
        rows.append(
            {
                "prompt": prompt,
                "seed": idx,
                "image_path": image_path,
            }
        )
    return rows


def load_bundle(path: str) -> List[Dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    prompts = payload["prompts"]
    seeds = payload["seeds"]
    images_uint8 = payload["images_uint8"]
    rows: List[Dict[str, Any]] = []
    for prompt, seed, image_arr in zip(prompts, seeds, images_uint8):
        image = Image.fromarray(image_arr.numpy(), mode="RGB")
        rows.append(
            {
                "prompt": str(prompt),
                "seed": int(seed),
                "image": image,
                "image_path": "",
            }
        )
    return rows


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
    parser.add_argument("--manifest_path", default="")
    parser.add_argument("--bundle_path", default="")
    parser.add_argument("--image_dir", default="outputs/finetuned_eval_images")
    parser.add_argument("--output_path", default="")
    parser.add_argument("--clip_model_name", default=f"{OPENCLIP_MODEL_NAME}:{OPENCLIP_PRETRAINED}")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    provided = [bool(args.manifest_path), bool(args.bundle_path), bool(args.image_dir)]
    if sum(provided) > 1 and (args.manifest_path or args.bundle_path):
        raise ValueError("pass only one of --manifest_path, --bundle_path, or --image_dir")
    if args.manifest_path:
        rows = load_manifest(args.manifest_path)
    elif args.bundle_path:
        rows = load_bundle(args.bundle_path)
    elif args.image_dir:
        rows = load_image_dir(args.image_dir)
    else:
        raise ValueError("no manifest, bundle, or image directory provided")
    reward_processor, reward_model = build_reward_model(args.clip_model_name, device)

    images = []
    for row in rows:
        if "image" in row:
            images.append(row["image"].convert("RGB"))
        else:
            images.append(Image.open(str(row["image_path"])).convert("RGB"))
    scores = score_images(reward_processor, reward_model, images, device).detach().cpu().tolist()

    eval_rows: List[Dict[str, Any]] = []
    for row, score in zip(rows, scores):
        eval_rows.append(
            {
                "prompt": str(row["prompt"]),
                "seed": int(row["seed"]),
                "image_path": str(row["image_path"]),
                "score": float(score),
            }
        )

    avg_score = sum(row["score"] for row in eval_rows) / max(len(eval_rows), 1)
    print(f"eval-score-avg: {avg_score:.4f}")
    for row in eval_rows:
        print(
            f"eval-score: prompt={row['prompt']}"
            f" seed={row['seed']}"
            f" score={row['score']:.4f}"
            f" image_path={row['image_path']}"
        )

    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(eval_rows, f, indent=2)


if __name__ == "__main__":
    main()
