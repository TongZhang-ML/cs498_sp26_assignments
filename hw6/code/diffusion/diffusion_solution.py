#!/usr/bin/env python3
"""
CLIP-score guided LoRA fine-tuning for Stable Diffusion 1.5.

This version targets the actual homework pipeline:
  - Stable Diffusion 1.5 image generation
  - at least 20 diffusion inference steps per sample
  - LoRA attached to the UNet attention layers
  - 8 generated candidates per prompt
  - frozen CLIP scoring
  - top-1 selection
  - one LoRA-only SFT-style diffusion update on the selected image
  - LoRA-only checkpoint saving and loading
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from PIL import Image, ImageDraw
except ImportError:
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]

_DIFFUSERS_IMPORT_ERROR: Exception | None = None
try:
    from diffusers import DDPMScheduler, StableDiffusionPipeline
except Exception as e:
    _DIFFUSERS_IMPORT_ERROR = e
    DDPMScheduler = None  # type: ignore[assignment]
    StableDiffusionPipeline = None  # type: ignore[assignment]

try:
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
except ImportError:
    LoraConfig = None  # type: ignore[assignment]
    get_peft_model_state_dict = None  # type: ignore[assignment]
    set_peft_model_state_dict = None  # type: ignore[assignment]

try:
    from transformers import AutoProcessor, CLIPModel
except ImportError:
    AutoProcessor = None  # type: ignore[assignment]
    CLIPModel = None  # type: ignore[assignment]

from diffusion_gendata import TRAIN_PROMPTS


SD_MODEL_NAME = "runwayml/stable-diffusion-v1-5"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
TARGET_CONCEPT = "a Monet painting of a flying pig"
TRAIN_PROMPT_COUNT = 16
CANDIDATES_PER_PROMPT = 8
EVAL_IMAGES = 4
MIN_INFERENCE_STEPS = 20
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_IMAGE_SIZE = 512

if torch is not None:
    _no_grad = torch.no_grad
else:
    def _no_grad():
        def decorator(fn):
            return fn
        return decorator


def _require_runtime() -> None:
    missing: List[str] = []
    if torch is None:
        missing.append("torch")
    if Image is None:
        missing.append("pillow")
    if StableDiffusionPipeline is None or DDPMScheduler is None:
        missing.append("diffusers")
    if AutoProcessor is None or CLIPModel is None:
        missing.append("transformers")
    if missing:
        pkg_list = ", ".join(missing)
        extra = ""
        if "diffusers" in missing and _DIFFUSERS_IMPORT_ERROR is not None:
            extra = f" Underlying diffusers import error: {type(_DIFFUSERS_IMPORT_ERROR).__name__}: {_DIFFUSERS_IMPORT_ERROR}"
        raise RuntimeError(
            f"Missing required packages: {pkg_list}. "
            "Install the diffusion dependencies first, for example: "
            "`pip install -r ../requirements.txt` from the code/diffusion directory "
            "or install torch pillow diffusers transformers peft accelerate manually."
            + extra
        )


def _require_peft() -> None:
    if LoraConfig is None or get_peft_model_state_dict is None or set_peft_model_state_dict is None:
        raise RuntimeError(
            "Missing required package: peft. "
            "Install the diffusion dependencies first, for example: "
            "`pip install -r ../requirements.txt` from the code/diffusion directory."
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_prompt_bank() -> List[str]:
    """
    Implement:
        Return the 16 training prompts used for LoRA fine-tuning.

        Requirements:
          - return exactly 16 prompts
          - the prompts must match the generated training split
          - use diffusion_gendata.py as the source of truth

        Returns:
          List[str]: the ordered training prompt bank.
    """
    return list(TRAIN_PROMPTS)


def _lora_target_modules() -> List[str]:
    return ["to_q", "to_k", "to_v", "to_out.0"]


def build_pipeline_and_reward_model(
    sd_model_name: str,
    clip_model_name: str,
    device: torch.device,
) -> Tuple[Any, Any, Any]:
    """
    Helper:
        Build the Stable Diffusion 1.5 pipeline with a trainable UNet LoRA
        adapter, plus a frozen CLIP reward model and processor.
    """
    _require_runtime()
    _require_peft()
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        sd_model_name,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        init_lora_weights="gaussian",
        target_modules=_lora_target_modules(),
    )
    pipe.unet.add_adapter(lora_config)

    reward_processor = AutoProcessor.from_pretrained(clip_model_name)
    reward_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    reward_model.eval()

    return pipe, reward_processor, reward_model


def _generation_device(device: torch.device) -> str:
    return "cuda" if device.type == "cuda" else "cpu"


def _set_lora_enabled(pipe: Any, enabled: bool) -> None:
    if enabled:
        if hasattr(pipe.unet, "enable_adapters"):
            pipe.unet.enable_adapters()
        return
    if hasattr(pipe.unet, "disable_adapters"):
        pipe.unet.disable_adapters()


@_no_grad()
def generate_candidate_images(
    pipe: Any,
    prompt: str,
    seeds: Sequence[int],
    device: torch.device,
    num_inference_steps: int = 25,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    height: int = DEFAULT_IMAGE_SIZE,
    width: int = DEFAULT_IMAGE_SIZE,
) -> Dict[str, Any]:
    """
    Implement:
        Generate multiple candidate images for one prompt with Stable
        Diffusion 1.5.

        Args:
          pipe: StableDiffusionPipeline with a LoRA-enabled UNet.
          prompt: one text prompt.
          seeds: deterministic seeds used to generate the candidates.
          device: target torch device.
          num_inference_steps: number of denoising steps. Must be at least 20.
          guidance_scale: classifier-free guidance scale.
          height: image height.
          width: image width.

        Requirements:
          - use at least 20 diffusion inference steps
          - generate one image per seed
          - keep generation deterministic with the provided seeds
          - return the PIL images and metadata needed by later stages

        Returns:
          Dict with keys:
            "prompt", "seeds", "images", "num_inference_steps",
            "guidance_scale".
    """
    _require_runtime()
    step_count = max(int(num_inference_steps), MIN_INFERENCE_STEPS)
    images: List[Image.Image] = []
    for seed in seeds:
        generator = torch.Generator(device=_generation_device(device)).manual_seed(int(seed))
        result = pipe(
            prompt=prompt,
            generator=generator,
            num_inference_steps=step_count,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
        )
        images.append(result.images[0])
    return {
        "prompt": prompt,
        "seeds": [int(seed) for seed in seeds],
        "images": images,
        "num_inference_steps": step_count,
        "guidance_scale": float(guidance_scale),
    }


@_no_grad()
def _clip_scores(
    reward_processor: Any,
    reward_model: Any,
    prompt: str,
    images: Sequence[Image.Image],
    device: torch.device,
) -> torch.Tensor:
    image_inputs = reward_processor(images=list(images), return_tensors="pt")
    text_inputs = reward_processor(text=[prompt], return_tensors="pt", padding=True)

    pixel_values = image_inputs["pixel_values"].to(device)
    input_ids = text_inputs["input_ids"].to(device)
    attention_mask = text_inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    image_features = reward_model.get_image_features(pixel_values=pixel_values)
    text_features = reward_model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    return image_features @ text_features.T


@_no_grad()
def pick_top1_candidate(
    reward_processor: Any,
    reward_model: Any,
    prompt: str,
    images: Sequence[Image.Image],
    seeds: Sequence[int],
    device: torch.device,
) -> Dict[str, Any]:
    """
    Implement:
        Score all candidates with frozen CLIP and keep the top-1 image.

        Args:
          reward_processor: CLIP processor.
          reward_model: frozen CLIP model.
          prompt: text prompt paired with all candidate images.
          images: candidate images for that prompt.
          seeds: candidate seeds in the same order as images.
          device: torch device.

        Requirements:
          - compute one CLIP score per candidate image
          - select the highest-scoring image
          - return the top-1 image, its score, and its seed

        Returns:
          Dict with keys:
            "scores", "best_index", "best_seed", "best_score", "best_image".
    """
    scores = _clip_scores(reward_processor, reward_model, prompt, images, device).squeeze(1)
    best_index = int(torch.argmax(scores).item())
    return {
        "scores": scores.detach().cpu(),
        "best_index": best_index,
        "best_seed": int(seeds[best_index]),
        "best_score": float(scores[best_index].item()),
        "best_image": images[best_index],
    }


def _image_to_tensor(image: Image.Image, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=dtype)


def _encode_prompt(pipe: Any, prompt: str, device: torch.device) -> torch.Tensor:
    text_inputs = pipe.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    with torch.no_grad():
        prompt_embeds = pipe.text_encoder(input_ids)[0]
    return prompt_embeds


def _train_step_from_selected_image(
    pipe: Any,
    noise_scheduler: Any,
    prompt: str,
    selected_image: Image.Image,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    pipe.unet.train()
    optimizer.zero_grad()

    pixel_values = _image_to_tensor(selected_image, device, pipe.vae.dtype)
    with torch.no_grad():
        latents = pipe.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * pipe.vae.config.scaling_factor
        prompt_embeds = _encode_prompt(pipe, prompt, device)

    noise = torch.randn_like(latents)
    timesteps = torch.randint(
        low=0,
        high=noise_scheduler.config.num_train_timesteps,
        size=(latents.shape[0],),
        device=device,
        dtype=torch.long,
    )
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=prompt_embeds).sample
    loss = F.mse_loss(noise_pred.float(), noise.float())
    loss.backward()
    optimizer.step()
    return float(loss.item())


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 3
    lr: float = 1e-5
    weight_decay: float = 0.0
    candidates_per_prompt: int = CANDIDATES_PER_PROMPT
    num_inference_steps: int = 25
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE
    height: int = DEFAULT_IMAGE_SIZE
    width: int = DEFAULT_IMAGE_SIZE


@_no_grad()
def evaluate_prompt(
    pipe: Any,
    reward_processor: Any,
    reward_model: Any,
    prompt: str,
    start_seed: int,
    device: torch.device,
    use_lora: bool,
    cfg: TrainConfig,
) -> Dict[str, Any]:
    seeds = [int(start_seed) + i for i in range(EVAL_IMAGES)]
    _set_lora_enabled(pipe, use_lora)
    sampled = generate_candidate_images(
        pipe=pipe,
        prompt=prompt,
        seeds=seeds,
        device=device,
        num_inference_steps=cfg.num_inference_steps,
        guidance_scale=cfg.guidance_scale,
        height=cfg.height,
        width=cfg.width,
    )
    scores = _clip_scores(reward_processor, reward_model, prompt, sampled["images"], device).squeeze(1)
    return {
        "prompt": prompt,
        "seeds": seeds,
        "avg_score": float(scores.mean().item()),
        "images": sampled["images"],
        "scores": scores.detach().cpu(),
    }


def train_lora_sft(
    pipe: Any,
    reward_processor: Any,
    reward_model: Any,
    train_rows: Sequence[Dict[str, Any]],
    val_row: Dict[str, Any],
    cfg: TrainConfig,
    device: torch.device,
) -> Tuple[Any, Dict[str, float], List[Dict[str, Any]]]:
    """
    Implement:
        Run CLIP-guided top-1 LoRA fine-tuning for Stable Diffusion 1.5.

        Training protocol:
          - iterate over the 16 training prompts
          - generate 8 candidates per prompt using at least 20 steps
          - score them with frozen CLIP
          - keep the top-1 image
          - perform one LoRA-only diffusion training step on the selected image
          - evaluate after each epoch on the visible validation prompt
          - keep the best LoRA weights by tuned validation score

        Returns:
          Tuple of:
            - the pipeline with best LoRA weights restored
            - metrics dict containing tuned, baseline, and improvement scores
            - a compact trace list with per-prompt selections
    """
    _require_runtime()
    _require_peft()
    trainable_params = [p for p in pipe.unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    best_state: Dict[str, torch.Tensor] | None = None
    best_val = -1e9
    best_metrics: Dict[str, float] | None = None
    trace: List[Dict[str, Any]] = []

    for epoch in range(cfg.epochs):
        print(f"train-epoch-start: epoch={epoch + 1}/{cfg.epochs}")
        random_order = list(train_rows)
        random.shuffle(random_order)
        epoch_loss = 0.0
        epoch_score = 0.0
        _set_lora_enabled(pipe, True)

        for step_idx, row in enumerate(random_order, start=1):
            prompt = str(row["prompt"])
            base_seed = int(row["seed"])
            candidate_seeds = [base_seed + i for i in range(cfg.candidates_per_prompt)]
            sampled = generate_candidate_images(
                pipe=pipe,
                prompt=prompt,
                seeds=candidate_seeds,
                device=device,
                num_inference_steps=cfg.num_inference_steps,
                guidance_scale=cfg.guidance_scale,
                height=cfg.height,
                width=cfg.width,
            )
            picked = pick_top1_candidate(
                reward_processor=reward_processor,
                reward_model=reward_model,
                prompt=prompt,
                images=sampled["images"],
                seeds=candidate_seeds,
                device=device,
            )
            loss = _train_step_from_selected_image(
                pipe=pipe,
                noise_scheduler=noise_scheduler,
                prompt=prompt,
                selected_image=picked["best_image"],
                optimizer=optimizer,
                device=device,
            )
            epoch_loss += loss
            epoch_score += float(picked["best_score"])
            print(
                "train-step:"
                f" epoch={epoch + 1}/{cfg.epochs}"
                f" step={step_idx}/{len(random_order)}"
                f" best_seed={int(picked['best_seed'])}"
                f" best_score={float(picked['best_score']):.4f}"
                f" loss={loss:.4f}"
                f" prompt={prompt}"
            )
            trace.append(
                {
                    "epoch": epoch,
                    "prompt": prompt,
                    "base_seed": base_seed,
                    "candidate_seeds": candidate_seeds,
                    "best_seed": int(picked["best_seed"]),
                    "best_score": float(picked["best_score"]),
                }
            )

        val_base_eval = evaluate_prompt(pipe, reward_processor, reward_model, str(val_row["prompt"]), int(val_row["seed"]), device, use_lora=False, cfg=cfg)
        val_tuned_eval = evaluate_prompt(pipe, reward_processor, reward_model, str(val_row["prompt"]), int(val_row["seed"]), device, use_lora=True, cfg=cfg)
        metrics = {
            "train_loss": epoch_loss / max(len(train_rows), 1),
            "train_top1_score": epoch_score / max(len(train_rows), 1),
            "val_base_avg_score": float(val_base_eval["avg_score"]),
            "val_tuned_avg_score": float(val_tuned_eval["avg_score"]),
            "val_improvement": float(val_tuned_eval["avg_score"] - val_base_eval["avg_score"]),
        }
        print(
            "train-epoch-end:"
            f" epoch={epoch + 1}/{cfg.epochs}"
            f" train_loss={metrics['train_loss']:.4f}"
            f" train_top1_score={metrics['train_top1_score']:.4f}"
            f" val_base_avg_score={metrics['val_base_avg_score']:.4f}"
            f" val_lora_avg_score={metrics['val_tuned_avg_score']:.4f}"
            f" val_improvement={metrics['val_improvement']:.4f}"
        )
        if metrics["val_tuned_avg_score"] > best_val:
            best_val = metrics["val_tuned_avg_score"]
            best_metrics = metrics
            best_state = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(pipe.unet).items()}
            print(
                "train-best-update:"
                f" epoch={epoch + 1}/{cfg.epochs}"
                f" best_val_score={best_val:.4f}"
            )

    if best_state is None or best_metrics is None:
        raise RuntimeError("training did not produce a best LoRA checkpoint")

    set_peft_model_state_dict(pipe.unet, best_state)
    print(
        "train-complete:"
        f" best_val_base_avg_score={best_metrics['val_base_avg_score']:.4f}"
        f" best_val_lora_avg_score={best_metrics['val_tuned_avg_score']:.4f}"
        f" best_val_improvement={best_metrics['val_improvement']:.4f}"
    )
    return pipe, best_metrics, trace


def save_lora_checkpoint(path: str, pipe: Any, metrics: Dict[str, float], cfg: TrainConfig) -> None:
    _require_peft()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "unet_lora": {k: v.detach().cpu() for k, v in get_peft_model_state_dict(pipe.unet).items()},
            "sd_model_name": SD_MODEL_NAME,
            "clip_model_name": CLIP_MODEL_NAME,
            "target_concept": TARGET_CONCEPT,
            "metrics": metrics,
            "train_config": cfg.__dict__,
        },
        path,
    )


def load_lora_checkpoint(path: str, pipe: Any) -> Dict[str, Any]:
    _require_peft()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    set_peft_model_state_dict(pipe.unet, payload["unet_lora"])
    return payload


def save_trace(path: str, trace_rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(trace_rows), f, indent=2)


def save_image_sheet(path: str, prompt: str, images: Sequence[Image.Image], scores: Sequence[float]) -> None:
    if Image is None or ImageDraw is None:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    thumb = 160
    gutter = 12
    text_h = 44
    canvas = Image.new("RGB", (EVAL_IMAGES * thumb + (EVAL_IMAGES + 1) * gutter, thumb + text_h + 2 * gutter), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    for idx, image in enumerate(images[:EVAL_IMAGES]):
        x = gutter + idx * (thumb + gutter)
        y = gutter
        canvas.paste(image.resize((thumb, thumb)), (x, y))
        draw.text((x, y + thumb + 8), f"score={float(scores[idx]):.4f}", fill=(24, 24, 24))
    draw.text((gutter, thumb + text_h + gutter - 16), prompt, fill=(48, 48, 48))
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--data_dir", default=os.path.join(os.path.dirname(__file__), "data"))
    parser.add_argument("--checkpoint_path", default="outputs/diffusion_model.pt")
    parser.add_argument("--trace_path", default="outputs/diffusion_trace.json")
    parser.add_argument("--val_sheet_path", default="outputs/diffusion_val_examples.png")
    parser.add_argument("--sd_model_name", default=SD_MODEL_NAME)
    parser.add_argument("--clip_model_name", default=CLIP_MODEL_NAME)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    _require_runtime()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TrainConfig()

    if args.mode == "train":
        train_rows = load_jsonl(os.path.join(args.data_dir, "train.jsonl"))
        val_rows = load_jsonl(os.path.join(args.data_dir, "val.jsonl"))
        if len(train_rows) != TRAIN_PROMPT_COUNT:
            raise ValueError(f"expected {TRAIN_PROMPT_COUNT} training prompts, found {len(train_rows)}")
        if len(val_rows) != 1:
            raise ValueError(f"expected exactly 1 validation prompt, found {len(val_rows)}")

        pipe, reward_processor, reward_model = build_pipeline_and_reward_model(args.sd_model_name, args.clip_model_name, device)
        pipe, metrics, trace_rows = train_lora_sft(pipe, reward_processor, reward_model, train_rows, val_rows[0], cfg, device)
        save_lora_checkpoint(args.checkpoint_path, pipe, metrics, cfg)
        save_trace(args.trace_path, trace_rows)

        val_base_eval = evaluate_prompt(pipe, reward_processor, reward_model, str(val_rows[0]["prompt"]), int(val_rows[0]["seed"]), device, use_lora=False, cfg=cfg)
        val_eval = evaluate_prompt(pipe, reward_processor, reward_model, str(val_rows[0]["prompt"]), int(val_rows[0]["seed"]), device, use_lora=True, cfg=cfg)
        save_image_sheet(args.val_sheet_path, val_eval["prompt"], val_eval["images"], val_eval["scores"].tolist())
        print(f"val-base-avg-score: {val_base_eval['avg_score']:.4f}")
        print(f"val-lora-avg-score: {val_eval['avg_score']:.4f}")
        print(f"val-improvement: {val_eval['avg_score'] - val_base_eval['avg_score']:.4f}")
        print(f"val-prompt: {val_eval['prompt']}")
        for i, score in enumerate(val_base_eval["scores"].tolist()):
            print(f"val-base-image-{i}: seed={val_base_eval['seeds'][i]} score={float(score):.4f}")
        for i, score in enumerate(val_eval["scores"].tolist()):
            print(f"val-lora-image-{i}: seed={val_eval['seeds'][i]} score={float(score):.4f}")
    else:
        test_rows = load_jsonl(os.path.join(args.data_dir, "test.jsonl"))
        if len(test_rows) != 1:
            raise ValueError(f"expected exactly 1 hidden test prompt, found {len(test_rows)}")
        pipe, reward_processor, reward_model = build_pipeline_and_reward_model(args.sd_model_name, args.clip_model_name, device)
        payload = load_lora_checkpoint(args.checkpoint_path, pipe)
        test_base_eval = evaluate_prompt(pipe, reward_processor, reward_model, str(test_rows[0]["prompt"]), int(test_rows[0]["seed"]), device, use_lora=False, cfg=cfg)
        test_eval = evaluate_prompt(pipe, reward_processor, reward_model, str(test_rows[0]["prompt"]), int(test_rows[0]["seed"]), device, use_lora=True, cfg=cfg)
        print(f"loaded-target: {payload['target_concept']}")
        print(f"test-base-avg-score: {test_base_eval['avg_score']:.4f}")
        print(f"test-lora-avg-score: {test_eval['avg_score']:.4f}")
        print(f"test-improvement: {test_eval['avg_score'] - test_base_eval['avg_score']:.4f}")


if __name__ == "__main__":
    main()
