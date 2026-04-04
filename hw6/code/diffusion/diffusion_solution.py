#!/usr/bin/env python3
"""
Fine-tune Stable Diffusion 1.5 on the visible training set and save base and
fine-tuned generations for a separate 10-prompt evaluation set.
"""

from __future__ import annotations

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


SD_MODEL_NAME = "runwayml/stable-diffusion-v1-5"
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
MIN_INFERENCE_STEPS = 20
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_IMAGE_SIZE = 256
DEFAULT_SEED = 13
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DEFAULT_BASE_EVAL_IMAGE_DIR = "outputs/base_eval_images"
DEFAULT_FINETUNED_EVAL_IMAGE_DIR = "outputs/finetuned_eval_images"
DEFAULT_COMPARISON_SHEET_PATH = "outputs/diffusion_eval_comparison.png"


def _require_runtime() -> None:
    missing: List[str] = []
    if torch is None:
        missing.append("torch")
    if Image is None:
        missing.append("pillow")
    if StableDiffusionPipeline is None or DDPMScheduler is None:
        missing.append("diffusers")
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
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


def build_prompt_bank(train_rows: Sequence[Dict[str, Any]]) -> List[str]:
    seen = set()
    prompts: List[str] = []
    for row in train_rows:
        prompt = str(row["prompt"])
        if prompt not in seen:
            seen.add(prompt)
            prompts.append(prompt)
    return prompts


def _generation_device(device: torch.device) -> str:
    return "cuda" if device.type == "cuda" else "cpu"


def build_pipeline(sd_model_name: str, device: torch.device) -> Any:
    """
    Implement:
        Build the Stable Diffusion 1.5 pipeline for full-UNet fine-tuning,
        move it to `device`, freeze the VAE and text encoder, and leave the
        UNet trainable.

    Args:
        sd_model_name:
            Local Stable Diffusion model identifier to load.
        device:
            Torch device where the pipeline should live.

    Returns:
        A `StableDiffusionPipeline` ready for fine-tuning and generation.

    Required behavior:
        - Load the Stable Diffusion pipeline from `sd_model_name`.
        - Keep loading local-only so the visible solution does not depend on
          network downloads at runtime.
        - Disable the safety checker.
        - Move the pipeline to `device`.
        - Freeze `pipe.vae` and `pipe.text_encoder`.
        - Leave `pipe.unet` trainable and cast it to `float32`.
        - Disable the progress bar for cleaner script output.

    Notes:
        - This homework uses full UNet fine-tuning, not LoRA.
        - The returned pipeline must work for both training and later image
          generation in the same script.
    """
    _require_runtime()
    pipe = StableDiffusionPipeline.from_pretrained(
        sd_model_name,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=True,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(True)
    pipe.unet.to(dtype=torch.float32)
    return pipe


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
) -> Dict[str, Any]:
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
    return {
        "prompt": prompt,
        "seeds": [int(seed) for seed in seeds],
        "images": list(result.images),
    }


def _image_to_tensor(image: Image.Image, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=dtype)


def _load_selected_image(image_path: str) -> Image.Image:
    if Image is None:
        raise RuntimeError("pillow is required to load finetuning images")
    if image_path.endswith(".pt"):
        image_uint8 = torch.load(image_path, map_location="cpu", weights_only=True)
        if not isinstance(image_uint8, torch.Tensor):
            raise TypeError(f"expected tensor in {image_path}")
        return Image.fromarray(image_uint8.numpy(), mode="RGB")
    return Image.open(image_path).convert("RGB")


def _encode_prompts(pipe: Any, prompts: Sequence[str], device: torch.device) -> torch.Tensor:
    text_inputs = pipe.tokenizer(
        list(prompts),
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    with torch.no_grad():
        prompt_embeds = pipe.text_encoder(input_ids)[0]
    return prompt_embeds


def _train_step_from_selected_batch(
    pipe: Any,
    noise_scheduler: Any,
    prompts: Sequence[str],
    selected_images: Sequence[Image.Image],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    pipe.unet.train()
    optimizer.zero_grad()

    pixel_values = torch.cat([_image_to_tensor(image, device, pipe.vae.dtype) for image in selected_images], dim=0)
    with torch.no_grad():
        latents = pipe.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * pipe.vae.config.scaling_factor
        prompt_embeds = _encode_prompts(pipe, prompts, device)

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


def train_diffusion_batch(
    pipe: Any,
    noise_scheduler: Any,
    optimizer: torch.optim.Optimizer,
    batch_rows: Sequence[Dict[str, Any]],
    device: torch.device,
    data_dir: str,
) -> Dict[str, float]:
    """
    Implement:
        Run one diffusion fine-tuning step on a batch of selected prompt/image
        pairs loaded from the visible training split.

    Args:
        pipe:
            Stable Diffusion pipeline with a trainable UNet.
        noise_scheduler:
            Diffusion scheduler used for training-time noise injection.
        optimizer:
            Optimizer for the UNet parameters.
        batch_rows:
            Batch of JSON rows with at least `prompt` and `image_path`.
        device:
            Torch device used for training.
        data_dir:
            Root directory containing the saved training images.

    Returns:
        Dict with scalar logging metrics for the batch.
        Required key:
          - `loss`

    Required behavior:
        - Read `prompt` and `image_path` from every row in `batch_rows`.
        - Load the saved images from `data_dir`.
        - Run exactly one diffusion training step on that batch.
        - Use `noise_scheduler` and `optimizer` for the update.
        - Return Python floats for logging, not tensors.

    Expected metric keys:
        - `loss`
    """
    batch_prompts = [str(row["prompt"]) for row in batch_rows]
    batch_images = [_load_selected_image(os.path.join(data_dir, str(row["image_path"]))) for row in batch_rows]
    loss = _train_step_from_selected_batch(
        pipe=pipe,
        noise_scheduler=noise_scheduler,
        prompts=batch_prompts,
        selected_images=batch_images,
        optimizer=optimizer,
        device=device,
    )
    return {
        "loss": float(loss),
    }


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 10
    lr: float = 1e-5
    weight_decay: float = 0.0
    batch_size: int = 10
    num_inference_steps: int = 20
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE
    height: int = DEFAULT_IMAGE_SIZE
    width: int = DEFAULT_IMAGE_SIZE


def train_sft(
    pipe: Any,
    train_rows: Sequence[Dict[str, Any]],
    cfg: TrainConfig,
    device: torch.device,
    data_dir: str,
) -> Tuple[Any, Dict[str, float], List[Dict[str, Any]]]:
    trainable_params = [p for p in pipe.unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    trace: List[Dict[str, Any]] = []
    last_metrics: Dict[str, float] | None = None
    epoch_train_losses: List[float] = []

    for epoch in range(cfg.epochs):
        print(f"train-epoch-start: epoch={epoch + 1}/{cfg.epochs}")
        random_order = list(train_rows)
        random.shuffle(random_order)
        epoch_loss = 0.0
        train_batches = [random_order[i : i + cfg.batch_size] for i in range(0, len(random_order), cfg.batch_size)]

        for step_idx, batch_rows in enumerate(train_batches, start=1):
            batch_metrics = train_diffusion_batch(
                pipe=pipe,
                noise_scheduler=noise_scheduler,
                optimizer=optimizer,
                batch_rows=batch_rows,
                device=device,
                data_dir=data_dir,
            )
            epoch_loss += batch_metrics["loss"]
            print(
                "train-step:"
                f" epoch={epoch + 1}/{cfg.epochs}"
                f" step={step_idx}/{len(train_batches)}"
                f" batch_size={len(batch_rows)}"
                f" loss={batch_metrics['loss']:.4f}"
            )
            trace.extend(
                {
                    "epoch": epoch,
                    "prompt": str(row["prompt"]),
                    "image_path": str(row["image_path"]),
                }
                for row in batch_rows
            )

        avg_epoch_loss = epoch_loss / max(len(train_batches), 1)
        epoch_train_losses.append(avg_epoch_loss)
        last_metrics = {
            "train_loss": avg_epoch_loss,
            "epoch_train_losses": list(epoch_train_losses),
        }
        print(
            "train-epoch-end:"
            f" epoch={epoch + 1}/{cfg.epochs}"
            f" batches_this_epoch={len(train_batches)}"
            f" train_loss={last_metrics['train_loss']:.4f}"
        )

    if last_metrics is None:
        raise RuntimeError("training did not produce any optimization steps")
    print(f"train-complete: final_train_loss={last_metrics['train_loss']:.4f}")
    return pipe, last_metrics, trace


def save_generated_images(image_dir: str, rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    os.makedirs(image_dir, exist_ok=True)
    saved_rows: List[Dict[str, Any]] = []
    for row in rows:
        prompt = str(row["prompt"])
        image_path = os.path.join(image_dir, f"{prompt}.png")
        row["image"].save(image_path)
        saved_rows.append(
            {
                "prompt": prompt,
                "seed": int(row["seed"]),
                "image_path": image_path,
                "image": row["image"],
            }
        )
    return saved_rows


def save_comparison_sheet(path: str, base_rows: Sequence[Dict[str, Any]], finetuned_rows: Sequence[Dict[str, Any]]) -> None:
    if Image is None or ImageDraw is None:
        return
    if len(base_rows) != len(finetuned_rows):
        raise ValueError("base and finetuned rows must have the same length")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    thumb = 160
    gutter = 12
    text_h = 18
    pairs_per_row = 2
    row_h = thumb + text_h + gutter
    row_count = (len(base_rows) + pairs_per_row - 1) // pairs_per_row
    canvas_w = 4 * thumb + 5 * gutter
    canvas_h = row_count * row_h + gutter
    canvas = Image.new("RGB", (canvas_w, canvas_h), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)

    for idx, (base_row, finetuned_row) in enumerate(zip(base_rows, finetuned_rows)):
        row_idx = idx // pairs_per_row
        pair_idx = idx % pairs_per_row
        base_col = pair_idx
        finetuned_col = pair_idx + 2
        y = gutter + row_idx * row_h
        for col_idx, row in ((base_col, base_row), (finetuned_col, finetuned_row)):
            x = gutter + col_idx * (thumb + gutter)
            image = row["image"].resize((thumb, thumb))
            canvas.paste(image, (x, y))
            label = str(row["prompt"])
            draw.text((x, y + thumb + 2), label[:24], fill=(24, 24, 24))

    draw.text((gutter, 2), "Base", fill=(24, 24, 24))
    draw.text((gutter + 2 * (thumb + gutter), 2), "Finetuned", fill=(24, 24, 24))
    canvas.save(path)


def format_latex_table(metrics: Dict[str, float]) -> str:
    epoch_train_losses = metrics.get("epoch_train_losses", [])
    if not isinstance(epoch_train_losses, list) or len(epoch_train_losses) == 0:
        epoch_train_losses = [float(metrics["train_loss"])]

    rows = [
        "\\begin{tabular}{lr}",
        "\\hline",
        "Epoch & Train loss \\\\",
        "\\hline",
    ]
    for epoch_idx, loss in enumerate(epoch_train_losses, start=1):
        rows.append(f"{epoch_idx} & {float(loss):.4f} \\\\")
    rows.extend(
        [
            "\\hline",
            "\\end{tabular}",
        ]
    )
    return "\n".join(rows)


@torch.no_grad()
def generate_prompt_set(
    pipe: Any,
    prompts: Sequence[str],
    device: torch.device,
    cfg: TrainConfig,
    seed_offset: int,
) -> List[Dict[str, Any]]:
    """
    Implement:
        Generate one deterministic image for each prompt in a prompt set and
        return the prompt/seed/image records needed for later saving.

    Args:
        pipe:
            Stable Diffusion pipeline used for generation.
        prompts:
            Prompt strings to generate.
        device:
            Torch device used for generation.
        cfg:
            Generation hyperparameters.
        seed_offset:
            Base seed offset used to make prompt-wise generations
            deterministic.

    Returns:
        List of dicts with keys:
          - `prompt`
          - `seed`
          - `image`

    Required behavior:
        - Generate exactly one image per prompt.
        - Use deterministic seeds `seed_offset + prompt_idx`.
        - Respect the configured inference steps, guidance scale, height, and
          width from `cfg`.
        - Return PIL images so later helpers can save them to disk or pack
          them into comparison figures.

    Notes:
        - These prompts are used only for qualitative/evaluation outputs, not
          for training updates.
    """
    rows: List[Dict[str, Any]] = []
    for prompt_idx, prompt in enumerate(prompts):
        generated = generate_images(
            pipe=pipe,
            prompt=prompt,
            seeds=[seed_offset + prompt_idx],
            device=device,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            height=cfg.height,
            width=cfg.width,
        )
        rows.append(
            {
                "prompt": prompt,
                "seed": int(generated["seeds"][0]),
                "image": generated["images"][0],
            }
        )
    return rows


def save_eval_outputs(
    image_dir: str,
    rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return save_generated_images(image_dir, rows)


def main() -> None:
    _require_runtime()
    set_seed(DEFAULT_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    cfg = TrainConfig(
        epochs=10,
        lr=1e-5,
        batch_size=10,
        num_inference_steps=20,
        guidance_scale=DEFAULT_GUIDANCE_SCALE,
        height=DEFAULT_IMAGE_SIZE,
        width=DEFAULT_IMAGE_SIZE,
    )
    train_rows = load_jsonl(os.path.join(DEFAULT_DATA_DIR, "train.jsonl"))
    train_prompts = build_prompt_bank(train_rows)
    if len(train_rows) != len(train_prompts):
        raise ValueError(
            "expected one training image per prompt, "
            f"found {len(train_rows)} rows and {len(train_prompts)} unique prompts"
        )

    pipe = build_pipeline(SD_MODEL_NAME, device)

    base_eval_rows = generate_prompt_set(
        pipe=pipe,
        prompts=EVAL_PROMPTS,
        device=device,
        cfg=cfg,
        seed_offset=9000,
    )
    base_eval_rows = save_eval_outputs(DEFAULT_BASE_EVAL_IMAGE_DIR, base_eval_rows)

    pipe, metrics, trace_rows = train_sft(pipe, train_rows, cfg, device, DEFAULT_DATA_DIR)
    del trace_rows
    print(format_latex_table(metrics))

    finetuned_eval_rows = generate_prompt_set(
        pipe=pipe,
        prompts=EVAL_PROMPTS,
        device=device,
        cfg=cfg,
        seed_offset=9000,
    )
    finetuned_eval_rows = save_eval_outputs(DEFAULT_FINETUNED_EVAL_IMAGE_DIR, finetuned_eval_rows)
    save_comparison_sheet(DEFAULT_COMPARISON_SHEET_PATH, base_eval_rows, finetuned_eval_rows)

    for row in base_eval_rows:
        print(
            f"base-eval-image: prompt={row['prompt']}"
            f" seed={row['seed']}"
            f" image_path={row['image_path']}"
        )
    for row in finetuned_eval_rows:
        print(
            f"finetuned-eval-image: prompt={row['prompt']}"
            f" seed={row['seed']}"
            f" image_path={row['image_path']}"
        )


if __name__ == "__main__":
    main()
