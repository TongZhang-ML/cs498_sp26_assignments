#!/usr/bin/env python3
"""
Pretrained VLM adaptation for synthetic visual question answering.

This version uses a small Hugging Face VLM:
  - AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")
  - AutoModelForVision2Seq.from_pretrained(...)

Train behavior:
  - run  zero-shot VLM inference on val
  - LoRA fine-tune the VLM on the visible train split
  - report train/val metrics after each epoch
  - save outputs/vlm_model.pt
  - save a few labeled example images

Test behavior:
  - load outputs/vlm_model.pt
  - run zero-shot VLM inference on the available eval split
  - run the fine-tuned VLM on the available eval split
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

try:
    import torch
    from PIL import Image, ImageDraw
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    torch = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    DataLoader = object  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment]

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:
    LoraConfig = None  # type: ignore[assignment]
    TaskType = None  # type: ignore[assignment]
    get_peft_model = None  # type: ignore[assignment]

try:
    from transformers import AutoModelForVision2Seq, AutoProcessor
except ImportError:
    AutoProcessor = None  # type: ignore[assignment]
    AutoModelForVision2Seq = None  # type: ignore[assignment]


MODEL_NAME = "HuggingFaceTB/SmolVLM-500M-Instruct"
ANSWER_VOCAB = ["red", "blue", "green", "yellow", "yes", "no"]
SYSTEM_PROMPT = "Answer with one word only from: red, blue, green, yellow, yes, no."
IMAGE_SIZE = 256
MAX_NEW_TOKENS = 2
IMAGE_SEQ_LEN = 16
_PRINTED_INPUT_DEBUG = False

if torch is not None:
    _no_grad = torch.no_grad
else:
    def _no_grad():
        def decorator(fn):
            return fn
        return decorator


def _require_runtime() -> None:
    if torch is None or AutoProcessor is None or AutoModelForVision2Seq is None:
        raise RuntimeError("torch and transformers are required for the VLM solution")


def _require_peft() -> None:
    if LoraConfig is None or get_peft_model is None:
        raise RuntimeError("peft is required for LoRA fine-tuning")


def set_seed(seed: int) -> None:
    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _resolve_image_path(data_dir: str, row: Dict[str, Any]) -> str:
    image_path = str(row["image_path"])
    if os.path.isabs(image_path):
        return image_path
    return os.path.join(data_dir, image_path)


def _load_image(image_path: str) -> Image.Image:
    if Image is None:
        raise RuntimeError("Pillow is required for image loading")
    return Image.open(image_path).convert("RGB")


def _extract_answer(text: str) -> str:
    text = text.strip().lower()
    tokens = re.findall(r"[a-z]+", text)
    for token in tokens:
        if token in ANSWER_VOCAB:
            return token
    return tokens[0] if tokens else ""


def _move_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def _maybe_print_input_debug(batch: Dict[str, Any]) -> None:
    global _PRINTED_INPUT_DEBUG
    if _PRINTED_INPUT_DEBUG:
        return
    parts: List[str] = []
    if "input_ids" in batch:
        parts.append(f"input_ids_shape={tuple(batch['input_ids'].shape)}")
    if "pixel_values" in batch:
        parts.append(f"pixel_values_shape={tuple(batch['pixel_values'].shape)}")
    if "attention_mask" in batch:
        parts.append(f"attention_mask_shape={tuple(batch['attention_mask'].shape)}")
    if parts:
        print("debug-vlm-input:", " ".join(parts))
        print("debug-note: SmolVLM uses compressed visual tokens; disable image splitting for a lighter path.")
        _PRINTED_INPUT_DEBUG = True


def _render_row_image(row: Dict[str, Any], image_size: int = IMAGE_SIZE) -> Image.Image:
    if Image is None or ImageDraw is None:
        raise RuntimeError("Pillow is required for image rendering")
    color_map = {
        "red": (230, 46, 38),
        "blue": (38, 89, 230),
        "green": (38, 179, 64),
        "yellow": (242, 204, 38),
    }
    rng = random.Random(int(row.get("seed", 0)))
    bg = 236 + rng.randint(-6, 6)
    panel = 244 + rng.randint(-5, 5)
    image = Image.new("RGB", (image_size, image_size), (bg, bg, bg))
    draw = ImageDraw.Draw(image)
    outline = 206 + rng.randint(-6, 6)
    draw.rounded_rectangle((8, 8, image_size - 8, image_size - 8), radius=12, fill=(panel, panel, panel), outline=(outline, outline, outline))

    def jittered_box(center_x: int, center_y: int, base_half_w: int, base_half_h: int) -> Tuple[int, int, int, int]:
        half_w = base_half_w + rng.randint(-4, 4)
        half_h = base_half_h + rng.randint(-8, 8)
        dx = rng.randint(-6, 6)
        dy = rng.randint(-6, 6)
        cx = max(24 + half_w, min(image_size - 24 - half_w, center_x + dx))
        cy = max(24 + half_h, min(image_size - 24 - half_h, center_y + dy))
        return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)

    def draw_shape(shape: str, box: Tuple[int, int, int, int], fill: Tuple[int, int, int]) -> None:
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

    left_box = jittered_box(56, 110, 30, 55)
    right_box = jittered_box(image_size - 56, 110, 30, 55)
    draw_shape(str(row["left_shape"]), left_box, color_map[str(row["left_color"])])
    draw_shape(str(row["right_shape"]), right_box, color_map[str(row["right_color"])])
    return image


def save_vqa_examples(rows: Sequence[Dict[str, Any]], out_path: str, num_examples: int = 4, cols: int = 2) -> None:
    if Image is None or ImageDraw is None:
        return
    shown = list(rows[:num_examples])
    if not shown:
        return
    image_size = IMAGE_SIZE
    text_height = 74
    gutter = 14
    num_rows = (len(shown) + cols - 1) // cols
    canvas = Image.new(
        "RGB",
        (cols * image_size + (cols + 1) * gutter, num_rows * (image_size + text_height) + (num_rows + 1) * gutter),
        (248, 248, 248),
    )
    draw = ImageDraw.Draw(canvas)
    for idx, row in enumerate(shown):
        rr = idx // cols
        cc = idx % cols
        x = gutter + cc * (image_size + gutter)
        y = gutter + rr * (image_size + text_height + gutter)
        canvas.paste(_render_row_image(row), (x, y))
        draw.rectangle((x, y + image_size, x + image_size, y + image_size + text_height), fill=(252, 252, 252))
        draw.text((x + 6, y + image_size + 6), f"Q: {row['question']}", fill=(20, 20, 20))
        draw.text((x + 6, y + image_size + 28), f"A: {row['answer']}", fill=(20, 20, 20))
        draw.text((x + 6, y + image_size + 50), f"Type: {row['question_type']}", fill=(60, 60, 60))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)


def build_prompt(question: str) -> str:
    """
    Implement:
        Build one text prompt for the VLM.

        The returned string is passed directly as the `text=...` argument in
        `processor(images=..., text=prompt, return_tensors="pt")`.
        So this function should return the full user-side prompt string, not a
        tokenized object.

        Args:
          question (str):
            Natural-language question for one image. Example:
            "what color is the object on the left ?"

        Requirements:
          - include the fixed system-style instruction stored in SYSTEM_PROMPT
          - include the question text
          - place the literal `<image>` token before the text instruction so the
            processor knows where the image should appear in the prompt
          - return one plain text string for processor(..., text=prompt)

        Returns:
          str:
            Full prompt string consumed by the processor. It should look like:
            "<image> ...instruction... Question: ...".
    """
    return f"<image>{SYSTEM_PROMPT}\nQuestion: {question}"


def build_model_and_processor(model_name: str, device: torch.device) -> Tuple[Any, Any]:
    """
    Implement:
        Load the pretrained VLM and its processor.

        Args:
          model_name (str):
            Hugging Face model name. In the released code this is
            "HuggingFaceTB/SmolVLM-500M-Instruct".
          device (torch.device):
            Target device for model inference/training, for example cpu or cuda.

        Requirements:
          - call AutoProcessor.from_pretrained(model_name)
          - if the processor has a tokenizer, set tokenizer.padding_side = "left"
          - if the processor has an image processor:
              - set do_image_splitting = False
              - set the longest-edge image size to 256
              - set the max image size to 256 when available
          - if the processor exposes image_seq_len, set it to IMAGE_SEQ_LEN
          - ensure the tokenizer has a valid pad token id
          - call AutoModelForVision2Seq.from_pretrained(model_name, ...)
          - move the model to the requested device
          - copy the tokenizer pad token id into model.config.pad_token_id
          - copy the tokenizer pad token id into model.generation_config.pad_token_id
          - set the model to eval mode before returning

        Returns:
          tuple[Any, Any]:
            A pair (model, processor), where:
              - model is the loaded pretrained SmolVLM model
              - processor handles image and text preprocessing for the model
    """
    _require_runtime()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(model_name)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    if hasattr(processor, "image_processor"):
        processor.image_processor.do_image_splitting = False
        if hasattr(processor.image_processor, "size"):
            processor.image_processor.size = {"longest_edge": 256}
        if hasattr(processor.image_processor, "max_image_size"):
            processor.image_processor.max_image_size = {"longest_edge": 256}
    if hasattr(processor, "image_seq_len"):
        processor.image_seq_len = IMAGE_SEQ_LEN
    if getattr(processor.tokenizer, "pad_token_id", None) is None:
        if getattr(processor.tokenizer, "eos_token", None) is not None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        elif getattr(processor.tokenizer, "unk_token", None) is not None:
            processor.tokenizer.pad_token = processor.tokenizer.unk_token
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=dtype,
        _attn_implementation="flash_attention_2" if device.type == "cuda" else "eager",
    )
    model.to(device)
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = processor.tokenizer.pad_token_id
    if getattr(model.generation_config, "pad_token_id", None) is None:
        model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    model.eval()
    print(
        "vlm-config:",
        f"model_name={model_name}",
        f"do_image_splitting={getattr(getattr(processor, 'image_processor', None), 'do_image_splitting', 'unknown')}",
        f"image_size={getattr(getattr(processor, 'image_processor', None), 'size', 'unknown')}",
        f"image_seq_len={getattr(processor, 'image_seq_len', 'unknown')}",
    )
    return model, processor


def _tokenize_generation_example(processor: Any, image_path: str, question: str) -> Dict[str, Any]:
    prompt = build_prompt(question)
    image = _load_image(image_path)
    batch = processor(images=[image], text=prompt, return_tensors="pt")
    _maybe_print_input_debug(batch)
    return batch


def _build_train_batch(processor: Any, rows: Sequence[Dict[str, Any]], data_dir: str) -> Dict[str, Any]:
    images = [_load_image(_resolve_image_path(data_dir, row)) for row in rows]
    prompts = [build_prompt(str(row["question"])) for row in rows]
    answers = [str(row["answer"]) for row in rows]
    prompt_batch = processor(images=images, text=prompts, return_tensors="pt", padding=True)
    full_texts = [f"{prompt} {answer}" for prompt, answer in zip(prompts, answers)]
    full_batch = processor(images=images, text=full_texts, return_tensors="pt", padding=True)
    labels = full_batch["input_ids"].clone()
    for idx in range(labels.size(0)):
        prompt_len = int(prompt_batch["attention_mask"][idx].sum().item())
        labels[idx, :prompt_len] = -100
        labels[idx, full_batch["attention_mask"][idx] == 0] = -100
    batch = {
        "input_ids": full_batch["input_ids"],
        "attention_mask": full_batch["attention_mask"],
        "pixel_values": full_batch["pixel_values"],
        "labels": labels,
    }
    _maybe_print_input_debug(batch)
    return batch


def _build_eval_batch(processor: Any, rows: Sequence[Dict[str, Any]], data_dir: str) -> Dict[str, Any]:
    images = [_load_image(_resolve_image_path(data_dir, row)) for row in rows]
    prompts = [build_prompt(str(row["question"])) for row in rows]
    batch = processor(images=images, text=prompts, return_tensors="pt", padding=True)
    _maybe_print_input_debug(batch)
    return batch


class VQARowDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[idx]


@_no_grad()
def predict_zero_shot(model: Any, processor: Any, row: Dict[str, Any], data_dir: str, device: torch.device) -> str:
    """
    Implement:
        Run zero-shot VLM inference for one image-question pair.

        Args:
          model (Any):
            Loaded pretrained VLM from build_model_and_processor(...).
          processor (Any):
            Matching processor for the VLM.
          row (dict[str, Any]):
            One dataset example containing at least "image_path" and "question".
          data_dir (str):
            Directory containing the dataset jsonl files and image files.
          device (torch.device):
            Device on which generation should run.

        Requirements:
          - resolve the image path from row["image_path"]
          - build the text prompt with build_prompt(...)
          - preprocess image + prompt with the processor
          - move the batch to device
          - call model.generate(...) with greedy decoding
          - decode only the generated continuation after the prompt tokens
          - normalize the decoded text to one answer string from the answer vocabulary

        Returns:
          str:
            Predicted answer string such as "red", "blue", "yes", or "no".
    """
    image_path = _resolve_image_path(data_dir, row)
    batch = _tokenize_generation_example(processor, image_path, str(row["question"]))
    prompt_len = int(batch["input_ids"].shape[1])
    batch = _move_to_device(batch, device)
    generated = model.generate(**batch, do_sample=False, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=processor.tokenizer.pad_token_id)
    continuation = generated[:, prompt_len:]
    text = processor.batch_decode(continuation, skip_special_tokens=True)[0]
    return _extract_answer(text)


def _print_eval_progress(label: str, step: int, total_steps: int) -> None:
    width = 24
    filled = int(width * step / max(total_steps, 1))
    bar = "#" * filled + "-" * (width - filled)
    sys.stdout.write(f"\r{label} [{bar}] {step}/{total_steps}")
    sys.stdout.flush()
    if step >= total_steps:
        sys.stdout.write("\n")


@_no_grad()
def evaluate_zero_shot(model: Any, processor: Any, rows: Sequence[Dict[str, Any]], data_dir: str, device: torch.device) -> Dict[str, float]:
    losses: List[float] = []
    correct = 0
    total_rows = len(rows)
    for idx, row in enumerate(rows, start=1):
        pred = predict_zero_shot(model, processor, row, data_dir, device)
        target = str(row["answer"]).lower()
        correct += int(pred == target)
        losses.append(0.0 if pred == target else 1.0)
        _print_eval_progress("zeroshot-eval", idx, total_rows)
    return {"loss": float(sum(losses) / max(len(losses), 1)), "accuracy": correct / max(len(rows), 1)}


@_no_grad()
def evaluate_finetuned(model: Any, processor: Any, rows: Sequence[Dict[str, Any]], data_dir: str, device: torch.device, batch_size: int) -> Dict[str, float]:
    dataset = VQARowDataset(rows)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    losses: List[float] = []
    correct = 0
    total = 0
    total_steps = len(loader)
    for step, batch_rows in enumerate(loader, start=1):
        if isinstance(batch_rows, dict):
            batch_rows = [
                {key: batch_rows[key][i] for key in batch_rows}
                for i in range(len(batch_rows["answer"]))
            ]
        loss_batch = _build_train_batch(processor, batch_rows, data_dir)
        loss_batch = _move_to_device(loss_batch, device)
        outputs = model(**loss_batch)
        losses.append(float(outputs.loss.item()))
        gen_batch = _build_eval_batch(processor, batch_rows, data_dir)
        prompt_lens = gen_batch["attention_mask"].sum(dim=1).tolist()
        gen_batch = _move_to_device(gen_batch, device)
        generated = model.generate(
            input_ids=gen_batch["input_ids"],
            attention_mask=gen_batch["attention_mask"],
            pixel_values=gen_batch["pixel_values"],
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
        for idx in range(generated.size(0)):
            prompt_len = int(prompt_lens[idx])
            text = processor.batch_decode(generated[idx : idx + 1, prompt_len:], skip_special_tokens=True)[0]
            pred = _extract_answer(text)
            target = str(batch_rows[idx]["answer"]).lower()
            correct += int(pred == target)
            total += 1
        _print_eval_progress("finetuned-eval", step, total_steps)
    return {"loss": float(sum(losses) / max(len(losses), 1)), "accuracy": correct / max(total, 1)}


def _print_progress(epoch: int, epochs: int, step: int, total_steps: int, loss: float) -> None:
    width = 24
    filled = int(width * step / max(total_steps, 1))
    bar = "#" * filled + "-" * (width - filled)
    sys.stdout.write(f"\repoch {epoch}/{epochs} [{bar}] {step}/{total_steps} batch_loss={loss:.4f}")
    sys.stdout.flush()
    if step >= total_steps:
        sys.stdout.write("\n")


@dataclass(frozen=True)
class TrainConfig:
    name: str
    batch_size: int
    lr: float
    weight_decay: float
    epochs: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float


def default_train_config() -> TrainConfig:
    return TrainConfig("A", batch_size=2, lr=1e-5, weight_decay=0.0, epochs=3, lora_rank=8, lora_alpha=16, lora_dropout=0.05)


def _build_lora_model(base_model: Any, cfg: TrainConfig) -> Any:
    _require_peft()
    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(base_model, peft_cfg)
    model.train()
    return model


def train_lora_adapter(
    model: Any,
    processor: Any,
    train_rows: Sequence[Dict[str, Any]],
    val_rows: Sequence[Dict[str, Any]],
    data_dir: str,
    cfg: TrainConfig,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Implement:
        Fine-tune the pretrained VLM with LoRA on the visible training split.

        Args:
          model (Any):
            Loaded pretrained VLM from build_model_and_processor(...).
          processor (Any):
            Matching processor for the VLM.
          train_rows (Sequence[dict[str, Any]]):
            Visible training examples.
          val_rows (Sequence[dict[str, Any]]):
            Visible validation examples.
          data_dir (str):
            Directory containing the image files referenced by the rows.
          cfg (TrainConfig):
            Training hyperparameters including batch size, learning rate, epochs,
            and LoRA settings.
          device (torch.device):
            Device on which fine-tuning runs.

        Requirements:
          - attach a LoRA adapter to the VLM using the hyperparameters in cfg
          - use the same prompt/image preprocessing path as the rest of the file
          - build training batches with labels that mask prompt tokens using -100
          - optimize only the LoRA parameters with AdamW
          - train for cfg.epochs epochs
          - after each epoch, evaluate on val_rows and record:
              train_loss, eval_loss, eval_accuracy
          - choose the best checkpoint by lowest eval_loss
          - break eval_loss ties by higher eval_accuracy
          - restore the best LoRA weights before returning
          - return the LoRA-wrapped model together with the metric history

        Returns:
          dict[str, Any]:
            Dictionary containing:
              - "model": fine-tuned LoRA-wrapped VLM
              - "history": list of per-epoch dicts with train_loss, eval_loss,
                and eval_accuracy
              - "eval_loss": best validation loss
              - "eval_accuracy": validation accuracy at the best checkpoint
    """
    lora_model = _build_lora_model(model, cfg)
    train_ds = VQARowDataset(train_rows)

    def collate(batch_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        return _build_train_batch(processor, batch_rows, data_dir)

    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW(lora_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_state = None
    best = {"eval_loss": float("inf"), "eval_accuracy": float("-inf")}
    history: List[Dict[str, float]] = []
    for epoch in range(cfg.epochs):
        lora_model.train()
        train_losses: List[float] = []
        total_steps = len(loader)
        for step, batch in enumerate(loader, start=1):
            batch = _move_to_device(batch, device)
            outputs = lora_model(**batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
            _print_progress(epoch + 1, cfg.epochs, step, total_steps, float(loss.item()))
        print(f"running validation for epoch {epoch + 1}/{cfg.epochs}...")
        metrics = evaluate_finetuned(lora_model, processor, val_rows, data_dir, device, batch_size=cfg.batch_size)
        train_loss = float(sum(train_losses) / max(len(train_losses), 1))
        print(
            f"finished epoch {epoch + 1}/{cfg.epochs}: "
            f"train_loss={train_loss:.4f} val_loss={metrics['loss']:.4f} val_acc={metrics['accuracy']:.4f}"
        )
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": train_loss,
                "eval_loss": float(metrics["loss"]),
                "eval_accuracy": float(metrics["accuracy"]),
            }
        )
        current = {"eval_loss": float(metrics["loss"]), "eval_accuracy": float(metrics["accuracy"])}
        if (current["eval_loss"], -current["eval_accuracy"]) < (best["eval_loss"], -best["eval_accuracy"]):
            best = current
            best_state = {k: v.detach().cpu().clone() for k, v in lora_model.state_dict().items() if "lora_" in k}
    if best_state is None:
        raise RuntimeError("LoRA training produced no checkpoint")
    lora_model.load_state_dict(best_state, strict=False)
    return {"model": lora_model, "history": history, "eval_loss": best["eval_loss"], "eval_accuracy": best["eval_accuracy"]}


def save_checkpoint(path: str, model: Any, cfg: TrainConfig) -> None:
    state = {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_name": MODEL_NAME,
            "answer_vocab": ANSWER_VOCAB,
            "adapter_type": "lora",
            "adapter_state": state,
            "cfg": {
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "epochs": cfg.epochs,
                "lora_rank": cfg.lora_rank,
                "lora_alpha": cfg.lora_alpha,
                "lora_dropout": cfg.lora_dropout,
            },
        },
        path,
    )


def load_checkpoint(path: str, device: torch.device) -> Tuple[Any, Any, Dict[str, Any]]:
    ckpt = torch.load(path, map_location="cpu")
    base_model, processor = build_model_and_processor(str(ckpt["model_name"]), device)
    cfg = TrainConfig("A", **ckpt["cfg"])
    model = _build_lora_model(base_model, cfg)
    adapter_state = ckpt["adapter_state"] if "adapter_state" in ckpt else ckpt["lora_state"]
    model.load_state_dict(adapter_state, strict=False)
    model.to(device)
    model.eval()
    return model, processor, ckpt


def latex_table(zero_shot_metrics: Dict[str, float], history: Sequence[Dict[str, float]], final_metrics: Dict[str, float]) -> str:
    lines = []
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\hline")
    lines.append(r"Method & Config & Train loss & Val loss & Val acc \\")
    lines.append(r"\hline")
    lines.append(f"Zero-shot VLM & -- & -- & {zero_shot_metrics['loss']:.4f} & {zero_shot_metrics['accuracy']:.4f} \\\\")
    for row in history:
        lines.append(
            f"LoRA VLM & epoch-{int(row['epoch'])} & {row['train_loss']:.4f} & {row['eval_loss']:.4f} & {row['eval_accuracy']:.4f} \\\\"
        )
    final_train_loss = history[-1]["train_loss"] if history else float("nan")
    lines.append(f"LoRA VLM & final-train & {final_train_loss:.4f} & {final_metrics['eval_loss']:.4f} & {final_metrics['eval_accuracy']:.4f} \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def main() -> None:
    _require_runtime()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--checkpoint", default="outputs/vlm_model.pt")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_rows = load_jsonl(os.path.join(args.data_dir, "train.jsonl"))
    val_rows = load_jsonl(os.path.join(args.data_dir, "val.jsonl"))

    if args.mode == "train":
        model, processor = build_model_and_processor(MODEL_NAME, device)
        zero_shot_metrics = evaluate_zero_shot(model, processor, val_rows, args.data_dir, device)
        print(f"val-zeroshot: loss={zero_shot_metrics['loss']:.4f} acc={zero_shot_metrics['accuracy']:.4f}")
        cfg = default_train_config()
        result = train_lora_adapter(model, processor, train_rows, val_rows, args.data_dir, cfg, device)
        for row in result["history"]:
            print(
                f"epoch-{int(row['epoch'])}: train_loss={row['train_loss']:.4f} "
                f"val_loss={row['eval_loss']:.4f} val_acc={row['eval_accuracy']:.4f}"
            )
        print(f"val-final-train: val_loss={result['eval_loss']:.4f} val_acc={result['eval_accuracy']:.4f}")
        save_checkpoint(args.checkpoint, result["model"], cfg)
        print(f"saved lora checkpoint: {args.checkpoint}")
        save_vqa_examples(train_rows, "outputs/vlm_train_examples.png")
        print("saved examples: outputs/vlm_train_examples.png")
        print("\n=== LaTeX (copy-paste) ===")
        print(r"\begin{solution}")
        print(latex_table(zero_shot_metrics, result["history"], result))
        print("")
        print(r"\end{solution}")
    else:
        if os.path.exists(os.path.join(args.data_dir, "test.jsonl")):
            eval_name = "test"
            eval_rows = load_jsonl(os.path.join(args.data_dir, "test.jsonl"))
        else:
            eval_name = "val"
            eval_rows = val_rows
        base_model, base_processor = build_model_and_processor(MODEL_NAME, device)
        zero_shot_metrics = evaluate_zero_shot(base_model, base_processor, eval_rows, args.data_dir, device)
        print(f"{eval_name}-zeroshot: loss={zero_shot_metrics['loss']:.4f} acc={zero_shot_metrics['accuracy']:.4f}")
        tuned_model, tuned_processor, _ = load_checkpoint(args.checkpoint, device)
        final_metrics = evaluate_finetuned(tuned_model, tuned_processor, eval_rows, args.data_dir, device, batch_size=default_train_config().batch_size)
        print(f"{eval_name}-final-train: loss={final_metrics['loss']:.4f} acc={final_metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
