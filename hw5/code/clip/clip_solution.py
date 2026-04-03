#!/usr/bin/env python3
"""
Pretrained CLIP for zero-shot and few-shot image classification.

This problem keeps the synthetic local image dataset, but uses a pretrained
CLIP model:
  - AutoProcessor.from_pretrained(...)
  - CLIPModel.from_pretrained(...)

Train mode behavior:
  - load a pretrained CLIP checkpoint
  - build zero-shot text prototypes from prompted class names
  - evaluate zero-shot accuracy on val
  - extract frozen CLIP image features
  - train few-shot linear probes with config A for the required shot counts
  - retrain a final probe on train+val
  - save a checkpoint containing the saved probes plus metadata
  - save labeled training examples for display

Test behavior:
  - load outputs/clip_model.pt
  - evaluate zero-shot CLIP on the available evaluation split
  - evaluate the saved few-shot probes for the requested shot counts
  - evaluate the saved final probe
  - report the average over the reported few-shot results
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, CLIPModel


CLASS_NAMES = [
    "red square",
    "blue circle",
    "green triangle",
    "yellow diamond",
    "purple plus",
    "orange x",
]

PROMPT_TEMPLATES = [
    "a photo of a {}",
    "an image of a {}",
    "a centered {}",
]

AVAILABLE_MODELS = {
    "clip-vit-base-patch32": "openai/clip-vit-base-patch32",
    "clip-vit-base-patch16": "openai/clip-vit-base-patch16",
    "openclip-vit-b-32": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
}

COLOR_RGB = {
    "red": (230, 46, 38),
    "blue": (38, 89, 230),
    "green": (38, 179, 64),
    "yellow": (242, 204, 38),
    "purple": (140, 64, 204),
    "orange": (242, 128, 26),
}

TEST_SHOT_VALUES = (1, 2, 4, 8)


def set_seed(seed: int) -> None:
    random.seed(seed)
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


def load_feature_split(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if "features" not in payload or "labels" not in payload:
        raise ValueError(f"expected keys 'features' and 'labels' in {path}")
    features = torch.as_tensor(payload["features"], dtype=torch.float32)
    labels = torch.as_tensor(payload["labels"], dtype=torch.long)
    return features, labels


def _triangle_mask(xx: torch.Tensor, yy: torch.Tensor, v0, v1, v2) -> torch.Tensor:
    x0, y0 = v0
    x1, y1 = v1
    x2, y2 = v2
    den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    a = ((y1 - y2) * (xx - x2) + (x2 - x1) * (yy - y2)) / den
    b = ((y2 - y0) * (xx - x2) + (x0 - x2) * (yy - y2)) / den
    c = 1.0 - a - b
    return (a >= 0) & (b >= 0) & (c >= 0)


def _shape_mask(shape: str, xx: torch.Tensor, yy: torch.Tensor, cx: float, cy: float, r: float) -> torch.Tensor:
    if shape == "square":
        return (torch.abs(xx - cx) <= r) & (torch.abs(yy - cy) <= r)
    if shape == "circle":
        return (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
    if shape == "triangle":
        return _triangle_mask(xx, yy, (cx, cy - r), (cx - r, cy + r), (cx + r, cy + r))
    if shape == "diamond":
        return torch.abs(xx - cx) + torch.abs(yy - cy) <= r
    if shape == "plus":
        t = max(4.0, r / 2.5)
        return ((torch.abs(xx - cx) <= t) & (torch.abs(yy - cy) <= r)) | ((torch.abs(yy - cy) <= t) & (torch.abs(xx - cx) <= r))
    if shape == "x":
        t = max(3.0, r / 3.0)
        mask = (torch.abs((yy - cy) - (xx - cx)) <= t) | (torch.abs((yy - cy) + (xx - cx)) <= t)
        return mask & (torch.abs(xx - cx) <= r) & (torch.abs(yy - cy) <= r)
    raise ValueError(f"unknown shape: {shape}")


def _blend_mask(image: torch.Tensor, mask: torch.Tensor, color: torch.Tensor, alpha: float) -> None:
    if not bool(mask.any()):
        return
    image[:, mask] = alpha * color[:, 0, 0].unsqueeze(1) + (1.0 - alpha) * image[:, mask]


def render_shape_image(record: Dict[str, Any], image_size: int = 128) -> Image.Image:
    seed = int(record["seed"])
    rng = random.Random(seed)
    gen = torch.Generator().manual_seed(seed)
    split = str(record.get("split", "train"))

    image = torch.full((3, image_size, image_size), 232.0, dtype=torch.float32)
    yy, xx = torch.meshgrid(torch.arange(image_size, dtype=torch.float32), torch.arange(image_size, dtype=torch.float32), indexing="ij")

    x_grad = (xx / max(image_size - 1, 1)).unsqueeze(0)
    y_grad = (yy / max(image_size - 1, 1)).unsqueeze(0)
    tint = torch.tensor([rng.uniform(-18, 18), rng.uniform(-18, 18), rng.uniform(-18, 18)], dtype=torch.float32).view(3, 1, 1)
    image = image + tint * (0.55 * x_grad + 0.45 * y_grad)
    image = image + 7.0 * torch.randn((3, image_size, image_size), generator=gen)

    if split == "train":
        distractor_count = rng.randint(1, 2)
        occluder_count = 0
    elif split == "val":
        distractor_count = rng.randint(2, 4)
        occluder_count = 1
    else:
        distractor_count = rng.randint(4, 6)
        occluder_count = rng.randint(1, 2)

    cx = image_size / 2 + rng.randint(-16, 16)
    cy = image_size / 2 + rng.randint(-16, 16)
    r = float(rng.randint(image_size // 8, image_size // 5))
    color = torch.tensor(COLOR_RGB[str(record["color"])], dtype=torch.float32).view(3, 1, 1)
    shape = str(record["shape"])

    palette = list(COLOR_RGB.values())
    shape_names = ["square", "circle", "triangle", "diamond", "plus", "x"]
    for _ in range(distractor_count):
        dcx = rng.randint(image_size // 8, image_size - image_size // 8)
        dcy = rng.randint(image_size // 8, image_size - image_size // 8)
        dr = float(rng.randint(image_size // 12, image_size // 7))
        dshape = rng.choice(shape_names)
        dcolor = torch.tensor(rng.choice(palette), dtype=torch.float32).view(3, 1, 1)
        dmask = _shape_mask(dshape, xx, yy, float(dcx), float(dcy), dr)
        _blend_mask(image, dmask, dcolor, alpha=0.38 if split == "train" else 0.46)

    mask = _shape_mask(shape, xx, yy, cx, cy, r)

    interior = torch.roll(mask, 1, 0) & torch.roll(mask, -1, 0) & torch.roll(mask, 1, 1) & torch.roll(mask, -1, 1)
    border = mask & (~interior)
    _blend_mask(image, mask, color, alpha=0.72)
    image[:, border] = 0.62 * image[:, border]

    for _ in range(occluder_count):
        x0 = rng.randint(0, image_size - image_size // 4)
        y0 = rng.randint(0, image_size - image_size // 4)
        w = rng.randint(image_size // 8, image_size // 3)
        h = rng.randint(image_size // 10, image_size // 4)
        occ_mask = (xx >= x0) & (xx <= x0 + w) & (yy >= y0) & (yy <= y0 + h)
        occ_color = torch.tensor([rng.randint(120, 210)] * 3, dtype=torch.float32).view(3, 1, 1)
        _blend_mask(image, occ_mask, occ_color, alpha=0.22 if split == "val" else 0.28)

    vignette = ((xx - image_size / 2) ** 2 + (yy - image_size / 2) ** 2) / (image_size * image_size / 3)
    image = image - vignette.unsqueeze(0) * (8.0 if split == "train" else 14.0)
    arr = image.clamp(0.0, 255.0).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)


class ShapeDataset(Dataset):
    def __init__(self, records: Sequence[Dict[str, Any]]) -> None:
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        return {
            "image": render_shape_image(record),
            "labels": int(record["label"]),
        }


def collate_images(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "images": [b["image"] for b in batch],
        "labels": torch.tensor([b["labels"] for b in batch], dtype=torch.long),
    }


def build_model_and_processor(model_name: str, device: torch.device) -> Tuple[AutoProcessor, CLIPModel]:
    """
    Implement:
        Load a pretrained CLIP processor and model.

        Args:
          model_name: Hugging Face model identifier for the CLIP checkpoint.
          device: Torch device where the CLIP model should be placed.

        Requirements:
          - use AutoProcessor.from_pretrained(model_name)
          - use CLIPModel.from_pretrained(model_name)
          - move the model to device
          - set eval mode before returning

        Returns:
          A tuple (processor, model) where processor is the loaded
          AutoProcessor and model is the CLIPModel on the requested device
          in eval mode.
    """
    processor = AutoProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return processor, model


@torch.no_grad()
def build_text_features(
    processor: AutoProcessor,
    model: CLIPModel,
    class_names: Sequence[str],
    prompt_templates: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    """
    Implement:
        Build one normalized zero-shot text feature per class.

        Args:
          processor: The CLIP processor used to tokenize text prompts.
          model: The pretrained CLIP model.
          class_names: Class names such as "red square" and "blue circle".
          prompt_templates: Prompt templates such as
            "a photo of a {}" and "an image of a {}".
            For each class name, create prompted text strings by filling
            each template with that class name.
          device: Torch device used for CLIP inference.

        Requirements:
          - create prompted text strings for each class by formatting
            every template in prompt_templates with the class name
          - tokenize with processor(..., padding=True, return_tensors='pt')
          - compute model.get_text_features(...)
          - normalize each template feature to unit length along dim=-1
          - average the normalized template features for each class
          - renormalize the averaged class prototype to unit length

        Returns:
          Tensor of shape [num_classes, hidden_dim], where num_classes is
          len(class_names) and hidden_dim is the CLIP text embedding size
          returned by model.get_text_features(...).
    """
    class_features: List[torch.Tensor] = []
    for class_name in class_names:
        prompts = [tmpl.format(class_name) for tmpl in prompt_templates]
        text_inputs = processor(text=prompts, padding=True, return_tensors="pt").to(device)
        text_features = model.get_text_features(**text_inputs)
        text_features = F.normalize(text_features, dim=-1)
        proto = F.normalize(text_features.mean(dim=0), dim=-1)
        class_features.append(proto)
    return torch.stack(class_features, dim=0)


@torch.no_grad()
def predict_zero_shot(
    processor: AutoProcessor,
    model: CLIPModel,
    image: Image.Image,
    text_features: torch.Tensor,
    class_names: Sequence[str],
    device: torch.device,
) -> int:
    """
    Helper:
        Predict one image with zero-shot CLIP.

        Args:
          processor: The CLIP processor used to preprocess the image.
          model: The pretrained CLIP model.
          image: One PIL image to classify.
          text_features: Tensor of class text prototypes with shape
            [num_classes, hidden_dim].
          class_names: Ordered class-name list corresponding to text_features.
          device: Torch device used for CLIP inference.

        Requirements:
          - preprocess image with processor(images=image, return_tensors='pt')
          - compute normalized image features with model.get_image_features(...)
          - compute logits using model.logit_scale.exp()
          - return the predicted class index

        Returns:
          The predicted class index as an integer in the range
          [0, len(class_names) - 1].
    """
    image_inputs = processor(images=image, return_tensors="pt").to(device)
    image_features = model.get_image_features(**image_inputs)
    image_features = F.normalize(image_features, dim=-1)
    logits = model.logit_scale.exp() * image_features @ text_features.t()
    return int(torch.argmax(logits, dim=-1).item())


@torch.no_grad()
def extract_image_features(
    processor: AutoProcessor,
    model: CLIPModel,
    dataset: Dataset,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Implement:
        Extract frozen CLIP image features for an entire dataset.

        Args:
          processor: The CLIP processor used to preprocess image batches.
          model: The pretrained CLIP model.
          dataset: Dataset returning image/label examples.
          batch_size: Batch size for the DataLoader.
          device: Torch device used for CLIP inference.

        Requirements:
          - iterate over a DataLoader
          - preprocess images with the processor
          - compute model.get_image_features(...)
          - normalize each image feature to unit length along dim=-1
          - return (features, labels)

        Returns:
          A tuple (features, labels) where:
            - features has shape [num_examples, hidden_dim]
            - labels has shape [num_examples]
          Here num_examples = len(dataset) and hidden_dim is the CLIP image
          embedding size returned by model.get_image_features(...).
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_images)
    all_features: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    for batch in loader:
        image_inputs = processor(images=batch["images"], return_tensors="pt").to(device)
        image_features = model.get_image_features(**image_inputs)
        image_features = F.normalize(image_features, dim=-1)
        all_features.append(image_features.cpu())
        all_labels.append(batch["labels"])
    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@dataclass(frozen=True)
class ProbeConfig:
    name: str
    lr: float
    weight_decay: float
    epochs: int


def default_probe_config() -> ProbeConfig:
    return ProbeConfig("A", lr=5e-2, weight_decay=0.0, epochs=40)


def sample_k_per_class(
    features: torch.Tensor,
    labels: torch.Tensor,
    shots_per_class: int,
    num_classes: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    chosen_indices: List[torch.Tensor] = []
    generator = torch.Generator().manual_seed(seed)
    for class_idx in range(num_classes):
        class_indices = torch.nonzero(labels == class_idx, as_tuple=False).squeeze(-1)
        if class_indices.numel() < shots_per_class:
            raise ValueError(f"class {class_idx} only has {class_indices.numel()} examples, need {shots_per_class}")
        perm = torch.randperm(class_indices.numel(), generator=generator)
        chosen_indices.append(class_indices[perm[:shots_per_class]])
    index_tensor = torch.cat(chosen_indices, dim=0)
    return features[index_tensor], labels[index_tensor]


@dataclass
class RunConfig:
    seed: int = 7
    batch_size: int = 16
    model_key: str = "clip-vit-base-patch32"
    checkpoint_path: str = "outputs/clip_model.pt"

    @property
    def model_name(self) -> str:
        return AVAILABLE_MODELS[self.model_key]


def train_linear_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    cfg: ProbeConfig,
    num_classes: int,
    device: torch.device,
) -> Tuple[LinearProbe, Dict[str, float]]:
    """
    Implement:
        Train a linear probe on top of frozen CLIP image features.

        Args:
          train_features: Training feature tensor with shape
            [num_train, hidden_dim].
          train_labels: Training labels with shape [num_train].
          val_features: Validation feature tensor with shape
            [num_val, hidden_dim].
          val_labels: Validation labels with shape [num_val].
          cfg: ProbeConfig containing the learning rate, weight decay,
            and number of training epochs.
          num_classes: Number of output classes for the classifier.
          device: Torch device used for training and evaluation.

        Requirements:
          - create a LinearProbe
          - optimize with AdamW using cfg.lr and cfg.weight_decay
          - train for cfg.epochs
          - use cross-entropy loss for training and validation
          - evaluate on the validation split after each epoch
          - keep the best model by validation loss
          - if two checkpoints have the same validation loss, break ties
            by higher validation accuracy
          - restore the best checkpoint before returning
          - return the best probe and its validation metrics

        Returns:
          A tuple (classifier, metrics) where:
            - classifier is the best LinearProbe found during training
            - metrics is a dictionary with keys:
                "eval_loss": best validation loss
                "eval_accuracy": corresponding validation accuracy
    """
    classifier = LinearProbe(train_features.size(1), num_classes).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    val_features = val_features.to(device)
    val_labels = val_labels.to(device)

    best_state = None
    best_metrics = {"eval_loss": float("inf"), "eval_accuracy": float("-inf")}

    for _ in range(cfg.epochs):
        classifier.train()
        logits = classifier(train_features)
        loss = F.cross_entropy(logits, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        classifier.eval()
        with torch.no_grad():
            val_logits = classifier(val_features)
            val_loss = float(F.cross_entropy(val_logits, val_labels).item())
            val_acc = float((torch.argmax(val_logits, dim=-1) == val_labels).float().mean().item())
        current = {"eval_loss": val_loss, "eval_accuracy": val_acc}
        if (current["eval_loss"], -current["eval_accuracy"]) < (best_metrics["eval_loss"], -best_metrics["eval_accuracy"]):
            best_metrics = current
            best_state = {k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()}

    assert best_state is not None
    classifier.load_state_dict(best_state)
    return classifier, best_metrics


@torch.no_grad()
def evaluate_zero_shot(
    processor: AutoProcessor,
    model: CLIPModel,
    dataset: Dataset,
    text_features: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_images)
    losses: List[float] = []
    correct = 0
    total = 0
    for batch in loader:
        image_inputs = processor(images=batch["images"], return_tensors="pt").to(device)
        image_features = model.get_image_features(**image_inputs)
        image_features = F.normalize(image_features, dim=-1)
        logits = model.logit_scale.exp() * image_features @ text_features.t()
        labels = batch["labels"].to(device)
        losses.append(float(F.cross_entropy(logits, labels).item()))
        correct += int((torch.argmax(logits, dim=-1) == labels).sum().item())
        total += labels.numel()
    return {"loss": float(sum(losses) / max(len(losses), 1)), "accuracy": correct / max(total, 1)}


@torch.no_grad()
def evaluate_probe(
    classifier: LinearProbe,
    features: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    classifier.eval()
    with torch.no_grad():
        logits = classifier(features.to(device))
        loss = float(F.cross_entropy(logits, labels.to(device)).item())
        acc = float((torch.argmax(logits, dim=-1) == labels.to(device)).float().mean().item())
    return {"loss": loss, "accuracy": acc}


@torch.no_grad()
def evaluate_zero_shot_features(
    image_features: torch.Tensor,
    labels: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    logits = logit_scale.to(device) * image_features.to(device) @ text_features.to(device).t()
    eval_labels = labels.to(device)
    loss = float(F.cross_entropy(logits, eval_labels).item())
    acc = float((torch.argmax(logits, dim=-1) == eval_labels).float().mean().item())
    return {"loss": loss, "accuracy": acc}


def save_labeled_examples(records: Sequence[Dict[str, Any]], out_path: str, num_examples: int = 12, cols: int = 4) -> None:
    if not records:
        return
    rows = list(records[:num_examples])
    image_size = 128
    label_height = 24
    gutter = 12
    cols = max(cols, 1)
    num_rows = (len(rows) + cols - 1) // cols
    canvas_w = cols * image_size + (cols + 1) * gutter
    canvas_h = num_rows * (image_size + label_height) + (num_rows + 1) * gutter

    canvas = Image.new("RGB", (canvas_w, canvas_h), (246, 246, 246))
    try:
        from PIL import ImageDraw
    except ImportError:
        return
    draw = ImageDraw.Draw(canvas)

    for idx, record in enumerate(rows):
        row = idx // cols
        col = idx % cols
        x = gutter + col * (image_size + gutter)
        y = gutter + row * (image_size + label_height + gutter)
        image = render_shape_image(record, image_size=image_size)
        canvas.paste(image, (x, y))
        draw.rectangle((x, y + image_size, x + image_size, y + image_size + label_height), fill=(252, 252, 252))
        draw.text((x + 4, y + image_size + 5), str(record["class_name"]), fill=(20, 20, 20))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)


def save_checkpoint(
    path: str,
    model_name: str,
    classifiers: Dict[str, LinearProbe],
    feature_dim: int,
    num_classes: int,
    shot_values: Sequence[int],
    final_shots: int,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "feature_dim": feature_dim,
            "num_classes": num_classes,
            "shot_values": list(shot_values),
            "final_shots": int(final_shots),
            "classifier_states": {name: classifier.state_dict() for name, classifier in classifiers.items()},
        },
        path,
    )


def load_checkpoint(path: str, device: torch.device) -> Tuple[Dict[str, Any], Dict[str, LinearProbe]]:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    classifiers: Dict[str, LinearProbe] = {}
    if "classifier_states" in ckpt:
        classifier_states = ckpt["classifier_states"]
    else:
        classifier_states = {"full": ckpt["classifier_state"]}
        ckpt["shot_values"] = []
        ckpt["final_shots"] = 0
    for name, state_dict in classifier_states.items():
        classifier = LinearProbe(int(ckpt["feature_dim"]), int(ckpt["num_classes"])).to(device)
        classifier.load_state_dict(state_dict)
        classifiers[str(name)] = classifier
    return ckpt, classifiers


def format_latex_table(
    zero_shot: Dict[str, float],
    fewshot_rows: Sequence[Tuple[int, Dict[str, float]]],
    final_shots: int,
    final_metrics: Dict[str, float],
) -> str:
    lines = []
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\hline")
    lines.append(r"Method & Config & Val loss & Val acc \\")
    lines.append(r"\hline")
    lines.append(f"Zero-shot CLIP & -- & {zero_shot['loss']:.4f} & {zero_shot['accuracy']:.4f} \\\\")
    for shots, metrics in fewshot_rows:
        lines.append(f"Linear probe & fewshot-{shots} & {metrics['eval_loss']:.4f} & {metrics['eval_accuracy']:.4f} \\\\")
    lines.append(f"Linear probe & final-fewshot-{final_shots} & {final_metrics['eval_loss']:.4f} & {final_metrics['eval_accuracy']:.4f} \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    args = parser.parse_args()

    cfg = RunConfig()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "data"
    checkpoint_path = cfg.checkpoint_path
    test_shots = list(TEST_SHOT_VALUES)

    train_records = load_jsonl(os.path.join(data_dir, "train.jsonl"))
    val_records = load_jsonl(os.path.join(data_dir, "val.jsonl"))
    test_path = os.path.join(data_dir, "test.jsonl")
    test_features_path = os.path.join(data_dir, "test_features.pt")
    test_records = load_jsonl(test_path) if os.path.exists(test_path) else None

    train_ds = ShapeDataset(train_records)
    val_ds = ShapeDataset(val_records)

    processor, model = build_model_and_processor(cfg.model_name, device)
    text_features = build_text_features(processor, model, CLASS_NAMES, PROMPT_TEMPLATES, device)

    if args.mode == "train":
        zero_shot_metrics = evaluate_zero_shot(processor, model, val_ds, text_features, cfg.batch_size, device)
        print(f"zeroshot: loss={zero_shot_metrics['loss']:.4f} acc={zero_shot_metrics['accuracy']:.4f}")

        train_features, train_labels = extract_image_features(processor, model, train_ds, cfg.batch_size, device)
        val_features, val_labels = extract_image_features(processor, model, val_ds, cfg.batch_size, device)

        probe_cfg = default_probe_config()
        max_train_per_class = min(int((train_labels == c).sum().item()) for c in range(len(CLASS_NAMES)))
        valid_shots = []
        for shots in test_shots:
            if shots <= 0:
                continue
            if shots > max_train_per_class:
                continue
            valid_shots.append(shots)

        saved_classifiers: Dict[str, LinearProbe] = {}
        fewshot_rows: List[Tuple[int, Dict[str, float]]] = []
        for shots in valid_shots:
            shot_features, shot_labels = sample_k_per_class(train_features, train_labels, shots, len(CLASS_NAMES), cfg.seed + shots)
            shot_classifier, shot_metrics = train_linear_probe(
                shot_features,
                shot_labels,
                val_features,
                val_labels,
                probe_cfg,
                len(CLASS_NAMES),
                device,
            )
            saved_classifiers[f"{shots}shot"] = shot_classifier
            print(f"fewshot-{shots}shot: val_loss={shot_metrics['eval_loss']:.4f} val_acc={shot_metrics['eval_accuracy']:.4f}")
            fewshot_rows.append((shots, shot_metrics))

        trainval_ds = ShapeDataset(train_records + val_records)
        trainval_features, trainval_labels = extract_image_features(processor, model, trainval_ds, cfg.batch_size, device)
        final_classifier, final_metrics = train_linear_probe(
            trainval_features,
            trainval_labels,
            val_features,
            val_labels,
            probe_cfg,
            len(CLASS_NAMES),
            device,
        )
        saved_classifiers["full"] = final_classifier

        save_checkpoint(
            checkpoint_path,
            cfg.model_name,
            saved_classifiers,
            int(train_features.size(1)),
            len(CLASS_NAMES),
            valid_shots,
            max_train_per_class,
        )
        print(f"final-fewshot-{max_train_per_class}: val_loss={final_metrics['eval_loss']:.4f} val_acc={final_metrics['eval_accuracy']:.4f}")
        save_labeled_examples(train_records, "outputs/train_examples_labeled.png")
        print("saved examples: outputs/train_examples_labeled.png")

        print("\n=== LaTeX (copy-paste) ===")
        print(r"\begin{solution}")
        print(format_latex_table(zero_shot_metrics, fewshot_rows, max_train_per_class, final_metrics))
        print(r"\end{solution}")
    else:
        ckpt, classifiers = load_checkpoint(checkpoint_path, device)
        if str(ckpt["model_name"]) != cfg.model_name:
            processor, model = build_model_and_processor(str(ckpt["model_name"]), device)
            text_features = build_text_features(processor, model, CLASS_NAMES, PROMPT_TEMPLATES, device)

        if os.path.exists(test_features_path):
            eval_name = "test"
            eval_features, eval_labels = load_feature_split(test_features_path)
        else:
            if test_records is not None:
                eval_name = "test"
                eval_ds = ShapeDataset(test_records)
            else:
                eval_name = "val"
                eval_ds = val_ds
            eval_features, eval_labels = extract_image_features(processor, model, eval_ds, cfg.batch_size, device)

        zero_shot_metrics = evaluate_zero_shot_features(
            eval_features,
            eval_labels,
            text_features,
            model.logit_scale.exp().detach(),
            device,
        )
        print(f"{eval_name}-zeroshot: loss={zero_shot_metrics['loss']:.4f} acc={zero_shot_metrics['accuracy']:.4f}")

        requested_shots = []
        for shots in test_shots:
            if shots <= 0:
                continue
            requested_shots.append(shots)

        reported_shot_metrics: List[Dict[str, float]] = []
        for shots in requested_shots:
            classifier_name = f"{shots}shot"
            if classifier_name not in classifiers:
                continue
            shot_metrics = evaluate_probe(classifiers[classifier_name], eval_features, eval_labels, device)
            reported_shot_metrics.append(shot_metrics)
            print(f"{eval_name}-fewshot-{shots}shot: loss={shot_metrics['loss']:.4f} acc={shot_metrics['accuracy']:.4f}")

        if "full" in classifiers:
            saved_metrics = evaluate_probe(classifiers["full"], eval_features, eval_labels, device)
            reported_shot_metrics.append(saved_metrics)
            final_shots = int(ckpt.get("final_shots", 0))
            if final_shots > 0:
                print(f"{eval_name}-fewshot-{final_shots}shot: loss={saved_metrics['loss']:.4f} acc={saved_metrics['accuracy']:.4f}")
            else:
                print(f"{eval_name}-fewshot-full: loss={saved_metrics['loss']:.4f} acc={saved_metrics['accuracy']:.4f}")

        if reported_shot_metrics:
            avg_loss = sum(m["loss"] for m in reported_shot_metrics) / len(reported_shot_metrics)
            avg_acc = sum(m["accuracy"] for m in reported_shot_metrics) / len(reported_shot_metrics)
            print(f"{eval_name}-fewshot-avg: loss={avg_loss:.4f} acc={avg_acc:.4f}")


if __name__ == "__main__":
    main()
