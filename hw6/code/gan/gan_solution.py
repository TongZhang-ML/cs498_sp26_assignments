#!/usr/bin/env python3
"""
Controllable generation with a lightweight pretrained-StyleGAN stand-in.

This reference solution keeps the assignment offline and deterministic:
  - a frozen base generator maps latent seeds to small image tensors
  - a trainable controller injects an attribute-conditioned style offset
  - a frozen attribute scorer evaluates controllability
  - validation balances control accuracy and preservation of the base image

The structure mirrors the intended StyleGAN adaptation workflow while staying
small enough for local autograding.
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
from torch.utils.data import DataLoader, Dataset


ATTR_NAMES = ["red", "green", "blue", "striped"]
IMAGE_SIZE = 16
IMAGE_DIM = 3 * IMAGE_SIZE * IMAGE_SIZE
LATENT_DIM = 32
STYLE_DIM = 64


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


def latent_from_seed(seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(int(seed))
    return torch.randn(LATENT_DIM, generator=gen)


class FrozenGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(7)
        self.fc1 = nn.Linear(LATENT_DIM, STYLE_DIM)
        self.fc2 = nn.Linear(STYLE_DIM, IMAGE_DIM)
        with torch.no_grad():
            self.fc1.weight.copy_(0.20 * torch.randn(self.fc1.weight.shape, generator=gen))
            self.fc1.bias.copy_(0.05 * torch.randn(self.fc1.bias.shape, generator=gen))
            self.fc2.weight.copy_(0.10 * torch.randn(self.fc2.weight.shape, generator=gen))
            self.fc2.bias.copy_(0.05 * torch.randn(self.fc2.bias.shape, generator=gen))
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, z: torch.Tensor, style_delta: torch.Tensor | None = None) -> torch.Tensor:
        h = torch.tanh(self.fc1(z))
        if style_delta is not None:
            h = h + style_delta
        x = self.fc2(h).view(-1, 3, IMAGE_SIZE, IMAGE_SIZE)
        return torch.sigmoid(x)


class Controller(nn.Module):
    def __init__(self, num_attrs: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(num_attrs, STYLE_DIM)
        nn.init.zeros_(self.embed.weight)

    def forward(self, attr_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(attr_ids)


class FrozenAttributeScorer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        stripe = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE)
        stripe[:, :, ::2] = 1.0
        stripe[:, :, 1::2] = -1.0
        self.register_buffer("stripe_mask", stripe)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        red = images[:, 0].mean(dim=(1, 2)) - 0.5 * (images[:, 1].mean(dim=(1, 2)) + images[:, 2].mean(dim=(1, 2)))
        green = images[:, 1].mean(dim=(1, 2)) - 0.5 * (images[:, 0].mean(dim=(1, 2)) + images[:, 2].mean(dim=(1, 2)))
        blue = images[:, 2].mean(dim=(1, 2)) - 0.5 * (images[:, 0].mean(dim=(1, 2)) + images[:, 1].mean(dim=(1, 2)))
        striped = (images.mean(dim=1, keepdim=True) * self.stripe_mask).mean(dim=(1, 2, 3))
        return torch.stack([red, green, blue, striped], dim=-1)


class ControlDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        return {
            "id": str(row["id"]),
            "seed": int(row["seed"]),
            "target_attr": int(row["target_attr"]),
        }


def collate_rows(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "id": [b["id"] for b in batch],
        "seed": torch.tensor([b["seed"] for b in batch], dtype=torch.long),
        "target_attr": torch.tensor([b["target_attr"] for b in batch], dtype=torch.long),
    }


def build_generator_and_controllers(device: torch.device) -> Tuple[FrozenGenerator, Controller, FrozenAttributeScorer]:
    """
    Implement:
        Build the frozen pretrained generator, trainable control module,
        and frozen attribute scorer.
    """
    generator = FrozenGenerator().to(device)
    controller = Controller(num_attrs=len(ATTR_NAMES)).to(device)
    scorer = FrozenAttributeScorer().to(device)
    scorer.eval()
    return generator, controller, scorer


def sample_training_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Implement:
        Convert seeds to latent codes and move the training batch to device.
    """
    latents = torch.stack([latent_from_seed(int(seed)) for seed in batch["seed"].tolist()], dim=0)
    return {
        "latents": latents.to(device),
        "target_attr": batch["target_attr"].to(device),
    }


def attribute_control_loss(
    generator: FrozenGenerator,
    controller: Controller,
    scorer: FrozenAttributeScorer,
    train_batch: Dict[str, torch.Tensor],
    preservation_weight: float,
    reg_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Implement:
        Combine attribute classification loss, preservation loss, and controller regularization.
    """
    latents = train_batch["latents"]
    target_attr = train_batch["target_attr"]
    base_images = generator(latents)
    deltas = controller(target_attr)
    images = generator(latents, style_delta=deltas)
    logits = scorer(images)
    attr_loss = F.cross_entropy(logits, target_attr)
    preserve_loss = F.mse_loss(images, base_images)
    reg_loss = deltas.pow(2).mean()
    total = attr_loss + preservation_weight * preserve_loss + reg_weight * reg_loss
    metrics = {
        "loss": float(total.item()),
        "attr_loss": float(attr_loss.item()),
        "preserve_loss": float(preserve_loss.item()),
        "reg_loss": float(reg_loss.item()),
    }
    return total, metrics


@torch.no_grad()
def evaluate_controller(
    generator: FrozenGenerator,
    controller: Controller,
    scorer: FrozenAttributeScorer,
    dataset: Dataset,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_rows)
    controller.eval()
    total = 0
    correct = 0
    preserve_sum = 0.0
    loss_sum = 0.0
    for batch in loader:
        tb = sample_training_batch(batch, device)
        latents = tb["latents"]
        target_attr = tb["target_attr"]
        base_images = generator(latents)
        images = generator(latents, style_delta=controller(target_attr))
        logits = scorer(images)
        loss_sum += float(F.cross_entropy(logits, target_attr).item())
        correct += int((torch.argmax(logits, dim=-1) == target_attr).sum().item())
        preserve_sum += float(1.0 - F.mse_loss(images, base_images).item())
        total += int(target_attr.numel())
    attr_acc = correct / max(total, 1)
    preserve = preserve_sum / max(len(loader), 1)
    total_score = 0.65 * attr_acc + 0.35 * preserve
    return {
        "attr_acc": attr_acc,
        "preserve": preserve,
        "total_score": total_score,
        "loss": loss_sum / max(len(loader), 1),
    }


@dataclass(frozen=True)
class SweepConfig:
    name: str
    lr: float
    preservation_weight: float
    reg_weight: float
    epochs: int


def default_sweep() -> List[SweepConfig]:
    return [
        SweepConfig("A", lr=0.20, preservation_weight=2.0, reg_weight=0.010, epochs=20),
        SweepConfig("B", lr=0.15, preservation_weight=3.5, reg_weight=0.015, epochs=24),
        SweepConfig("C", lr=0.10, preservation_weight=5.0, reg_weight=0.020, epochs=28),
    ]


def train_controller(
    generator: FrozenGenerator,
    controller: Controller,
    scorer: FrozenAttributeScorer,
    train_ds: Dataset,
    val_ds: Dataset,
    cfg: SweepConfig,
    batch_size: int,
    device: torch.device,
) -> Tuple[Controller, Dict[str, float]]:
    """
    Implement:
        Train the controller and restore the best checkpoint by validation total score.
    """
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_rows)
    opt = torch.optim.Adam(controller.parameters(), lr=cfg.lr)
    best_state: Dict[str, torch.Tensor] | None = None
    best_metrics: Dict[str, float] | None = None
    best_key = -1e9
    for _ in range(cfg.epochs):
        controller.train()
        for batch in loader:
            tb = sample_training_batch(batch, device)
            loss, _ = attribute_control_loss(
                generator=generator,
                controller=controller,
                scorer=scorer,
                train_batch=tb,
                preservation_weight=cfg.preservation_weight,
                reg_weight=cfg.reg_weight,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
        metrics = evaluate_controller(generator, controller, scorer, val_ds, batch_size, device)
        key = metrics["total_score"]
        if key > best_key:
            best_key = key
            best_metrics = metrics
            best_state = {k: v.detach().cpu().clone() for k, v in controller.state_dict().items()}
    if best_state is None or best_metrics is None:
        raise RuntimeError("training did not produce a checkpoint")
    controller.load_state_dict(best_state)
    return controller, best_metrics


@torch.no_grad()
def generate_controlled_images(
    generator: FrozenGenerator,
    controller: Controller,
    scorer: FrozenAttributeScorer,
    seeds: Sequence[int],
    attr_ids: Sequence[int],
    device: torch.device,
) -> Dict[str, Any]:
    """
    Implement:
        Generate deterministic controlled outputs and record predicted scores.
    """
    latents = torch.stack([latent_from_seed(int(seed)) for seed in seeds], dim=0).to(device)
    attrs = torch.tensor(list(attr_ids), dtype=torch.long, device=device)
    images = generator(latents, style_delta=controller(attrs))
    logits = scorer(images)
    return {
        "images": images.cpu(),
        "pred_attr": torch.argmax(logits, dim=-1).cpu(),
        "scores": logits.cpu(),
    }


def save_checkpoint(path: str, controller: Controller, config_name: str, metrics: Dict[str, float]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "controller": {k: v.detach().cpu() for k, v in controller.state_dict().items()},
        "config_name": config_name,
        "metrics": metrics,
        "attr_names": ATTR_NAMES,
    }
    torch.save(payload, path)


def load_checkpoint(path: str, controller: Controller) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    controller.load_state_dict(payload["controller"])
    return payload


def select_best_config(train_ds: Dataset, val_ds: Dataset, batch_size: int, device: torch.device) -> Tuple[SweepConfig, Dict[str, Dict[str, float]]]:
    results: Dict[str, Dict[str, float]] = {}
    best_cfg: SweepConfig | None = None
    best_score = -1e9
    for cfg in default_sweep():
        generator, controller, scorer = build_generator_and_controllers(device)
        _, metrics = train_controller(generator, controller, scorer, train_ds, val_ds, cfg, batch_size, device)
        results[cfg.name] = metrics
        if metrics["total_score"] > best_score:
            best_score = metrics["total_score"]
            best_cfg = cfg
    if best_cfg is None:
        raise RuntimeError("no sweep config selected")
    return best_cfg, results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--data_dir", default=os.path.join(os.path.dirname(__file__), "data"))
    parser.add_argument("--checkpoint_path", default="outputs/gan_model.pt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "train":
        train_rows = load_jsonl(os.path.join(args.data_dir, "train.jsonl"))
        val_rows = load_jsonl(os.path.join(args.data_dir, "val.jsonl"))
        train_ds = ControlDataset(train_rows)
        val_ds = ControlDataset(val_rows)
        best_cfg, sweep_results = select_best_config(train_ds, val_ds, args.batch_size, device)

        merged_ds = ControlDataset(train_rows + val_rows)
        generator, controller, scorer = build_generator_and_controllers(device)
        controller, _ = train_controller(generator, controller, scorer, merged_ds, val_ds, best_cfg, args.batch_size, device)
        metrics = evaluate_controller(generator, controller, scorer, val_ds, args.batch_size, device)
        save_checkpoint(args.checkpoint_path, controller, best_cfg.name, metrics)
        print("validation sweep:")
        for name, result in sweep_results.items():
            print(name, result)
        print("best:", best_cfg.name, metrics)
    else:
        test_rows = load_jsonl(os.path.join(args.data_dir, "test.jsonl"))
        test_ds = ControlDataset(test_rows)
        generator, controller, scorer = build_generator_and_controllers(device)
        payload = load_checkpoint(args.checkpoint_path, controller)
        metrics = evaluate_controller(generator, controller.to(device), scorer, test_ds, args.batch_size, device)
        print("loaded config:", payload["config_name"])
        print("test:", metrics)


if __name__ == "__main__":
    main()
