#!/usr/bin/env python3
"""
Train a small unconditional VAE on MNIST and export 16D latent datasets
for the Homework 6 GAN problem.

The intended staff workflow is:
  1. train the VAE on MNIST
  2. encode the visible MNIST training images into 16D latent means
  3. save the visible training latents for the GAN assignment
  4. encode a hidden MNIST test split for autograder evaluation
  5. save the VAE checkpoint so student code can decode generated latents
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


NUM_CLASSES = 10
IMAGE_SIZE = 28
LATENT_DIM = 16


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_mnist_splits(batch_size: int, dataset_root: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > 0.5).to(torch.float32)),
        ]
    )
    train_base = datasets.MNIST(root=dataset_root, train=True, download=True, transform=transform)
    test_base = datasets.MNIST(root=dataset_root, train=False, download=True, transform=transform)
    train_indices = list(range(0, 50_000))
    train_ds = Subset(train_base, train_indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    eval_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_base, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, eval_loader, test_loader


class Encoder(nn.Module):
    def __init__(self, z_dim: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),   # 28 -> 14
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 14 -> 7
            nn.SiLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
        )
        self.fc_mu = nn.Linear(128, z_dim)
        self.fc_logvar = nn.Linear(128, z_dim)

    def forward(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_img = self.conv(y).reshape(y.size(0), -1)
        h = self.fc(h_img)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, z_dim: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 64 * 7 * 7),
            nn.SiLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 7 -> 14
            nn.SiLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 14 -> 28
            nn.SiLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(z.size(0), 64, 7, 7)
        return self.deconv(h)


class VAE(nn.Module):
    def __init__(self, z_dim: int) -> None:
        super().__init__()
        self.encoder = Encoder(z_dim=z_dim)
        self.decoder = Decoder(z_dim=z_dim)
        self.z_dim = int(z_dim)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(y)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(z)
        return logits, mu, logvar


def vae_loss(logits: torch.Tensor, y: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float) -> Tuple[torch.Tensor, Dict[str, float]]:
    recon = F.binary_cross_entropy_with_logits(logits, y, reduction="none").flatten(1).sum(dim=1)
    kl = 0.5 * torch.sum(mu.square() + torch.exp(logvar) - logvar - 1.0, dim=1)
    loss = (recon + beta * kl).mean()
    return loss, {
        "loss": float(loss.item()),
        "recon": float(recon.mean().item()),
        "kl": float(kl.mean().item()),
    }


def run_epoch(
    model: VAE,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    beta: float,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
    total_n = 0
    for images, labels in loader:
        images = images.to(device)
        logits, mu, logvar = model(images)
        loss, stats = vae_loss(logits, images, mu, logvar, beta)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        batch_n = int(images.size(0))
        total_n += batch_n
        for key in totals:
            totals[key] += stats[key] * batch_n
    return {key: totals[key] / max(total_n, 1) for key in totals}


@torch.no_grad()
def encode_split(model: VAE, dataset: Dataset, device: torch.device, batch_size: int) -> Dict[str, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    all_latents = []
    all_labels = []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        mu, _ = model.encoder(images)
        all_latents.append(mu.cpu())
        all_labels.append(labels.cpu())
    return {
        "latents": torch.cat(all_latents, dim=0),
        "labels": torch.cat(all_labels, dim=0),
    }


def save_latent_split(path: str, payload: Dict[str, torch.Tensor]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def save_checkpoint(path: str, model: VAE, config: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
        },
        path,
    )


@dataclass(frozen=True)
class Args:
    batch_size: int
    epochs: int
    lr: float
    beta: float
    seed: int
    data_dir: str


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--data_dir", default=os.path.join(os.path.dirname(__file__), "data"))
    ns = parser.parse_args()
    return Args(
        batch_size=ns.batch_size,
        epochs=ns.epochs,
        lr=ns.lr,
        beta=ns.beta,
        seed=ns.seed,
        data_dir=ns.data_dir,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_root = os.path.join(args.data_dir, "mnist_raw")
    train_loader, eval_loader, test_loader = get_mnist_splits(batch_size=args.batch_size, dataset_root=dataset_root)
    model = VAE(z_dim=LATENT_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_state = None
    best_train = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(model, train_loader, device, optimizer, args.beta)
        print(
            f"[epoch {epoch:02d}] "
            f"train loss={train_stats['loss']:.4f} recon={train_stats['recon']:.4f} kl={train_stats['kl']:.4f}"
        )
        if train_stats["loss"] < best_train:
            best_train = train_stats["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("VAE training did not produce a checkpoint")
    model.load_state_dict(best_state)

    train_payload = encode_split(model, eval_loader.dataset, device, args.batch_size)
    test_payload = encode_split(model, test_loader.dataset, device, args.batch_size)

    save_latent_split(os.path.join(args.data_dir, "train_latents.pt"), train_payload)
    save_latent_split(os.path.join(args.data_dir, "test_latents.pt"), test_payload)
    save_checkpoint(
        os.path.join(args.data_dir, "mnist_vae.pt"),
        model,
        {
            "latent_dim": LATENT_DIM,
            "image_size": IMAGE_SIZE,
            "beta": args.beta,
            "best_train_loss": float(best_train),
        },
    )

    print(
        f"saved latents: train={train_payload['latents'].shape} "
        f"test={test_payload['latents'].shape}"
    )
    print(f"saved VAE checkpoint to {os.path.join(args.data_dir, 'mnist_vae.pt')}")


if __name__ == "__main__":
    main()
