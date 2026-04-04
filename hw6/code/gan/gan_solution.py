#!/usr/bin/env python3
"""
Class-conditioned GAN in the 16D latent space of a staff-trained MNIST VAE.

The intended workflow is:
  - load the provided latent-space train split
  - train a simple class-conditioned generator and discriminator in latent space
  - save the generator checkpoint
  - decode one generated latent per class with the provided VAE decoder
  - save 10,000 generated latent points for distribution matching
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


NUM_CLASSES = 10
LATENT_DIM = 16
NOISE_DIM = 16


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_latent_split(path: str) -> Dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if "latents" not in payload or "labels" not in payload:
        raise ValueError(f"expected 'latents' and 'labels' in {path}")
    return {
        "latents": torch.as_tensor(payload["latents"], dtype=torch.float32),
        "labels": torch.as_tensor(payload["labels"], dtype=torch.long),
    }


class Encoder(nn.Module):
    def __init__(self, z_dim: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
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
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
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


def load_decoder(path: str, device: torch.device) -> Decoder:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = payload["config"]
    model = VAE(z_dim=int(config["latent_dim"]))
    model.load_state_dict(payload["model_state_dict"])
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model.decoder


class LatentDataset(Dataset):
    def __init__(self, latents: torch.Tensor, labels: torch.Tensor) -> None:
        self.latents = latents
        self.labels = labels

    def __len__(self) -> int:
        return int(self.latents.size(0))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "latents": self.latents[idx],
            "labels": self.labels[idx],
        }


def collate_latents(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "latents": torch.stack([item["latents"] for item in batch], dim=0),
        "labels": torch.stack([item["labels"] for item in batch], dim=0),
    }


class Generator(nn.Module):
    def __init__(self, noise_dim: int, latent_dim: int, num_classes: int, hidden_dim: int) -> None:
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, 32)
        self.net = nn.Sequential(
            nn.Linear(noise_dim + 32, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_features = self.label_embed(labels)
        return self.net(torch.cat([noise, label_features], dim=1))


class Discriminator(nn.Module):
    def __init__(self, latent_dim: int, num_classes: int, hidden_dim: int) -> None:
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, 32)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 32, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, latents: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_features = self.label_embed(labels)
        return self.net(torch.cat([latents, label_features], dim=1)).squeeze(1)


def sample_noise(batch_size: int, noise_dim: int, device: torch.device, seed: int | None = None) -> torch.Tensor:
    if seed is None:
        return torch.randn(batch_size, noise_dim, device=device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(batch_size, noise_dim, generator=gen).to(device)


def build_gan_models(device: torch.device, hidden_dim: int = 128) -> Tuple[Generator, Discriminator]:
    """
    Implement:
        Build the class-conditioned latent-space generator and discriminator,
        move them to the requested device, and return them.

    Args:
        device:
            Torch device where both models should live.
        hidden_dim:
            Hidden width for the MLP layers inside the generator and
            discriminator.

    Returns:
        A tuple `(generator, discriminator)` where:
          - `generator` maps `(noise, labels)` to fake latent vectors
          - `discriminator` maps `(latents, labels)` to real/fake logits

    Notes:
        - Use the provided `Generator` and `Discriminator` classes.
        - The generator should use `NOISE_DIM` input noise and produce
          `LATENT_DIM` outputs.
        - The discriminator should score latent vectors of size `LATENT_DIM`.
    """
    generator = Generator(
        noise_dim=NOISE_DIM,
        latent_dim=LATENT_DIM,
        num_classes=NUM_CLASSES,
        hidden_dim=hidden_dim,
    ).to(device)
    discriminator = Discriminator(
        latent_dim=LATENT_DIM,
        num_classes=NUM_CLASSES,
        hidden_dim=hidden_dim,
    ).to(device)
    return generator, discriminator


def generator_step_loss(
    generator: Generator,
    discriminator: Discriminator,
    labels: torch.Tensor,
    noise: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Implement:
        Generate class-conditioned fake latents and compute the non-saturating
        generator loss against the discriminator.

    Args:
        generator:
            The class-conditioned latent generator.
        discriminator:
            The class-conditioned discriminator.
        labels:
            Tensor of digit labels with shape `(B,)`.
        noise:
            Noise tensor with shape `(B, NOISE_DIM)`.

    Returns:
        A tuple `(loss, metrics)` where:
          - `loss` is a scalar tensor used for backpropagation
          - `metrics` is a dict of Python floats for logging

    Required behavior:
        - Generate fake latents from `(noise, labels)`.
        - Score them with the discriminator using the same labels.
        - Use the non-saturating GAN generator loss, meaning fake logits
          should be pushed toward the real label (`1`).

    Expected metric keys:
        - `g_loss`
        - `g_logit_mean`
    """
    fake_latents = generator(noise, labels)
    fake_logits = discriminator(fake_latents, labels)
    loss = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
    return loss, {
        "g_loss": float(loss.item()),
        "g_logit_mean": float(fake_logits.mean().item()),
    }


def discriminator_step_loss(
    generator: Generator,
    discriminator: Discriminator,
    real_latents: torch.Tensor,
    labels: torch.Tensor,
    noise: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Implement:
        Compute the discriminator logistic loss on real latents from the data
        split and fake latents sampled from the generator.

    Args:
        generator:
            The class-conditioned latent generator.
        discriminator:
            The class-conditioned discriminator.
        real_latents:
            Real latent tensor of shape `(B, LATENT_DIM)` from the dataset.
        labels:
            Class labels for the real batch, shape `(B,)`.
        noise:
            Noise tensor with shape `(B, NOISE_DIM)` used to create fake
            latents with the same class labels.

    Returns:
        A tuple `(loss, metrics)` where:
          - `loss` is a scalar tensor used for backpropagation
          - `metrics` is a dict of Python floats for logging

    Required behavior:
        - Sample fake latents from the generator using `(noise, labels)`.
        - Treat real logits as target `1` and fake logits as target `0`.
        - Add the real and fake BCE-with-logits losses.
        - Do not backpropagate into the generator during the discriminator
          loss computation.

    Expected metric keys:
        - `d_loss`
        - `d_real_logit_mean`
        - `d_fake_logit_mean`
    """
    with torch.no_grad():
        fake_latents = generator(noise, labels)
    real_logits = discriminator(real_latents, labels)
    fake_logits = discriminator(fake_latents, labels)
    real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
    fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    loss = real_loss + fake_loss
    return loss, {
        "d_loss": float(loss.item()),
        "d_real_logit_mean": float(real_logits.mean().item()),
        "d_fake_logit_mean": float(fake_logits.mean().item()),
    }


def train_gan_batch(
    generator: Generator,
    discriminator: Discriminator,
    d_opt: torch.optim.Optimizer,
    g_opt: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, float]:
    """
    Implement:
        Run one alternating GAN training step on a batch:
        first update the discriminator on real and fake latent points,
        then update the generator against the discriminator.
        Return scalar logging metrics for the batch.

    Args:
        generator:
            The class-conditioned latent generator.
        discriminator:
            The class-conditioned discriminator.
        d_opt:
            Optimizer for the discriminator parameters.
        g_opt:
            Optimizer for the generator parameters.
        batch:
            Dictionary from the dataloader with keys:
              - `latents`: tensor of shape `(B, LATENT_DIM)`
              - `labels`: tensor of shape `(B,)`
        device:
            Torch device used for training.

    Returns:
        A dict of Python floats containing batch logging metrics.

    Required behavior:
        - Move `batch["latents"]` and `batch["labels"]` to `device`.
        - Sample fresh noise for the discriminator step.
        - Compute discriminator loss with `discriminator_step_loss(...)`,
          zero discriminator gradients, backpropagate, and step `d_opt`.
        - Sample fresh noise for the generator step.
        - Compute generator loss with `generator_step_loss(...)`,
          zero generator gradients, backpropagate, and step `g_opt`.
        - Return at least:
          `d_loss`, `g_loss`, `d_real_logit_mean`, `d_fake_logit_mean`,
          and `g_logit_mean`.
    """
    real_latents = batch["latents"].to(device)
    labels = batch["labels"].to(device)

    d_noise = sample_noise(labels.size(0), NOISE_DIM, device)
    d_loss, d_metrics = discriminator_step_loss(generator, discriminator, real_latents, labels, d_noise)
    d_opt.zero_grad(set_to_none=True)
    d_loss.backward()
    d_opt.step()

    g_noise = sample_noise(labels.size(0), NOISE_DIM, device)
    g_loss, g_metrics = generator_step_loss(generator, discriminator, labels, g_noise)
    g_opt.zero_grad(set_to_none=True)
    g_loss.backward()
    g_opt.step()

    return {
        "d_loss": float(d_loss.item()),
        "g_loss": float(g_loss.item()),
        "d_real_logit_mean": d_metrics["d_real_logit_mean"],
        "d_fake_logit_mean": d_metrics["d_fake_logit_mean"],
        "g_logit_mean": g_metrics["g_logit_mean"],
    }


def covariance(x: torch.Tensor) -> torch.Tensor:
    x_centered = x - x.mean(dim=0, keepdim=True)
    denom = max(int(x_centered.size(0)) - 1, 1)
    return x_centered.T @ x_centered / denom


def sqrtm_psd(matrix: torch.Tensor) -> torch.Tensor:
    evals, evecs = torch.linalg.eigh(matrix)
    evals = torch.clamp(evals, min=0.0)
    return (evecs * torch.sqrt(evals).unsqueeze(0)) @ evecs.T


def frechet_distance(real_latents: torch.Tensor, fake_latents: torch.Tensor) -> float:
    mu_r = real_latents.mean(dim=0)
    mu_f = fake_latents.mean(dim=0)
    cov_r = covariance(real_latents)
    cov_f = covariance(fake_latents)
    cov_r_sqrt = sqrtm_psd(cov_r)
    middle = cov_r_sqrt @ cov_f @ cov_r_sqrt
    middle = 0.5 * (middle + middle.T)
    cov_prod_sqrt = sqrtm_psd(middle)
    mean_term = (mu_r - mu_f).square().sum()
    trace_term = torch.trace(cov_r + cov_f - 2.0 * cov_prod_sqrt)
    value = mean_term + trace_term
    return float(torch.clamp(value, min=0.0).item())


@torch.no_grad()
def sample_conditioned_latents(
    generator: Generator,
    class_counts: Sequence[int],
    device: torch.device,
    seed_offset: int,
) -> Dict[str, torch.Tensor]:
    latents = []
    labels = []
    for class_id, count in enumerate(class_counts):
        if count <= 0:
            continue
        class_labels = torch.full((count,), class_id, dtype=torch.long, device=device)
        noise = sample_noise(count, NOISE_DIM, device, seed=seed_offset + class_id)
        class_latents = generator(noise, class_labels)
        latents.append(class_latents.cpu())
        labels.append(class_labels.cpu())
    return {
        "latents": torch.cat(latents, dim=0),
        "labels": torch.cat(labels, dim=0),
    }


@dataclass(frozen=True)
class GanConfig:
    hidden_dim: int
    lr_g: float
    lr_d: float
    epochs: int


def default_config() -> GanConfig:
    return GanConfig(hidden_dim=128, lr_g=1.5e-3, lr_d=1.5e-3, epochs=24)


def train_gan(
    generator: Generator,
    discriminator: Discriminator,
    train_ds: LatentDataset,
    cfg: GanConfig,
    batch_size: int,
    device: torch.device,
    run_name: str = "train",
) -> Tuple[Generator, Dict[str, float]]:
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_latents)
    g_opt = torch.optim.Adam(generator.parameters(), lr=cfg.lr_g, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=cfg.lr_d, betas=(0.5, 0.999))
    last_metrics: Dict[str, float] | None = None

    for epoch in range(cfg.epochs):
        generator.train()
        discriminator.train()
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        num_steps = 0
        if tqdm is not None:
            batch_iter = tqdm(
                loader,
                desc=f"{run_name} epoch {epoch + 1:02d}/{cfg.epochs:02d}",
                leave=False,
            )
        else:
            batch_iter = loader

        for batch in batch_iter:
            batch_metrics = train_gan_batch(
                generator=generator,
                discriminator=discriminator,
                d_opt=d_opt,
                g_opt=g_opt,
                batch=batch,
                device=device,
            )

            epoch_d_loss += batch_metrics["d_loss"]
            epoch_g_loss += batch_metrics["g_loss"]
            num_steps += 1
            avg_d_loss = epoch_d_loss / num_steps
            avg_g_loss = epoch_g_loss / num_steps
            if tqdm is not None:
                batch_iter.set_postfix(
                    d_loss=f"{avg_d_loss:.4f}",
                    g_loss=f"{avg_g_loss:.4f}",
                )

        avg_d_loss = epoch_d_loss / max(num_steps, 1)
        avg_g_loss = epoch_g_loss / max(num_steps, 1)
        last_metrics = {
            "d_loss": avg_d_loss,
            "g_loss": avg_g_loss,
        }
        print(
            f"[{run_name}] epoch {epoch + 1:02d}/{cfg.epochs:02d} "
            f"d_loss={avg_d_loss:.4f} g_loss={avg_g_loss:.4f}",
            flush=True,
        )
    if last_metrics is None:
        raise RuntimeError("GAN training did not produce any metrics")
    return generator, last_metrics


def save_checkpoint(path: str, generator: Generator, cfg: GanConfig, metrics: Dict[str, float]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "generator": {k: v.detach().cpu() for k, v in generator.state_dict().items()},
            "config": {
                "hidden_dim": cfg.hidden_dim,
                "lr_g": cfg.lr_g,
                "lr_d": cfg.lr_d,
                "epochs": cfg.epochs,
            },
            "metrics": metrics,
            "latent_dim": LATENT_DIM,
            "noise_dim": NOISE_DIM,
            "num_classes": NUM_CLASSES,
        },
        path,
    )


def load_checkpoint(path: str, device: torch.device) -> Tuple[Generator, Dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    hidden_dim = int(payload["config"]["hidden_dim"])
    generator, _ = build_gan_models(device=device, hidden_dim=hidden_dim)
    generator.load_state_dict(payload["generator"])
    generator.eval()
    return generator, payload


def sample_fixed_class_counts(samples_per_class: int) -> List[int]:
    return [samples_per_class for _ in range(NUM_CLASSES)]


def stratified_subset(dataset: LatentDataset, samples_per_class: int, seed: int) -> Dict[str, torch.Tensor]:
    rng = random.Random(seed)
    latents = []
    labels = []
    for class_id in range(NUM_CLASSES):
        class_indices = torch.nonzero(dataset.labels == class_id, as_tuple=False).flatten().tolist()
        if len(class_indices) < samples_per_class:
            raise ValueError(
                f"class {class_id} only has {len(class_indices)} examples, need {samples_per_class}"
            )
        chosen = rng.sample(class_indices, samples_per_class)
        latents.append(dataset.latents[chosen])
        labels.append(dataset.labels[chosen])
    return {
        "latents": torch.cat(latents, dim=0),
        "labels": torch.cat(labels, dim=0),
    }


def latent_match_score(reference: torch.Tensor, candidate: torch.Tensor) -> Dict[str, float]:
    avg_frechet = frechet_distance(reference, candidate)
    return {
        "avg_frechet": avg_frechet,
    }


@torch.no_grad()
def save_generated_outputs(
    decoder: Decoder,
    output_latents: Dict[str, torch.Tensor],
    output_dir: str,
    device: torch.device,
    latent_file_name: str,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)

    first_indices = []
    for class_id in range(NUM_CLASSES):
        class_positions = torch.nonzero(output_latents["labels"] == class_id, as_tuple=False).flatten()
        if class_positions.numel() == 0:
            raise ValueError(f"missing generated samples for class {class_id}")
        first_indices.append(int(class_positions[0].item()))
    report_payload = {
        "latents": torch.stack(
            [output_latents["latents"][idx] for idx in first_indices],
            dim=0,
        ),
        "labels": torch.arange(NUM_CLASSES, dtype=torch.long),
    }
    report_logits = decoder(report_payload["latents"].to(device))
    report_images = torch.sigmoid(report_logits).cpu()
    torchvision.utils.save_image(report_images, os.path.join(output_dir, "gan_samples.png"), nrow=5, pad_value=1.0)

    torch.save(output_latents, os.path.join(output_dir, latent_file_name))

    return {
        "report_labels": report_payload["labels"],
        "generated_shape": tuple(output_latents["latents"].shape),
    }


def make_dataset(payload: Dict[str, torch.Tensor]) -> LatentDataset:
    return LatentDataset(payload["latents"], payload["labels"])


def print_latex_report_train(metrics: Dict[str, float]) -> None:
    print("latex-train-row:")
    print(f"24 & {metrics['d_loss']:.4f} & {metrics['g_loss']:.4f} \\\\")


def print_latex_report_test(train_vs_test: Dict[str, float], generated_vs_test: Dict[str, float]) -> None:
    print("latex-test-rows:")
    print(f"train-vs-test: avg_frechet={train_vs_test['avg_frechet']:.4f}")
    print(f"generated-vs-test: avg_frechet={generated_vs_test['avg_frechet']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--data_dir", default=os.path.join(os.path.dirname(__file__), "data"))
    parser.add_argument("--checkpoint_path", default="outputs/gan_model.pt")
    parser.add_argument("--latent_output_name", default="gan_generated_latents.pt")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--samples_per_class", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = load_decoder(os.path.join(args.data_dir, "mnist_vae.pt"), device)
    cfg = default_config()

    if args.mode == "train":
        train_payload = load_latent_split(os.path.join(args.data_dir, "train_latents.pt"))
        train_ds = make_dataset(train_payload)

        print(
            f"[config] hidden_dim={cfg.hidden_dim} lr_g={cfg.lr_g} lr_d={cfg.lr_d} "
            f"epochs={cfg.epochs} samples_per_class={args.samples_per_class}",
            flush=True,
        )
        generator, discriminator = build_gan_models(device=device, hidden_dim=cfg.hidden_dim)
        generator, metrics = train_gan(
            generator,
            discriminator,
            train_ds,
            cfg,
            args.batch_size,
            device,
            run_name="train",
        )
        save_checkpoint(args.checkpoint_path, generator, cfg, metrics)
        generated_latents = sample_conditioned_latents(
            generator,
            sample_fixed_class_counts(args.samples_per_class),
            device,
            seed_offset=60_000,
        )
        output_info = save_generated_outputs(decoder, generated_latents, "outputs", device, args.latent_output_name)
        print("final-train-metrics:", metrics)
        print_latex_report_train(metrics)
        print("saved samples:", output_info)
    else:
        train_payload = load_latent_split(os.path.join(args.data_dir, "train_latents.pt"))
        train_ds = make_dataset(train_payload)
        generator, payload = load_checkpoint(args.checkpoint_path, device)
        generated_latents = sample_conditioned_latents(
            generator,
            sample_fixed_class_counts(args.samples_per_class),
            device,
            seed_offset=60_000,
        )
        output_info = save_generated_outputs(decoder, generated_latents, "outputs", device, args.latent_output_name)
        print("loaded config:", payload["config"])
        test_path = os.path.join(args.data_dir, "test_latents.pt")
        if os.path.exists(test_path):
            test_payload = load_latent_split(test_path)
            test_ds = make_dataset(test_payload)
            baseline_latents = stratified_subset(train_ds, args.samples_per_class, seed=args.seed + 123)
            train_vs_test = latent_match_score(test_ds.latents, baseline_latents["latents"])
            generated_vs_test = latent_match_score(test_ds.latents, generated_latents["latents"])
            print("train-vs-test:", train_vs_test)
            print("generated-vs-test:", generated_vs_test)
            print_latex_report_test(train_vs_test, generated_vs_test)
        else:
            print(f"hidden test split not available locally: {test_path}")
            print("skipped train-vs-test and generated-vs-test scoring")
        print("saved samples:", output_info)


if __name__ == "__main__":
    main()
