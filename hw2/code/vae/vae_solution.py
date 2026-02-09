################################################################
#
# Conditional VAE on binarized MNIST with Bernoulli likelihood.
#
# Please refer to the code provided with the lecture notes as a related reference implementation.
#
# This script models
#  - Binarized MNIST images (pixels in {0,1})
#  - use Bernoulli decoder likelihood with binary cross-entropy (BCE) loss
#
# Data files
# -----
# This script expects torchvision to download MNIST automatically.
# 
################################################################

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import argparse
import os
import random
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as tv
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms



# ----------------------------
# Reproducibility
# ----------------------------

def set_seed(seed: int) -> None:
    """Set seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------
# Data
# ----------------------------
class RemappedSubset(torch.utils.data.Dataset):
    def __init__(self, base, indices, label_map):
        self.base = base
        self.indices = indices
        self.label_map = label_map  # dict: original_label -> 0..K-1

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, y = self.base[self.indices[i]]
        y = int(y)
        return img, self.label_map[y]

def filter_classes(dataset: datasets.MNIST, class_indices: Tuple[int, ...]) -> Tuple[torch.utils.data.Dataset, int]:
    """
    Return (subset_dataset, num_classes) where labels are remapped to 0..K-1
    in the order given by class_indices.
    """
    class_indices = tuple(int(c) for c in class_indices)
    label_map = {c: i for i, c in enumerate(class_indices)}  # original -> remapped
    idx = [i for i in range(len(dataset)) if int(dataset.targets[i]) in label_map]
    return RemappedSubset(dataset, idx, label_map), len(class_indices)

def get_mnist_dataloaders(
    *,
    batch_size: int,
    binary_pixels: bool = True,
    classes: Optional[Tuple[int, ...]] = None,
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Load MNIST and return (train_loader, test_loader, num_classes).

    If binary_pixels is True, images are mapped to {0,1} by thresholding at 0.5.
    If classes is not None, only those digit classes are kept.
    """
    tfms = [transforms.ToTensor()]
    if binary_pixels:
        tfms.append(transforms.Lambda(lambda x: (x > 0.5).to(torch.float32)))
    transform = transforms.Compose(tfms)

    trainset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    testset = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    if classes is not None:
        trainset, num_classes = filter_classes(trainset, classes)
        testset, _ = filter_classes(testset, classes)
    else:
        num_classes = 10

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, test_loader, num_classes


# ----------------------------
# Model: Encoder / Decoder / VAE
# ----------------------------

class Encoder(nn.Module):
    """
    Encoder network q(z | x, y).

    Input:
      - y: image tensor of shape (B, 1, 28, 28)
      - x: condition vector of shape (B, num_classes) (one-hot label)

    Output:
      - mu:    (B, z_dim)
      - logvar:(B, z_dim)  where logvar = log(sigma^2)
    """
    def __init__(self, z_dim: int, x_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 28 -> 14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 14 -> 7
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 7 + x_dim, 256),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, z_dim)
        self.fc_logvar = nn.Linear(256, z_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(y).reshape(y.size(0), -1)
        h = torch.cat([h, x], dim=1)
        h = self.fc(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    """
    Decoder network p(y | x, z) for *binary* MNIST.

    This module implements the conditional decoder in a VAE-style model,
    modeling the likelihood p(y | x, z), where:
      - z is a latent vector sampled from the encoder,
      - x is an observed conditioning variable (for example, class label,
        side information, or another input vector),
      - y is a binary MNIST image with shape (1, 28, 28).

    Likelihood model
    ----------------
    Each pixel is modeled as an independent Bernoulli random variable.
    The decoder outputs *logits* (real-valued scores) for each pixel:
        p(y_{i,j} = 1 | x, z) = sigmoid(logit_{i,j})

    Therefore:
      - The decoder MUST NOT apply a sigmoid at the output.
      - The logits are intended to be used with BCEWithLogitsLoss.

    Architecture
    ------------
    The decoder has two stages:

    1) Fully connected network mapping concatenated (z, x) to a feature map:
           input  : (B, z_dim + x_dim)
           output : (B, 32 * 7 * 7)

       Suggested MLP:
           Linear(z_dim + x_dim, 256) -> ReLU -> Linear(256, 32*7*7) -> ReLU

    2) Deconvolutional network upsampling to MNIST resolution:
           (B, 32, 7, 7)
             -> (B, 16, 14, 14) via ConvTranspose2d(32, 16, k=4, s=2, p=1) + ReLU
             -> (B,  1, 28, 28) via ConvTranspose2d(16,  1, k=4, s=2, p=1)

    Output
    ------
    logits : torch.Tensor
        Shape (B, 1, 28, 28), containing Bernoulli logits for each pixel.
    """

    def __init__(self, z_dim: int, x_dim: int):
        """
        Implement according to the class description.

        Parameters
        ----------
        z_dim : int
            Dimension of the latent variable z.
        x_dim : int
            Dimension of the conditioning variable x.
        """
        super().__init__()
        self.z_dim = int(z_dim)
        self.x_dim = int(x_dim)

        self.fc = nn.Sequential(
            nn.Linear(self.z_dim + self.x_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 32 * 7 * 7),
            nn.ReLU(),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 7 -> 14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),   # 14 -> 28
            # no sigmoid here; return logits for BCEWithLogitsLoss
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Implement according to the class description.

        Parameters
        ----------
        x : torch.Tensor
            Conditioning input of shape (B, x_dim).
        z : torch.Tensor
            Latent variable of shape (B, z_dim).

        Returns
        -------
        logits : torch.Tensor
            Bernoulli logits for MNIST pixels, shape (B, 1, 28, 28).
        """
        h = torch.cat([z, x], dim=1)                # (B, z_dim + x_dim)
        h = self.fc(h).view(z.size(0), 32, 7, 7)    # (B, 32, 7, 7)
        logits = self.deconv(h)                     # (B, 1, 28, 28)
        return logits 

class VAE(nn.Module):
    """Conditional VAE wrapper: encoder + decoder + reparameterization."""
    def __init__(self, z_dim: int, x_dim: int):
        super().__init__()
        self.encoder = Encoder(z_dim=z_dim, x_dim=x_dim)
        self.decoder = Decoder(z_dim=z_dim, x_dim=x_dim)
        self.z_dim = z_dim

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = mu + sigma * eps, with eps ~ N(0, I) and sigma = exp(0.5 * logvar)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x, y)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(x, z)
        return logits, mu, logvar

    @torch.no_grad()
    def sample(self, x: torch.Tensor, n: Optional[int] = None) -> torch.Tensor:
        """
        Sample y ~ p(y | x) by sampling z ~ N(0, I), then decoding.

        Returns samples in {0,1} (by thresholding sigmoid probabilities at 0.5 instead of sampling).
        """
        self.eval()
        n = x.size(0)
        z = torch.randn(n, self.z_dim, device=x.device)
        logits = self.decoder(x, z)
        probs = torch.sigmoid(logits)
        return (probs > 0.5).to(torch.float32)


# ----------------------------
# Loss: Bernoulli NLL + KL
# ----------------------------

def kl_diag_gaussian_standard_normal(
    mu: torch.Tensor,
    logvar: torch.Tensor
) -> torch.Tensor:
    """
    Implement according to the following description.

    Compute KL( N(mu, diag(sigma^2)) || N(0, I) ) for each sample, summed over
    latent dimensions.

    Using log-variance parameterization logvar = log(sigma^2), the closed-form
    expression is:
        0.5 * sum_j ( mu_j^2 + exp(logvar_j) - logvar_j - 1 )

    Inputs:
      - mu, logvar: shape (B, d)

    Output:
      - KL divergence per sample: shape (B,)
    """
    return 0.5 * torch.sum(
        mu * mu + torch.exp(logvar) - logvar - 1.0,
        dim=1,
    )


def bernoulli_nll_from_logits(
    logits: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor:
    """
    Implement according to the following description.

    Compute the negative log-likelihood for Bernoulli pixels, summed over all
    pixels per sample.

    If p(y=1 | x, z) = sigmoid(logits), then
        -log p(y | x, z)
        = sum_k [ -y_k log sigma(l_k) - (1 - y_k) log(1 - sigma(l_k)) ]

    This is exactly binary cross-entropy. Use BCEWithLogitsLoss for numerical
    stability.

    Inputs:
      - logits, y: shape (B, 1, 28, 28)

    Output:
      - negative log-likelihood per sample: shape (B,)
    """
    bce = F.binary_cross_entropy_with_logits(
        logits, y, reduction="none"
    )
    return bce.view(bce.size(0), -1).sum(dim=1)

def vae_loss(
    *,
    logits: torch.Tensor,
    y: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute negative ELBO = Bernoulli NLL + KL.
    
    Returns:
      - loss: scalar tensor (mean over batch)
      - stats: dict with mean components (nll, kl)
    """
    nll = bernoulli_nll_from_logits(logits, y)          # (B,)
    kl = kl_diag_gaussian_standard_normal(mu, logvar)   # (B,)
    loss = (nll + kl).mean()
    stats = {
        "nll": float(nll.mean().item()),
        "kl": float(kl.mean().item()),
        "loss": float(loss.item()),
    }
    return loss, stats


# ----------------------------
# Training
# ----------------------------

@dataclass
class RunConfig:
    z_dim: int = 64
    epochs: int = 10
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-3
    seed: int = 42
    save_path: str = "outputs/vae_model.pt"
    load_path: Optional[str] = None
    classes: Optional[str] = "0,1,2"


def one_hot(labels: torch.Tensor, num_classes: int, device: torch.device) -> torch.Tensor:
    return F.one_hot(labels, num_classes=num_classes).to(torch.float32).to(device)


def run_epoch(
    *,
    model: VAE,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    optimizer: Optional[optim.Optimizer],
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_nll = 0.0
    total_kl = 0.0
    total_n = 0

    for imgs, labels in loader:
        y = imgs.to(device)                         # (B, 1, 28, 28), in {0,1}
        x = one_hot(labels, num_classes, device)    # (B, C)

        logits, mu, logvar = model(x, y)
        loss, stats = vae_loss(logits=logits, y=y, mu=mu, logvar=logvar)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        bsz = int(y.size(0))
        total_loss += stats["loss"] * bsz
        total_nll += stats["nll"] * bsz
        total_kl += stats["kl"] * bsz
        total_n += bsz

    if total_n == 0:
        return {"loss": float("nan"), "nll": float("nan"), "kl": float("nan")}

    return {
        "loss": total_loss / total_n,
        "nll": total_nll / total_n,
        "kl": total_kl / total_n,
    }



# ----------------------------
# Generate sample images
# ----------------------------

def myimshow(img: torch.Tensor) -> None:
    """
    img: (1, 28, 28) in [0, 1]
    """
    grid = tv.utils.make_grid(img)
    npimg = grid.detach().cpu().numpy()
    plt.axis("off")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def save_samples_png(
    *,
    model,
    device: torch.device,
    num_classes: int,
    classes: tuple[int, ...] | None,
    save_path: str = "vae_imgs.png",
    max_show: int = 10,
    make_mixture: bool = True,
) -> None:
    model.eval()

    if classes is None:
        label_list = list(range(min(num_classes, max_show)))
    else:
        label_list = [int(c) for c in classes if 0 <= int(c) < num_classes]
        if len(label_list) == 0:
            raise ValueError("No valid class indices to sample. Check `classes`.")

    # Base one-hot conditions
    labels = torch.tensor(label_list, dtype=torch.long, device=device)
    x = F.one_hot(labels, num_classes=num_classes).to(torch.float32)

    # Optional mixture condition
    mix_info = None
    if make_mixture and x.size(0) >= 3:
        mix = (x[0] + x[1] + x[2]) / 2.0
        x = torch.cat([x, mix.unsqueeze(0)], dim=0)
        mix_info = f"mix({label_list[0]},{label_list[1]},{label_list[2]})"

    K = x.size(0)
    num_rows = 4

    # Repeat conditions to get 4 samples per condition
    x_rep = x.repeat(num_rows, 1)

    with torch.no_grad():
        images = model.sample(x_rep)  # (4*K, 1, 28, 28)

    images = images.view(num_rows, K, *images.shape[1:])

    plt.figure(figsize=(3 * K, 3 * num_rows))
    for r in range(num_rows):
        for k in range(K):
            plt.subplot(num_rows, K, r * K + k + 1)

            if mix_info is not None and k == K - 1:
                title = mix_info if r == 0 else None
            else:
                title = f"label={label_list[k]}" if r == 0 else None

            if title is not None:
                head = x[k, :3].detach().cpu().numpy()
                plt.title(f"{title}\nx[:3]={np.array_str(head, precision=2)}")

            myimshow(images[r, k])

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[save] samples -> {save_path}")
    
# ----------------------------
# Checkpoint
# ----------------------------

def save_checkpoint(path: str, model: VAE, config: Dict) -> None:
    """
    Save a checkpoint dict with:
      - model_state_dict
      - config (must include z_dim, x_dim, and the final train/test losses)

    The autograder will reload the checkpoint and recompute losses. Storing the
    final losses here lets us check for consistency and lets students see what
    was saved.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": config}, path)


def load_checkpoint(path: str, map_location: torch.device | str) -> Tuple[VAE, Dict]:
    ckpt = torch.load(path, map_location=map_location)
    cfg = ckpt["config"]
    model = VAE(z_dim=int(cfg["z_dim"]), x_dim=int(cfg["x_dim"]))
    model.load_state_dict(ckpt["model_state_dict"])
    return model, cfg


# ----------------------------
# Main
# ----------------------------

def parse_args() -> RunConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=int, default=1)  # 1=train, 2=load+eval
    args = p.parse_args()

    cfg = RunConfig()
    if args.mode == 2:
        cfg.load_path = cfg.save_path  # reuse default save_path

    return cfg


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"[device] {device}")

    # Parse classes like "0,1,2" or "" (meaning: use all)
    parts = [s.strip() for s in str(cfg.classes).split(",")]
    parts = [int(s) for s in parts if s != ""]
    classes = tuple(parts) if len(parts) > 0 else None

    train_loader, test_loader, num_classes = get_mnist_dataloaders(
        batch_size=cfg.batch_size,
        binary_pixels=True,
        classes=classes,
    )
    x_dim = num_classes

    # ----------------------------
    # Mode 2: autograder (load+eval)
    # ----------------------------
    if cfg.load_path is not None:
        model, saved = load_checkpoint(cfg.load_path, map_location=device)
        model.to(device)
        print(f"[load] {cfg.load_path}  z_dim={saved['z_dim']}  x_dim={saved['x_dim']}")

        # Recompute losses exactly the way the autograder will.
        tr = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            num_classes=num_classes,
            optimizer=None,
        )
        te = run_epoch(
            model=model,
            loader=test_loader,
            device=device,
            num_classes=num_classes,
            optimizer=None,
        )

        print(
            "[eval]\n"
            f"  train loss={tr['loss']:.6f} (nll={tr['nll']:.6f}, kl={tr['kl']:.6f})\n"
            f"  test  loss={te['loss']:.6f} (nll={te['nll']:.6f}, kl={te['kl']:.6f})"
        )

        # Optional consistency print if losses were stored in checkpoint
        if "final_train_loss" in saved and "final_test_loss" in saved:
            print(
                "[ckpt]\n"
                f"  saved train loss={float(saved['final_train_loss']):.6f}\n"
                f"  saved test  loss={float(saved['final_test_loss']):.6f}"
            )

    # ----------------------------
    # Mode 1: train + save ckpt with final losses
    # ----------------------------
    else:
        model = VAE(z_dim=cfg.z_dim, x_dim=x_dim).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        print(
            f"[train] epochs={cfg.epochs}  z_dim={cfg.z_dim}  "
            f"classes={(classes if classes is not None else 'all')}"
        )

        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()
            tr = run_epoch(
                model=model,
                loader=train_loader,
                device=device,
                num_classes=num_classes,
                optimizer=optimizer,
            )
            te = run_epoch(
                model=model,
                loader=test_loader,
                device=device,
                num_classes=num_classes,
                optimizer=None,
            )
            dt = time.time() - t0
            print(
                f"[epoch {epoch:02d}] "
                f"train loss={tr['loss']:.4f} (nll={tr['nll']:.4f}, kl={tr['kl']:.4f})  "
                f"test  loss={te['loss']:.4f} (nll={te['nll']:.4f}, kl={te['kl']:.4f})  "
                f"time={dt:.1f}s"
            )

        # recompute final train/test losses *after* training, with optimizer=None,
        # so these match autograder evaluation exactly (no dropout, no training-mode differences).
        final_tr = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            num_classes=num_classes,
            optimizer=None,
        )
        final_te = run_epoch(
            model=model,
            loader=test_loader,
            device=device,
            num_classes=num_classes,
            optimizer=None,
        )

        print(
            "[final eval]\n"
            f"  train loss={final_tr['loss']:.6f} (nll={final_tr['nll']:.6f}, kl={final_tr['kl']:.6f})\n"
            f"  test  loss={final_te['loss']:.6f} (nll={final_te['nll']:.6f}, kl={final_te['kl']:.6f})"
        )

        save_cfg = {
            "z_dim": int(cfg.z_dim),
            "x_dim": int(x_dim),
            "classes": (None if classes is None else list(classes)),
            "final_train_loss": float(final_tr["loss"]),
            "final_train_nll": float(final_tr["nll"]),
            "final_train_kl": float(final_tr["kl"]),
            "final_test_loss": float(final_te["loss"]),
            "final_test_nll": float(final_te["nll"]),
            "final_test_kl": float(final_te["kl"]),
        }
        save_checkpoint(cfg.save_path, model, config=save_cfg)
        print(f"[save] {cfg.save_path}")

    # ------------------------------------------------------------
    # Sampling demo: generate one image for each requested class
    # ------------------------------------------------------------
    save_samples_png(
        model=model,
        device=device,
        num_classes=num_classes,
        classes=classes,
        save_path="outputs/vae_imgs.png",
        max_show=10,
        make_mixture=True,
    )
    
if __name__ == "__main__":
    main()
