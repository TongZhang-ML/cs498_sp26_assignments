import torch
import torch.nn.functional as F

from vae import (
    Decoder,
    kl_diag_gaussian_standard_normal,
    bernoulli_nll_from_logits,
    VAE,
    one_hot,
    vae_loss,
    get_mnist_dataloaders
)

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader



# ======================================================
# Tests ONLY for student-implemented parts
# ======================================================

# ------------------------------------------------------
# Decoder
# ------------------------------------------------------

def test_decoder_forward_shape():
    model = Decoder(z_dim=8, x_dim=3)

    x = torch.randn(5, 3)
    z = torch.randn(5, 8)

    logits = model(x, z)

    assert logits.shape == (5, 1, 28, 28)


def test_decoder_no_sigmoid_behavior():
    """
    Ensure decoder outputs logits (not probabilities).
    Values should NOT be restricted to [0, 1].
    """
    model = Decoder(z_dim=4, x_dim=2)

    x = torch.randn(4, 2)
    z = torch.randn(4, 4)

    logits = model(x, z)

    # if sigmoid was applied, everything would be in [0,1]
    assert logits.min() < 0 or logits.max() > 1


# ------------------------------------------------------
# KL divergence
# ------------------------------------------------------

def test_kl_zero_case():
    """
    KL(N(0,I) || N(0,I)) == 0
    """
    mu = torch.zeros(6, 10)
    logvar = torch.zeros(6, 10)

    kl = kl_diag_gaussian_standard_normal(mu, logvar)

    assert torch.allclose(kl, torch.zeros_like(kl))


def test_kl_matches_closed_form():
    """
    Compare against manual formula.
    """
    mu = torch.randn(4, 7)
    logvar = torch.randn(4, 7)

    kl = kl_diag_gaussian_standard_normal(mu, logvar)

    manual = 0.5 * torch.sum(mu**2 + torch.exp(logvar) - logvar - 1, dim=1)

    assert torch.allclose(kl, manual)


# ------------------------------------------------------
# Bernoulli NLL
# ------------------------------------------------------

def test_nll_shape():
    logits = torch.randn(3, 1, 28, 28)
    y = torch.randint(0, 2, logits.shape).float()

    nll = bernoulli_nll_from_logits(logits, y)

    assert nll.shape == (3,)


def test_nll_matches_pytorch_bce():
    """
    Must exactly equal BCEWithLogitsLoss summed per sample.
    """
    logits = torch.randn(2, 1, 28, 28)
    y = torch.randint(0, 2, logits.shape).float()

    ours = bernoulli_nll_from_logits(logits, y)

    ref = F.binary_cross_entropy_with_logits(
        logits, y, reduction="none"
    ).view(2, -1).sum(dim=1)

    assert torch.allclose(ours, ref)


# ------------------------------------------------------
# Checkpoint evaluation (vae.pth on MNIST)
# ------------------------------------------------------

def helper(model, config, loader, device):
    total_loss = 0.0
    total_nll = 0.0
    total_kl = 0.0
    total_n = 0

    with torch.no_grad():
        for imgs, labels in loader:
            y = imgs.to(device)                         # (B, 1, 28, 28), in {0,1}
            x = one_hot(labels, config["x_dim"], device)    # (B, C)

            logits, mu, logvar = model(x, y)
            loss, stats = vae_loss(logits=logits, y=y, mu=mu, logvar=logvar)

            bsz = int(y.size(0))
            total_loss += stats["loss"] * bsz
            total_nll += stats["nll"] * bsz
            total_kl += stats["kl"] * bsz
            total_n += bsz


    loss =  total_loss / total_n
    nll = total_nll / total_n
    kl =  total_kl / total_n

    return loss, nll, kl


def test_checkpoint_eval():
    device = torch.device("cpu")

    # -------------------------
    # file exists
    # -------------------------
    if not os.path.isfile("vae_model.pt"):
        raise AssertionError("vae_model.pt not found")
    saved_dict = torch.load("vae_model.pt", map_location=device)
    config = saved_dict['config']
    state_dict = saved_dict['model_state_dict']


    # -------------------------
    # load model
    # assumes full model was saved with torch.save(model)
    # -------------------------
    model = VAE(
        z_dim=int(config['z_dim']),
        x_dim=int(config['x_dim'])
    )

    model.load_state_dict(state_dict)
    model.eval()

    # -------------------------
    # MNIST loader
    # -------------------------
    transform = [transforms.ToTensor()]
    transform.append(transforms.Lambda(lambda x: (x > 0.5).to(torch.float32)))
    transform = transforms.Compose(transform)

    train_loader, test_loader, num_classes = get_mnist_dataloaders(
        batch_size=128, binary_pixels=True, classes=config['classes']
    )

    train_loss, train_nll, train_kl = helper(model, config, train_loader, device)
    test_loss, test_nll, test_kl = helper(model, config, test_loader, device)

    assert train_loss < 75 and train_kl < 19 and train_nll < 56
    assert test_loss < 75 and test_kl < 19 and test_nll < 56





# ======================================================
# REQUIRED by autograder (same FCN/CNN style)
# ======================================================


ALL_TESTS = [
    ("Decoder Implementation A", 5, test_decoder_forward_shape),
    ("Decoder Implementation B", 5, test_decoder_no_sigmoid_behavior),
    ("KL Implementation A", 2, test_kl_zero_case),
    ("KL Implementation B", 2, test_kl_matches_closed_form),
    ("Bernoulli NLL Implementation A", 2, test_nll_shape),
    ("Bernoulli NLL Implementation B", 2, test_nll_matches_pytorch_bce),
    ("Hidden Test", 4, test_checkpoint_eval),
]
