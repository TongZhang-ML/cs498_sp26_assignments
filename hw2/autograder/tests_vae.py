import torch
import torch.nn.functional as F

from vae import (
    Decoder,
    kl_diag_gaussian_standard_normal,
    bernoulli_nll_from_logits,
)


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


# ======================================================
# REQUIRED by autograder (same FCN/CNN style)
# ======================================================

ALL_TESTS = [
    ("Decoder forward shape", 2, test_decoder_forward_shape),
    ("Decoder outputs logits (no sigmoid)", 2, test_decoder_no_sigmoid_behavior),
    ("KL zero case", 2, test_kl_zero_case),
    ("KL closed-form correctness", 2, test_kl_matches_closed_form),
    ("Bernoulli NLL shape", 1, test_nll_shape),
    ("Bernoulli NLL correctness", 1, test_nll_matches_pytorch_bce),
]
