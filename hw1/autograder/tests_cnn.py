import torch
import torch.nn as nn
import numpy as np

import cnn_solution as sol


# =========================================================
# Helpers
# =========================================================

def fake_dataset(n=256):
    """
    Small synthetic MNIST-like dataset.
    Avoids downloading real MNIST for grading speed.
    """
    x = torch.randn(n, 1, 28, 28)
    y = torch.randint(0, 10, (n,))
    return torch.utils.data.TensorDataset(x, y)


# =========================================================
# Tests
# =========================================================

# ---------------------------------------------------------
# 1. ConvNet architecture correctness
# ---------------------------------------------------------
def test_convnet_structure():
    m = sol.MNISTConvNet(use_bn=True)

    assert isinstance(m.conv1, nn.Conv2d)
    assert m.conv1.out_channels == 16
    assert m.conv1.kernel_size == (3, 3)
    assert m.conv1.padding == (1, 1)

    assert isinstance(m.bn1, nn.BatchNorm2d)
    assert isinstance(m.bn2, nn.BatchNorm2d)

    assert isinstance(m.fc1, nn.Linear)
    assert m.fc1.out_features == 128

    assert isinstance(m.fc2, nn.Linear)
    assert m.fc2.out_features == 10


# ---------------------------------------------------------
# 2. BatchNorm toggle behavior
# ---------------------------------------------------------
def test_bn_toggle():
    m = sol.MNISTConvNet(use_bn=False)

    assert isinstance(m.bn1, nn.Identity)
    assert isinstance(m.bn2, nn.Identity)


# ---------------------------------------------------------
# 3. Forward output shape
# ---------------------------------------------------------
def test_convnet_forward_shape():
    m = sol.MNISTConvNet()
    x = torch.randn(5, 1, 28, 28)
    out = m(x)

    assert out.shape == (5, 10)


# ---------------------------------------------------------
# 4. ResBlock preserves shape
# ---------------------------------------------------------
def test_resblock_shape():
    block = sol.ResBlock(16)

    x = torch.randn(4, 16, 14, 14)
    y = block(x)

    assert y.shape == x.shape


# ---------------------------------------------------------
# 5. ResBlock residual math (identity case)
# ---------------------------------------------------------
def test_resblock_residual_identity():
    block = sol.ResBlock(8)

    for p in block.parameters():
        nn.init.zeros_(p)

    x = torch.randn(2, 8, 10, 10)
    y = block(x)

    # since f(x)=0 → output = ReLU(x)
    assert torch.allclose(y, torch.relu(x), atol=1e-6)


# ---------------------------------------------------------
# 6. ResNet forward shape + block count
# ---------------------------------------------------------
def test_resnet_structure():
    m = sol.MNISTResNet(C=16, num_blocks=3)

    assert len(m.blocks) == 3

    x = torch.randn(7, 1, 28, 28)
    out = m(x)

    assert out.shape == (7, 10)


# ---------------------------------------------------------
# 7. fit_and_evaluate training loop correctness
# ---------------------------------------------------------
def test_fit_and_evaluate_runs():
    dataset = fake_dataset(128)

    model = sol.MNISTConvNet()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    call_counter = {"n": 0}

    def eval_fn(m):
        call_counter["n"] += 1
        return 0.0, 0.0

    result = sol.fit_and_evaluate(
        model=model,
        optimizer=opt,
        train_set=dataset,
        batch_size=16,
        epochs=3,
        eval_fn=eval_fn,
    )

    assert len(result.train_losses) == 3
    assert len(result.test_losses) == 3
    assert len(result.test_accs) == 3

    # eval_fn must be called exactly once per epoch
    assert call_counter["n"] == 3


# ---------------------------------------------------------
# 8. DataLoader uses shuffle=True
# ---------------------------------------------------------
def test_shuffle_used():
    dataset = fake_dataset(32)

    model = sol.MNISTConvNet()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    seen_first_batches = []

    def eval_fn(m):
        return 0.0, 0.0

    # capture first element each epoch
    original_loader = sol.DataLoader

    def patched_loader(*args, **kwargs):
        loader = original_loader(*args, **kwargs)
        seen_first_batches.append(loader.shuffle)
        return loader

    sol.DataLoader = patched_loader

    sol.fit_and_evaluate(model, opt, dataset, 8, 2, eval_fn)

    sol.DataLoader = original_loader

    assert all(seen_first_batches), "Training loader must shuffle=True"


# ---------------------------------------------------------
# 9. Loss decreases after training
# ---------------------------------------------------------
def test_training_decreases_loss():
    dataset = fake_dataset(256)

    model = sol.MNISTConvNet()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    losses = []

    def eval_fn(m):
        with torch.no_grad():
            loader = torch.utils.data.DataLoader(dataset, batch_size=64)
            ce = nn.CrossEntropyLoss()
            total = 0
            loss_sum = 0
            for x, y in loader:
                l = ce(m(x), y)
                loss_sum += l.item()
                total += 1
            return loss_sum / total, 0.0

    result = sol.fit_and_evaluate(model, opt, dataset, 32, 3, eval_fn)

    losses = result.train_losses

    assert losses[-1] <= losses[0]


# =========================================================
# Points
# =========================================================

ALL_TESTS = [
    ("ConvNet structure", 50, test_convnet_structure),
    #("BatchNorm toggle", 10, test_bn_toggle),
    #("ConvNet forward shape", 10, test_convnet_forward_shape),
    #("ResBlock shape", 10, test_resblock_shape),
    #("ResBlock residual math", 10, test_resblock_residual_identity),
    #("ResNet structure", 15, test_resnet_structure),
    #("fit_and_evaluate pipeline", 15, test_fit_and_evaluate_runs),
    #("Shuffle enabled", 5, test_shuffle_used),
    #("Training reduces loss", 10, test_training_decreases_loss),
]
