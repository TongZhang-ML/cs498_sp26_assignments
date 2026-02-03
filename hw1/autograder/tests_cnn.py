import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
import cnn as sol
from torchvision import datasets, transforms

# =========================================================
# Helpers
# =========================================================

def get_hidden_test_set() -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Create MNIST train and test datasets with fixed preprocessing.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    return  test_set


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

    # Conv Checks
    assert isinstance(m.conv1, nn.Conv2d)
    assert m.conv1.out_channels == 16
    assert m.conv1.kernel_size == (3, 3)
    assert m.conv1.padding == (1, 1)
    assert isinstance(m.conv2, nn.Conv2d)
    assert m.conv2.out_channels == 32
    assert m.conv2.kernel_size == (3, 3)
    assert m.conv2.padding == (1, 1)

    # MaxPool Checks
    assert isinstance(m.pool1, nn.MaxPool2d)
    assert m.pool1.kernel_size == 2
    assert isinstance(m.pool2, nn.MaxPool2d)
    assert m.pool2.kernel_size == 2

    # BatchNorm Checks
    assert isinstance(m.bn1, nn.BatchNorm2d)
    assert isinstance(m.bn2, nn.BatchNorm2d)

    # FC Checks
    assert isinstance(m.fc1, nn.Linear)
    assert m.fc1.out_features == 128

    assert isinstance(m.fc2, nn.Linear)
    assert m.fc2.out_features == 10

# ---------------------------------------------------------
# 2. Check forward pass output
# ---------------------------------------------------------
def test_forward_pass():
    pass


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
# 5. ResBlock Implementation Checks
# ---------------------------------------------------------
def test_resblock_implementation():
    block = sol.ResBlock(8)

    # Check layers exist
    assert hasattr(block, "conv1"), "ResBlock must have attribute conv1"
    assert hasattr(block, "bn1"), "ResBlock must have attribute bn1"
    assert hasattr(block, "conv2"), "ResBlock must have attribute conv2"
    assert hasattr(block, "bn2"), "ResBlock must have attribute bn2"

    # Check layer types
    assert isinstance(block.conv1, nn.Conv2d), "conv1 must be nn.Conv2d"
    assert isinstance(block.bn1, nn.BatchNorm2d), "bn1 must be nn.BatchNorm2d"
    assert isinstance(block.conv2, nn.Conv2d), "conv2 must be nn.Conv2d"
    assert isinstance(block.bn2, nn.BatchNorm2d), "bn2 must be nn.BatchNorm2d"
    
    # Check conv layer parameters
    assert block.conv1.in_channels == 8, "conv1 in_channels must be C"
    assert block.conv1.out_channels == 8, "conv1 out_channels must be C"
    assert block.conv1.kernel_size == (3, 3), "conv1 kernel_size must be 3"
    assert block.conv1.padding == (1, 1), "conv1 padding must be 1"
    assert block.conv1.bias == None, "conv1 must not have bias"

    assert block.conv2.in_channels == 8, "conv2 in_channels must be C"
    assert block.conv2.out_channels == 8, "conv2 out_channels must be C"
    assert block.conv2.kernel_size == (3, 3), "conv2 kernel_size must be 3"
    assert block.conv2.padding == (1, 1), "conv2 padding must be 1"
    assert block.conv2.bias == None, "conv2 must not have bias"



# ---------------------------------------------------------
# 6. ResBlock residual math (identity case)
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
# 7. ResNet forward shape + block count
# ---------------------------------------------------------
def test_resnet_structure():
    m = sol.MNISTResNet(C=16, num_blocks=3)

    x = torch.randn(7, 1, 28, 28)
    out = m(x)

    assert out.shape == (7, 10)


# ---------------------------------------------------------
# 9. fit_and_evaluate training loop correctness
# ---------------------------------------------------------
def test_fit_and_evaluate_runs():
    train_set, test_set = sol.make_datasets()
    model = sol.MNISTConvNet(use_bn=False)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    call_counter = {"n": 0}
    def eval_fn(model: nn.Module) -> Tuple[float, float]:
        call_counter["n"] += 1
        return sol.eval_on_testset(model=model, test_set=test_set, batch_size=32)


    result = sol.fit_and_evaluate(
        model=model,
        optimizer=opt,
        train_set=train_set,
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
# 10. DataLoader uses shuffle=True
# ---------------------------------------------------------
def test_accuracy_ResNet():
    test_set = get_hidden_test_set()

    model = sol.MNISTResNet(C=16, num_blocks=3)
    model.load_state_dict(torch.load('resnet3x16_sgd.pth'))  # reset weights
    
    def eval_fn(model: nn.Module) -> Tuple[float, float]:
        return sol.eval_on_testset(model=model, test_set=test_set, batch_size=32)

    test_loss, test_acc = eval_fn(model)

    assert test_acc >= 0.90, f"Test accuracy must be at least 90%, got {test_acc*100:.2f}%"
    assert test_loss <= 0.35, f"Test loss must be at most 0.35, got {test_loss:.4f}"

        


# # ---------------------------------------------------------
# # 10. DataLoader uses shuffle=True
# # ---------------------------------------------------------
# def test_shuffle_used():
#     dataset = fake_dataset(32)

#     model = sol.MNISTConvNet()
#     opt = torch.optim.SGD(model.parameters(), lr=0.01)

#     seen_first_batches = []

#     def eval_fn(m):
#         return 0.0, 0.0

#     # capture first element each epoch
#     original_loader = sol.DataLoader

#     def patched_loader(*args, **kwargs):
#         loader = original_loader(*args, **kwargs)
#         seen_first_batches.append(loader.shuffle)
#         return loader

#     sol.DataLoader = patched_loader

#     sol.fit_and_evaluate(model, opt, dataset, 8, 2, eval_fn)

#     sol.DataLoader = original_loader

#     assert all(seen_first_batches), "Training loader must shuffle=True"


# # ---------------------------------------------------------
# # 11. Loss decreases after training
# # ---------------------------------------------------------
# def test_training_decreases_loss():
#     dataset = fake_dataset(256)

#     model = sol.MNISTConvNet()
#     opt = torch.optim.SGD(model.parameters(), lr=0.1)

#     losses = []

#     def eval_fn(m):
#         with torch.no_grad():
#             loader = torch.utils.data.DataLoader(dataset, batch_size=64)
#             ce = nn.CrossEntropyLoss()
#             total = 0
#             loss_sum = 0
#             for x, y in loader:
#                 l = ce(m(x), y)
#                 loss_sum += l.item()
#                 total += 1
#             return loss_sum / total, 0.0

#     result = sol.fit_and_evaluate(model, opt, dataset, 32, 3, eval_fn)

#     losses = result.train_losses

#     assert losses[-1] <= losses[0]


# =========================================================
# Points
# =========================================================

ALL_TESTS = [
    ("ConvNet structure", 5, test_convnet_structure),
    ("ConvNet forward shape", 5, test_convnet_forward_shape),
    ("ResBlock Structure", 5, test_resblock_implementation),
    ("ResBlock shape", 5, test_resblock_shape),
    ("ResBlock residual math", 5, test_resblock_residual_identity),
    ("ResNet structure", 5, test_resnet_structure),
    ("fit_and_evaluate pipeline", 5, test_fit_and_evaluate_runs),
    ("Accuracy ResNet", 20, test_accuracy_ResNet),
    #("Shuffle enabled", 5, test_shuffle_used),
    #("Training reduces loss", 10, test_training_decreases_loss),
]
