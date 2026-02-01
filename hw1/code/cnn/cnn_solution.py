# cnn_solution.py
# Reference implementation (can be used as solution or as a starting point).

from __future__ import annotations

import os
import random
import numpy as np

from dataclasses import dataclass
from typing import Callable, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import matplotlib.pyplot as plt


# ----------------------------
# Reproducibility
# ----------------------------

def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    This function seeds Python, NumPy, and PyTorch random number generators.
    It should be called once at the beginning of each independent run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # safe even if no CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------
# Models
# ----------------------------

class MNISTConvNet(nn.Module):
    """
    Convolutional neural network for MNIST digit classification.

    This model takes grayscale MNIST images of shape (N, 1, 28, 28)
    and outputs class scores of shape (N, 10).

    Architecture (applied in the given order):

    1. Conv2d:
       - out_channels = 16
       - kernel_size = 3
       - padding = 1
       - bias = True

    2. BatchNorm2d, optional:
       - Used only if use_bn = True
       - Otherwise replaced by identity

    3. ReLU 

    4. MaxPool2d(kernel_size = 2):
       - Downsamples spatial dimensions by a factor of 2

    5. Conv2d:
       - out_channels = 32
       - kernel_size = 3
       - padding = 1
       - bias = True

    6. BatchNorm2d(32), optional:
       - Used only if use_bn = True
       - Otherwise replaced by identity

    7. ReLU (elementwise)

    8. MaxPool2d(kernel_size = 2):

    9. Flatten:
       - Reshapes each example into a vector

    10. Linear layer:
        - Output dimension: 128
        Followed by ReLU

    11. Linear layer:
        - Output dimension: 10

    Parameters
    ----------
    use_bn : bool, default=True
        If True, batch normalization layers are used after each
        convolution. If False, batch normalization is skipped.

    Notes
    -----
    - All convolutions preserve spatial size due to padding = 1.
    - Max pooling reduces height and width by a factor of 2.
    - Students should verify all intermediate tensor shapes.
    """
    def __init__(self, use_bn: bool = True):
        # Implement based on the class description
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(16) if use_bn else nn.Identity()
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(32) if use_bn else nn.Identity()
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Implement based on the class description
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


class ResBlock(nn.Module):
    """
    Residual block used in a simplified ResNet.

    For an input tensor x of shape (N, C, H, W), the block computes

        ResBlock(x) = ReLU(x + f(x)),

    where f(x) is a residual mapping defined by the following sequence
    of layers (applied in order):

    1. Conv2d:
       - in_channels = C
       - out_channels = C
       - kernel_size = 3
       - stride = 1
       - padding = 1
       - bias = False

    2. BatchNorm2d

    3. ReLU 

    4. Conv2d:
       - in_channels = C
       - out_channels = C
       - kernel_size = 3
       - stride = 1
       - padding = 1
       - bias = False

    5. BatchNorm2d

    The input x is added elementwise to the output of f(x), and a final
    ReLU is applied.

    Parameters
    ----------
    C : int
        Number of input and output channels. The spatial dimensions
        (H, W) are preserved throughout the block.

    Notes
    -----
    - Padding = 1 ensures spatial size is unchanged.
    - No projection is used on the skip connection; therefore the input
      and output must have the same shape.
    """
    def __init__(self, C: int):
        # Implement based on the class description
        super().__init__()
        self.conv1 = nn.Conv2d(C, C, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(C)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(C, C, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Implement based on the class description
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.relu(x + y)


class MNISTResNet(nn.Module):
    """
    Simplified ResNet-style convolutional neural network for MNIST.

    The network takes grayscale MNIST images of shape (N, 1, 28, 28)
    and outputs class scores of shape (N, 10).

    Architecture overview:

    1. Initial convolution:
       - Conv2d(1, C, kernel_size=3, stride=2, padding=1, bias=False)

    2. BatchNorm2d(C)

    3. ReLU

    4. MaxPool2d(kernel_size=2)

    5. Residual blocks:
       - Apply separate ResBlock(C) repeatedly, for num_blocks times

    6. Adaptive average pooling:
       - AdaptiveAvgPool2d((1, 1))

    7. Flatten:
       - Reshapes to (N, C)

    8. Fully connected layer:
       - Linear(C, 10)

    Parameters
    ----------
    C : int, default=16
        Number of channels used throughout the network.

    num_blocks : int, default=1
        Number of times the residual block is applied in sequence.
        Different ResBlock instances (no parameter share) should be used.

    Notes
    -----
    - All residual blocks operate at the same spatial resolution.
    - No downsampling occurs inside the residual blocks.
    - Adaptive average pooling removes dependence on spatial size
      before the final classifier.
    """
    def __init__(self, C: int = 16, num_blocks: int = 1):
        # Implement based on the class description
        super().__init__()

        self.conv = nn.Conv2d(1, C, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(C)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.blocks = nn.ModuleList([ResBlock(C) for _ in range(num_blocks)])

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(C, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Implement based on the class description
        x = self.pool(self.relu(self.bn(self.conv(x))))
        for block in self.blocks:
            x = block(x)
        x = self.avg(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


# ----------------------------
# Dataset
# ----------------------------

def make_datasets() -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Create MNIST train and test datasets with fixed preprocessing.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    total_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_set, test_set = random_split(total_data, [0.8, 0.2])
    return train_set, test_set


# ----------------------------
# Training and evaluation
# ----------------------------

@torch.no_grad()
def eval_on_testset(
    model: nn.Module,
    test_set: torch.utils.data.Dataset,
    batch_size: int ) -> Tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    total_loss = 0.0
    total = 0
    correct = 0

    for x, y in loader:
        logits = model(x)
        loss = loss_fn(logits, y)

        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        total += bs

        pred = torch.argmax(logits, dim=1)
        correct += (pred == y).sum().item()

    return total_loss / max(total, 1), correct / max(total, 1)

@dataclass
class RunResult:
    name: str 
    train_losses: List[float]
    test_losses: List[float]
    test_accs: List[float]

def fit_and_evaluate(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_set: torch.utils.data.Dataset,
    batch_size: int,
    epochs: int,
    eval_fn: Callable[[nn.Module], Tuple[float, float]],
) -> RunResult:
    """
    Implement the function based on the following description.

    Train a model for multiple epochs and evaluate it after each epoch
    using a provided evaluation function.

    IMPORTANT:
    ----------
    This function receives ONLY the training dataset.
    It must NOT access the test dataset directly.

    Evaluation must be performed exclusively by calling `eval_fn(model)`,
    which internally evaluates the model on the test data and returns
    (test_loss, test_accuracy).

    Required behavior
    -----------------
    The function must perform the following steps:

    1. Create a training DataLoader from `train_set` with:
         - batch_size = batch_size
         - shuffle = True

    2. Define the loss function as:
         nn.CrossEntropyLoss()

    3. For each epoch (repeat `epochs` times):

       a. TRAINING PHASE
          - Set the model to training mode.
          - Iterate over the training DataLoader.
          - For each mini-batch (x, y):
              * clear gradients
              * forward pass
              * compute loss
              * backward pass
              * optimizer step
          - Compute the average training loss over all training samples.

       b. EVALUATION PHASE
          - Call `eval_fn(model)` exactly once.
          - `eval_fn` returns:
              * test_loss : float
              * test_accuracy : float

       c. Record:
          - training loss for the epoch
          - test loss for the epoch
          - test accuracy for the epoch

    Parameters
    ----------
    model : nn.Module
        Neural network model to be trained. The model is updated
        in place during training.

    optimizer : torch.optim.Optimizer
        Optimizer used to update the parameters of `model`.

    train_set : torch.utils.data.Dataset
        Training dataset only.

    batch_size : int
        Mini-batch size for training.

    epochs : int
        Number of training epochs.

    eval_fn : Callable[[nn.Module], Tuple[float, float]]
        Evaluation function that takes the trained model and returns
        (test_loss, test_accuracy). The test dataset is hidden
        inside this function.

    Returns
    -------
    RunResult Class
        An object containing results recorded after each epoch:
          - name : str  (set to "")
          - train_losses : List[float]
          - test_losses : List[float]
          - test_accs : List[float]
    """
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    loss_fn = nn.CrossEntropyLoss()

    train_losses: List[float] = []
    test_losses: List[float] = []
    test_accs: List[float] = []

    for _ in range(epochs):
        # ---- training phase ----
        model.train()
        total_loss = 0.0
        total = 0

        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            total_loss += float(loss.item()) * bs
            total += bs

        train_loss = total_loss / max(total, 1)

        # ---- evaluation phase (black box) ----
        test_loss, test_acc = eval_fn(model)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

    return RunResult(
        name="",
        train_losses=train_losses,
        test_losses=test_losses,
        test_accs=test_accs,
    )

# ----------------------------
# Plotting and Reporting
# ----------------------------

def report_result(result: RunResult, run_name: str) -> str:
    result.name=run_name
    epochs = list(range(1, len(result.train_losses) + 1))

    plt.figure()
    plt.plot(epochs, result.train_losses, label="train loss")
    plt.plot(epochs, result.test_losses, label="test loss")
    plt.xlabel("epoch")
    plt.ylabel("cross entropy loss")
    plt.legend()

    path = f"outputs/{result.name}.png"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()

    final_acc = result.test_accs[-1] * 100.0
    final_loss = result.test_losses[-1]

    print(
        f"{result.name}: "
        f"reported test acc = {final_acc:.2f}%  "
        f"test loss = {final_loss:.4f}  "
        f"(plot: {path})"
    )
    return path

def check_result(result, model, eval_fn, atol=1e-6):
    """
    Verify that the stored final test metrics match a fresh evaluation.
    """
    final_acc = result.test_accs[-1] * 100.0
    final_loss = result.test_losses[-1]

    test_loss, test_acc = eval_fn(model)
    test_acc = test_acc * 100.0

    acc_ok = abs(final_acc - test_acc) < atol
    loss_ok = abs(final_loss - test_loss) < atol

    is_passed = acc_ok and loss_ok

    print(f"{result.name}: result check passed = {is_passed}")

    return is_passed

# ----------------------------
# Main
# ----------------------------

def main() :

    # Datasets (train_set is passed into fit_and_evaluate; test_set is hidden inside eval_fn)
    train_set, test_set = make_datasets()

    # Evaluation function that closes over test_set (students do not see / do not receive test_set)
    def eval_fn(model: nn.Module) -> Tuple[float, float]:
        return eval_on_testset(model=model, test_set=test_set, batch_size=32)

    # (1) CNN + BN + SGD
    set_seed(42)
    m1 = MNISTConvNet(use_bn=True)
    opt1 = torch.optim.SGD(m1.parameters(), lr=5e-2, momentum=0.9)
    r1 = fit_and_evaluate(
        model=m1,
        optimizer=opt1,
        train_set=train_set,
        batch_size=32,
        epochs=5,
        eval_fn=eval_fn,
    )
    report_result(r1,"cnn_bn_sgd")
    check_result(r1,m1,eval_fn)

    # (2) CNN without BN + SGD
    set_seed(42)
    m2 = MNISTConvNet(use_bn=False)
    opt2 = torch.optim.SGD(m2.parameters(), lr=5e-2, momentum=0.9)
    r2 = fit_and_evaluate(
        model=m2,
        optimizer=opt2,
        train_set=train_set,
        batch_size=32,
        epochs=5,
        eval_fn=eval_fn,
    )
    report_result(r2,"cnn_nobn_sgd")
    check_result(r2,m2,eval_fn)


    # (3) CNN + BN + Adam
    set_seed(42)
    m3 = MNISTConvNet(use_bn=True)
    opt3 = torch.optim.Adam(m3.parameters(), lr=1e-2)
    r3 = fit_and_evaluate(
        model=m3,
        optimizer=opt3,
        train_set=train_set,
        batch_size=32,
        epochs=5,
        eval_fn=eval_fn,
    )
    report_result(r3,"cnn_bn_adam")
    check_result(r3,m3,eval_fn)

    # (4) ResNet variants
    for num_blocks in [1, 3]:
        set_seed(42)
        m4 = MNISTResNet(C=16, num_blocks=num_blocks)
        opt4 = torch.optim.SGD(m4.parameters(), lr=5e-2, momentum=0.9)
        r4 = fit_and_evaluate(
            model=m4,
            optimizer=opt4,
            train_set=train_set,
            batch_size=32,
            epochs=5,
            eval_fn=eval_fn,
        )
        report_result(r4,f"resnet{num_blocks}x16_sgd")
        check_result(r4,m4,eval_fn)



if __name__ == "__main__":
    main()
 
