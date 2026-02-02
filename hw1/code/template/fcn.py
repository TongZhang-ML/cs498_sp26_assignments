# fcn_solution.py
#

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import os
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler


# ----------------------------
# Reproducibility
# ----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # ok if no cuda
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------
# Model
# ----------------------------

class MLP(nn.Module):
    """
    Fully-connected neural network mapping R^d -> R (one scalar logit).

    Architecture (as you specified):

      Input: x shape (batch, input_dim)

      Layer 1: Linear(input_dim -> hidden_dim) with bias
               ReLU

      Layer 2: Linear(hidden_dim -> hidden_dim) without bias
               ReLU

      Layer 3: Linear(hidden_dim -> hidden_dim) without bias
               ReLU

      Output:  Linear(hidden_dim -> 1) with bias

    Notes:
      - Output is a real number (logit for logistic regression, and regresion value for least squares)
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        # Implement based on the class description
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Implement the function based on the description of the model
        pass

# ----------------------------
# Loss + metrics
# ----------------------------
def compute_loss(logits: torch.Tensor, y: torch.Tensor, loss_name: str) -> torch.Tensor:
    """
    Compute mean loss over a batch.

    logits: model outputs, shape (batch,)
    y: labels in {0,1}, shape (batch,)

    loss_name:
      - "least_squares": 0.5 * (logits - y)^2
      - "logistic_regression": binary cross entropy with logits
    """
    if loss_name == "least_squares":
        return 0.5 * torch.mean((logits - y) ** 2)
    if loss_name == "logistic_regression":
        return F.binary_cross_entropy_with_logits(logits, y)
    raise ValueError(f"Unknown loss_name={loss_name}")

@torch.no_grad()
def compute_accuracy(output: torch.Tensor, y: torch.Tensor, loss_name: str)  -> float:
    """
    Compute classification accuracy for y in {0,1}.

    - logistic_regression:
        predict y=1 if output >= 0
    - least_squares:
        predict y=1 if output >= 0.5
    """
    if loss_name == "logistic_regression":
        pred = (output >= 0).float()
    elif loss_name == "least_squares":
        pred = (output >= 0.5).float()
    else:
        raise ValueError(f"Unknown loss_name={loss_name}")

    return float((pred == y).float().mean().item())


# ----------------------------
# Optimizer
# ----------------------------

def make_optimizer(params,  lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """
    Implement based on the following description.

    Create and return an SGD optimizer with momentum 0.9.

    Parameters
    ----------
    params :
        Iterable of model parameters, typically obtained from
        model.parameters().

    lr : float
        Learning rate (step size) for stochastic gradient descent.

    weight_decay : float
        Weight decay coefficient (L2 regularization strength).

    Returns
    -------
    torch.optim.Optimizer
        A torch.optim.SGD optimizer initialized with:
          - learning rate = lr
          - momentum = 0.9
          - weight_decay = weight_decay

    Required Implementation
    -----------------------
    The function must return an instance of torch.optim.SGD()
    No other optimizer should be used.
    """
    pass


# ----------------------------
# Transform: fit scaler on train, reuse on val/test
# ----------------------------

def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """
    Implement based on the following description.

    Fit a feature-wise StandardScaler using the training data.

    This function computes the mean and standard deviation of each feature
    in the training set and stores them in a StandardScaler object.

    For each feature j, the scaler learns:
      - mean:  μ_j
      - standard deviation: σ_j

    The fitted scaler can later be used to normalize data using:
        x_j -> (x_j - μ_j) / σ_j

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix of shape (n_samples, n_features).

    Returns
    -------
    StandardScaler
        A fitted sklearn.preprocessing.StandardScaler instance.

    Notes
    -----
    - The scaler must be fit using ONLY the training data.
    - The returned scaler should be reused to transform training,
      validation, and test data to ensure consistent feature scaling.
    """
    pass


def apply_scaler(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    return scaler.transform(X)


def make_train_loader(X: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
    """
    Implement based on the following description

    Create a DataLoader for the training set.

    Parameters
    ----------
    X : np.ndarray
        Training features of shape (n_samples, n_features).

    y : np.ndarray
        Training labels of shape (n_samples,), with values in {0,1}.

    batch_size : int
        Number of training examples per mini-batch.

    Returns
    -------
    DataLoader
        A PyTorch DataLoader that iterates over the training data.

    Required Implementation
    -----------------------
    The function must return a torch.utils.data.DataLoader with
     batch_size, drop_last = False, and decide whether to shuffle (choose the better one)

    Important
    ---------
    You must decide whether `shuffle` should be set to True or False
    for the training DataLoader.

    In your solution, briefly explain (in comments or in your write-up)
    why shuffling is or is not appropriate for training in this setting.
    """
    pass


def make_test_loader(X: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

# ----------------------------
# Training + eval
# ----------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_name: str,
    device: torch.device,
) -> float:
    """
    Implement the function based on the following description.

    Train the neural network for one epoch (one full pass over the training data).

    Parameters
    ----------
    model : nn.Module
        The neural network model to be trained.

    loader : DataLoader
        A PyTorch DataLoader that provides mini-batches of
        (features, labels) from the training set.

    optimizer : torch.optim.Optimizer
        Optimizer used to update model parameters (e.g., SGD or AdamW).

    loss_name : str
        Name of the loss function to use. Supported values are:
          - "least_squares"
          - "logistic_regression"

    device : torch.device
        Device on which to perform computation (CPU or GPU).

    Returns
    -------
    float
        The average training loss over all training examples in this epoch.

    Required Implementation
    -----------------------
    The function must perform the following steps:

      1. Set the model to training mode.

      2. For each mini-batch (xb, yb) provided by the DataLoader:
           a. Move xb and yb to the specified device.
           b. Compute model outputs
           c. Compute the loss using compute_loss()
           d. Clear previous gradients 
           e. Backpropagate the loss
           f. Update model parameters

      3. Accumulate and return the average loss over all training examples.
    """
    train_loss = 0.0
    total_n = 0

    pass


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_name: str, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_n = 0
    logits_list = []
    y_list = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = compute_loss(logits, yb, loss_name)

        bs = xb.shape[0]
        total_loss += float(loss.item()) * bs
        total_n += bs

        logits_list.append(logits.detach().cpu())
        y_list.append(yb.detach().cpu())

    logits_full = torch.cat(logits_list, dim=0)
    y_full = torch.cat(y_list, dim=0)
    acc = compute_accuracy(logits_full, y_full,loss_name)
    return total_loss / max(total_n, 1), acc


# ----------------------------
# CV + LaTeX output
# ----------------------------

@dataclass
class TrainConfig:
    loss_name: str
    weight_decay: float
    hidden_dim: int
    batch_size: int
    epochs: int
    seed: int 


def latex_cv_table(lr_to_acc: Dict[float, float], best_lr: float) -> str:
    lines = []
    lines.append(r"\begin{tabular}{cc}")
    lines.append(r"\hline")
    lines.append(r"Step Size & CV Accuracy \\")
    lines.append(r"\hline")
    for lr in sorted(lr_to_acc.keys()):
        lines.append(f" {lr:0.3f} & {lr_to_acc[lr]:0.2f} \\\\")
        lines.append(r"\hline")
    lines.append(rf"\multicolumn{{2}}{{c}}{{Best step size:  {best_lr:0.3f}}} \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    learning_rates: List[float],
    cfg: TrainConfig,
    k_folds: int,
    device: torch.device,
) -> Tuple[float, Dict[float, float], str]:
    set_seed(cfg.seed)

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=cfg.seed)
    lr_to_accs: Dict[float, List[float]] = {lr: [] for lr in learning_rates}

    for lr in learning_rates:
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            scaler = fit_scaler(X_train)
            X_train_s = apply_scaler(scaler, X_train)
            X_val_s = apply_scaler(scaler, X_val)

            train_loader = make_train_loader(X_train_s, y_train, batch_size=cfg.batch_size)
            val_loader = make_test_loader(X_val_s, y_val, batch_size=cfg.batch_size)

            model = MLP(input_dim=X.shape[1], hidden_dim=cfg.hidden_dim).to(device)
            optimizer = make_optimizer(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)

            for _ in range(cfg.epochs):
                train_one_epoch(model, train_loader, optimizer, cfg.loss_name, device)

            _, val_acc = evaluate(model, val_loader, cfg.loss_name, device)
            lr_to_accs[lr].append(val_acc)

    lr_to_mean_acc = {lr: float(np.mean(accs)) for lr, accs in lr_to_accs.items()}
    best_lr = sorted(lr_to_mean_acc.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    latex = latex_cv_table(lr_to_mean_acc, best_lr)
    return best_lr, lr_to_mean_acc, latex


# ----------------------------
# CSV loading
# ----------------------------

def load_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    feature_cols = [f"feature{i}" for i in range(1, 7)]
    if any(c not in df.columns for c in feature_cols + ["target"]):
        raise ValueError(f"CSV must contain columns {feature_cols + ['target']}")
    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df["target"].to_numpy(dtype=np.float64)
    return X, y


# ----------------------------
# Final train on all train.csv, eval on test.csv
# ----------------------------

def train_final_and_test(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: TrainConfig,
    best_lr: float,
    device: torch.device,
    save_path: str,
) -> None:
    scaler = fit_scaler(X_train)
    X_train_s = apply_scaler(scaler, X_train)
    X_test_s = apply_scaler(scaler, X_test)

    train_loader = make_train_loader(X_train_s, y_train, batch_size=cfg.batch_size)
    test_loader = make_test_loader(X_test_s, y_test, batch_size=cfg.batch_size)

    model = MLP(input_dim=X_train.shape[1], hidden_dim=cfg.hidden_dim).to(device)
    optimizer = make_optimizer(model.parameters(), lr=best_lr, weight_decay=cfg.weight_decay)

    for _ in range(cfg.epochs):
        train_one_epoch(model, train_loader, optimizer, cfg.loss_name, device)

    test_loss, test_acc = evaluate(model, test_loader, cfg.loss_name, device)
    print(f"\nTest results ({cfg.loss_name}): loss={test_loss:.4f}, accuracy={test_acc:.4f}\n")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(
       {
           "model_state": model.state_dict(),
           "scaler_mean": scaler.mean_,
           "scaler_scale": scaler.scale_,
           "config": cfg.__dict__,
           "best_lr": best_lr,
       },
       save_path,
    )
    print(f"Saved model+scaler to {save_path}\n")


# ----------------------------
# Main 
# ----------------------------

def main() -> None:
    train_path = "data/train.csv"
    test_path = "data/test.csv"
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Could not find {train_path}.")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Could not find {test_path}.")

    X, y = load_csv(train_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    learning_rates = [0.001, 0.01, 0.10, 1.00, 10.0]
    k_folds = 5

    base_cfg = TrainConfig(
        loss_name="logistic_regression",  
        weight_decay=0.001,               
        hidden_dim=64,
        batch_size=32,
        epochs=10,
        seed=42,
        )

    loss_names = ["logistic_regression", "least_squares"]

    best_lr_by_loss = {}

    # 1) CV loop
    for loss_name in loss_names:
        cfg = TrainConfig(**{**base_cfg.__dict__, "loss_name": loss_name})
        best_lr, _, latex = cross_validation(X_train, y_train, learning_rates, cfg, k_folds, device)
        best_lr_by_loss[loss_name] = best_lr
        print(f"\n{{Results for loss function: {loss_name}}}\n")
        print(latex)

    # 2) Final train+test loop
    for loss_name in loss_names:
        cfg = TrainConfig(**{**base_cfg.__dict__, "loss_name": loss_name})
        train_final_and_test(
            X_train, y_train, X_test, y_test,
            cfg=cfg,
            best_lr=best_lr_by_loss[loss_name],
            device=device,
            save_path=f"saved_models/mlp_{loss_name}.pt",
          )    

if __name__ == "__main__":
    main()
    
