import torch
import numpy as np

# This imports the student's submitted file
import fcn_solution as sol
from torch.utils.data import DataLoader, TensorDataset
from fcn_solution import MLP, evaluate

# =========================================================
# Helpers
# =========================================================

def make_toy_data(n=128, d=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = rng.normal(size=d)
    y = (X @ w > 0).astype(np.float32)
    return X, y


def assert_close(a, b, tol=1e-5):
    assert np.allclose(a, b, atol=tol), f"{a} != {b}"


# =========================================================
# Tests
# =========================================================

# ---------------------------------------------------------
# 1. Model architecture
# ---------------------------------------------------------
def test_model_architecture():
    model = sol.MLP(input_dim=6, hidden_dim=32)

    layers = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]

    assert len(layers) == 4, "Must have exactly 4 Linear layers"

    assert layers[0].bias is not None, "Layer 1 must have bias"
    assert layers[1].bias is None, "Layer 2 must not have bias"
    assert layers[2].bias is None, "Layer 3 must not have bias"
    assert layers[3].bias is not None, "Output layer must have bias"

    x = torch.randn(5, 6)
    out = model(x)

    assert out.shape == (5,), "Output must be shape (batch,)"


# ---------------------------------------------------------
# 2. Optimizer correctness
# ---------------------------------------------------------
def test_optimizer():
    m = sol.MLP(6, 8)

    opt = sol.make_optimizer(m.parameters(), lr=0.01, weight_decay=0.1)

    assert isinstance(opt, torch.optim.SGD), "Must use torch.optim.SGD"

    g = opt.param_groups[0]

    assert_close(g["lr"], 0.01)
    assert_close(g["momentum"], 0.9)
    assert_close(g["weight_decay"], 0.1)


# ---------------------------------------------------------
# 3. Scaler correctness
# ---------------------------------------------------------
def test_scaler():
    X = np.random.randn(200, 6) * 5 + 10

    scaler = sol.fit_scaler(X)
    Xs = scaler.transform(X)

    assert_close(Xs.mean(0), np.zeros(6), 1e-6)
    assert_close(Xs.std(0), np.ones(6), 1e-6)


# ---------------------------------------------------------
# 4. Train loader shuffle
# ---------------------------------------------------------
def test_train_loader_batch_and_drop_last():
    n = 65
    batch_size = 16
    X, y = make_toy_data(n=n)

    loader = sol.make_train_loader(X, y, batch_size=batch_size)

    sizes = [xb.size(0) for xb, _ in loader]

    # full batches must equal batch_size
    for s in sizes[:-1]:
        assert s == batch_size

    # last batch must exist and be partial
    assert sizes[-1] == n % batch_size

    # total samples must match exactly
    assert sum(sizes) == n



# ---------------------------------------------------------
# 5. Training reduces loss
# ---------------------------------------------------------
def test_train_one_epoch_core_behavior():
    """
    Tests:
      1) parameters update
      2) loss averaged per-example
      3) training decreases loss
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    torch.manual_seed(0)

    n = 33
    X = torch.randn(n, 2)

    # FIX: match (B,1) shape for BCEWithLogits
    y = (X[:, 0] + X[:, 1] > 0).float().unsqueeze(1)

    loader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=False)

    model = nn.Linear(2, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.2)

    device = torch.device("cpu")

    # ---- 1) parameters must change ----
    before = [p.detach().clone() for p in model.parameters()]

    loss1 = sol.train_one_epoch(model, loader, opt,
                                "logistic_regression", device)

    after = list(model.parameters())

    assert any(not torch.allclose(b, a) for b, a in zip(before, after)), \
        "Parameters did not update"

    # ---- 2) training decreases loss ----
    loss2 = sol.train_one_epoch(model, loader, opt,
                                "logistic_regression", device)

    assert loss2 < loss1

    # ---- 3) per-example averaging ----
    opt = torch.optim.SGD(model.parameters(), lr=0.0)

    true_loss = sol.compute_loss(model(X), y, "logistic_regression").item()

    returned_loss = sol.train_one_epoch(model, loader, opt,
                                       "logistic_regression", device)

    assert abs(true_loss - returned_loss) < 1e-6



def test_accuracy():
    """
    Loads saved models and evaluates them on the hidden/private test set.
    Tests both:
        - least squares model
        - logistic regression model
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # load hidden test data once
    # -------------------------
    test_path = "data/test.csv"
    X_test, y_test = sol.load_csv(test_path)

    model_files = [
        "mlp_least_squares.pt",
        "mlp_logistic_regression.pt",
    ]

    loss_thres = [0.1, 0.6]
    acc_thre = [0.75, 0.75]

    for idx, path in enumerate(model_files):
        # -------------------------
        # load checkpoint
        # -------------------------
        ckpt = torch.load(path, map_location=device)
        cfg = ckpt["config"]

        # -------------------------
        # rebuild model
        # -------------------------
        model = sol.MLP(
            input_dim=X_test.shape[1],
            hidden_dim=cfg["hidden_dim"],
        ).to(device)

        model.load_state_dict(ckpt["model_state"])
        model.eval()

        # -------------------------
        # apply saved scaling
        # -------------------------
        X_scaled = (X_test - ckpt["scaler_mean"]) / ckpt["scaler_scale"]

        # -------------------------
        # loader
        # -------------------------
        test_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_scaled, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.float32),
            ),
            batch_size=cfg["batch_size"],
            shuffle=False,
        )

        # -------------------------
        # evaluate
        # -------------------------
        test_loss, test_acc = sol.evaluate(
            model,
            test_loader,
            cfg["loss_name"],
            device,
        )

        # -------------------------
        # thresholds
        # -------------------------
        print(test_acc, test_loss)
        assert test_acc > acc_thre[idx], f"{path}: accuracy too low"
        assert test_loss < loss_thres[idx], f"{path}: loss too high"



# =========================================================
# Points allocation
# =========================================================

ALL_TESTS = [
    ("Model architecture", 5, test_model_architecture),
    ("Optimizer", 5, test_optimizer),
    ("Scaler", 5, test_scaler),
    ("Train loader shuffle", 5, test_train_loader_batch_and_drop_last),
    ("Training loop", 5, test_train_one_epoch_core_behavior),
    ("Evaluate()", 20, test_accuracy),
    #("Cross validation pipeline", 15, test_cross_validation_runs),
]
