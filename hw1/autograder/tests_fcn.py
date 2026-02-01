import torch
import numpy as np

# This imports the student's submitted file
import fcn_solution as sol


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
def test_train_loader_shuffle():
    X, y = make_toy_data()

    loader = sol.make_train_loader(X, y, batch_size=16)

    # DataLoader does not expose shuffle flag directly.
    # Instead we inspect the sampler type.
    from torch.utils.data import RandomSampler

    assert isinstance(loader.sampler, RandomSampler), \
        "Training loader should shuffle (use RandomSampler)"


# ---------------------------------------------------------
# 5. Training reduces loss
# ---------------------------------------------------------
def test_training_step():
    torch.manual_seed(0)

    X, y = make_toy_data()
    loader = sol.make_train_loader(X, y, batch_size=32)

    device = torch.device("cpu")

    model = sol.MLP(6, 16)
    opt = sol.make_optimizer(model.parameters(), lr=0.1, weight_decay=0.0)

    loss1 = sol.train_one_epoch(model, loader, opt, "logistic_regression", device)
    loss2 = sol.train_one_epoch(model, loader, opt, "logistic_regression", device)

    assert loss2 < loss1, "Loss should decrease after training"


# ---------------------------------------------------------
# 6. Evaluate works
# ---------------------------------------------------------
def test_evaluate():
    X, y = make_toy_data()

    loader = sol.make_test_loader(X, y, batch_size=32)

    model = sol.MLP(6, 8)

    loss, acc = sol.evaluate(model, loader, "logistic_regression", torch.device("cpu"))

    assert 0 <= acc <= 1, "Accuracy must be in [0,1]"
    assert loss >= 0, "Loss must be non-negative"


# ---------------------------------------------------------
# 7. Cross-validation pipeline runs
# ---------------------------------------------------------
def test_cross_validation_runs():
    X, y = make_toy_data(120)

    cfg = sol.TrainConfig(
        loss_name="logistic_regression",
        weight_decay=0.0,
        hidden_dim=8,
        batch_size=16,
        epochs=1,
        seed=0
    )

    best_lr, lr_map, latex = sol.cross_validation(
        X,
        y,
        learning_rates=[0.01, 0.1],
        cfg=cfg,
        k_folds=2,
        device=torch.device("cpu")
    )

    assert best_lr in lr_map
    assert isinstance(latex, str)


# =========================================================
# Points allocation
# =========================================================

ALL_TESTS = [
    ("Model architecture", 50, test_model_architecture),
    #("Optimizer", 10, test_optimizer),
    #("Scaler", 10, test_scaler),
    #("Train loader shuffle", 10, test_train_loader_shuffle),
    #("Training loop", 25, test_training_step),
    #("Evaluate()", 10, test_evaluate),
    #("Cross validation pipeline", 15, test_cross_validation_runs),
]
