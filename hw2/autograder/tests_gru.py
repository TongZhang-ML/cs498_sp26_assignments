import torch
import numpy as np
import os
from transformers import GPT2TokenizerFast

from gru import (
    tokenize_string,
    NonOverlappingNextTokenDataset,
    GRULanguageModel,
    train_one_epoch,
    generate_text,
    train_and_validate,
    make_loader,
    compute_ce_loss,
)

# ======================================================
# Helpers
# ======================================================

def tiny_ids():
    # deterministic toy tokens
    return list(range(50))


# ======================================================
# 1. tokenize_string
# ======================================================

def test_tokenize_string_basic():
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    ids = tokenize_string("hello world", tok)
    assert isinstance(ids, list)
    assert all(isinstance(x, int) for x in ids)
    assert len(ids) > 0


# ======================================================
# 2. Dataset
# ======================================================

def test_dataset_len():
    ids = list(range(33))
    ds = NonOverlappingNextTokenDataset(ids, seq_len=8)
    # floor((33-9)/8)+1 = 4
    assert len(ds) == 4


def test_dataset_item_shapes():
    ids = list(range(20))
    ds = NonOverlappingNextTokenDataset(ids, seq_len=5)

    x, y = ds[0]
    assert x.shape == (5,)
    assert y.shape == (5,)
    assert torch.equal(y[:-1], x[1:])


def test_dataset_non_overlap():
    ids = list(range(20))
    ds = NonOverlappingNextTokenDataset(ids, seq_len=5)

    x0, _ = ds[0]
    x1, _ = ds[1]
    assert not torch.any(x0 == x1)  # disjoint segments


# ======================================================
# 3. Model init + forward
# ======================================================

def test_model_forward_shapes():
    model = GRULanguageModel(
        vocab_size=100,
        emb_dim=16,
        hidden_dim=32,
        num_layers=2,
        dropout=0.1,
        pad_token_id=0,
    )

    x = torch.randint(0, 100, (4, 10))
    logits, h = model(x)

    assert logits.shape == (4, 10, 100)
    assert h.shape == (2, 4, 32)


def test_model_gradients():
    model = GRULanguageModel(50, 8, 16, 1, 0.0, 0)

    x = torch.randint(0, 50, (2, 5))
    y = torch.randint(0, 50, (2, 5))

    logits, _ = model(x)
    loss = compute_ce_loss(logits, y)
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)


# ======================================================
# 4. train_one_epoch
# ======================================================

def test_train_one_epoch_reduces_loss():
    ids = tiny_ids()

    loader = make_loader(ids, seq_len=4, batch_size=4, shuffle=False)

    model = GRULanguageModel(50, 8, 16, 1, 0.0, 0)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    device = torch.device("cpu")

    loss1 = train_one_epoch(model, loader, opt, device)
    loss2 = train_one_epoch(model, loader, opt, device)

    # should not increase dramatically
    assert loss2 <= loss1 + 0.5


# ======================================================
# 5. generate_text
# ======================================================

def test_generate_text_runs():
    tok = GPT2TokenizerFast.from_pretrained("gpt2")

    model = GRULanguageModel(tok.vocab_size, 8, 16, 1, 0.0, tok.eos_token_id)

    text = generate_text(
        model=model,
        tokenizer=tok,
        prompt="hello",
        max_new_tokens=5,
        temperature=1.0,
        device=torch.device("cpu"),
    )

    assert isinstance(text, str)
    assert len(text) > 0


def test_generate_empty_prompt_error():
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GRULanguageModel(tok.vocab_size, 8, 16, 1, 0.0, tok.eos_token_id)

    try:
        generate_text(model, tok, "", 5, 1.0, torch.device("cpu"))
        assert False
    except ValueError:
        pass


# ======================================================
# 6. train_and_validate
# ======================================================

def test_train_and_validate_small():
    tok = GPT2TokenizerFast.from_pretrained("gpt2")

    ids = tiny_ids()

    class Cfg:
        seq_len = 4
        batch_size = 4
        epochs = 1
        lr = 0.01
        weight_decay = 0.0

    model_cfg = {
        "name": "T",
        "emb_dim": 8,
        "hidden_dim": 16,
        "num_layers": 1,
        "dropout": 0.0,
    }

    model, train_loss, val_loss = train_and_validate(
        tokenizer=tok,
        pad_token_id=tok.eos_token_id,
        train_ids=ids,
        val_ids=ids,
        device=torch.device("cpu"),
        cfg=Cfg(),
        model_cfg=model_cfg,
    )

    assert isinstance(model, torch.nn.Module)
    assert np.isfinite(train_loss)
    assert np.isfinite(val_loss)
# ======================================================
# 7. test set evaluation (model.pth + test.txt)
# ======================================================

def test_test_loss_eval():
    device = torch.device("cpu")

    # -------------------------
    # files must exist
    # -------------------------
    if not os.path.isfile("gru_model.pt"):
        raise AssertionError("gru_model.pt not found")

    saved_dict = torch.load("gru_model.pt", map_location=device)
    config = saved_dict['config']
    state_dict = saved_dict['model_state_dict']


    # -------------------------
    # tokenizer
    # -------------------------
    tok = GPT2TokenizerFast.from_pretrained("gpt2")

    # -------------------------
    # read test text
    # -------------------------
    with open("test.txt", "r", encoding="utf-8") as f:
        text = f.read()

    ids = tokenize_string(text, tok)

    loader = make_loader(
        ids,
        seq_len=config['seq_len'],
        batch_size=8,
        shuffle=False
    )

    # -------------------------
    # load model
    # assumes checkpoint contains full model
    # -------------------------
    
    model = GRULanguageModel(
        vocab_size=tok.vocab_size,
        emb_dim=int(config["emb_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
        pad_token_id=config["pad_token_id"],
    )
    model.load_state_dict(state_dict)
    model.eval()

    # -------------------------
    # compute CE loss
    # -------------------------
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            logits, _ = model(x)
            loss = compute_ce_loss(logits, y)

            total_loss += loss.item()
            count += 1

    avg_loss = total_loss / max(count, 1)
    # sanity checks
    assert np.isfinite(avg_loss)
    assert avg_loss < 4.0


if __name__ == "__main__":
    test_test_loss_eval()

# ======================================================
# Gradescope format
# ======================================================

ALL_TESTS = [
    ("tokenizer implementation", 10, test_tokenize_string_basic),
    ("Dataset Implementation A", 3, test_dataset_len),
    ("Dataset Implementation B", 3, test_dataset_item_shapes),
    ("Dataset Implementation C", 4, test_dataset_non_overlap),
    ("Model Implementation A", 7, test_model_forward_shapes),
    ("Model Implementation B", 3, test_model_gradients),
    ("Training Implementation", 10, test_train_one_epoch_reduces_loss),
    ("Text Geneartion A", 8, test_generate_text_runs),
    ("Text Generation B", 2, test_generate_empty_prompt_error),
    ("Model selection Implementation ", 10, test_train_and_validate_small),
    ("Hidden Test", 10, test_test_loss_eval),

]
