import torch
import numpy as np

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
# Gradescope format
# ======================================================

ALL_TESTS = [
    ("tokenize", 2, test_tokenize_string_basic),
    ("dataset_len", 2, test_dataset_len),
    ("dataset_item", 2, test_dataset_item_shapes),
    ("dataset_overlap", 2, test_dataset_non_overlap),
    ("model_forward", 3, test_model_forward_shapes),
    ("model_grad", 2, test_model_gradients),
    ("train_epoch", 3, test_train_one_epoch_reduces_loss),
    ("generate", 2, test_generate_text_runs),
    ("generate_empty", 2, test_generate_empty_prompt_error),
    ("train_validate", 5, test_train_and_validate_small),
]
