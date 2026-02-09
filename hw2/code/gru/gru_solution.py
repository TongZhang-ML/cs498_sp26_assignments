################################################################
#
# A small GRU language model trained for next-token prediction.
#
# Please refer to the code provided with the lecture notes as a related reference implementation.
#
# Data files
# ----------
# - train.txt : provided to students
# - val.txt   : provided to students (for model selection)
# - test.txt  : hidden from students and evaluated on autograder
#
# Expected workflow
# -----------------
# 1) Students run a small sweep over a few preset model configs on train.txt,
#    evaluate loss on val.txt, and pick the best.
# 2) Students save the chosen model checkpoint (a single .pt file).
# 3) Autograder loads that checkpoint and evaluates loss on hidden test.txt.
#
################################################################


from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import GPT2TokenizerFast


# ----------------------------
# Reproducibility
# ----------------------------

def set_seed(seed: int) -> None:
    """
    Set seeds for Python, NumPy, and PyTorch.

    Notes
    -----
    - Determinism on GPU is best-effort; some ops can still be non-deterministic
      depending on your CUDA / PyTorch version and enabled kernels.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # ok if no cuda
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------
# Data
# ----------------------------

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def tokenize_string(text: str, tokenizer: GPT2TokenizerFast) -> List[int]:
    """
    Implement according to the following description.

    Convert a text string into a list of integer token IDs using a GPT-2 tokenizer.

    This function should use the provided GPT-2 tokenizer to perform tokenization.
    The returned list represents the exact token ID sequence produced by the
    tokenizer for the given input string.

    Notes
    -----
    - The mapping from text to integers is fully determined by the tokenizer
      (its vocabulary and BPE merge rules).
    - For grading, the autograder will call this function directly on hidden text.
      Therefore, all text-to-ID conversion should go through this function.

    Parameters
    ----------
    text : str
        Input text to be tokenized.
    tokenizer : GPT2TokenizerFast
        A GPT-2 tokenizer instance created by
        GPT2TokenizerFast.from_pretrained("gpt2").

    Returns
    -------
    List[int]
        A list of integer token IDs corresponding to the input text.
    """
    return tokenizer.encode(text, add_special_tokens=False)


def tokenize_text(tokenizer: GPT2TokenizerFast, text: str) -> List[int]:
    """
    Tokenize a long text file into GPT-2 token ids.

    This is just a convenience wrapper that calls tokenize_string().
    """
    return tokenize_string(text=text, tokenizer=tokenizer)



class NonOverlappingNextTokenDataset(Dataset):
    """
    Non-overlapping dataset for next-token prediction.

    We are given a token-id list t[0], t[1], ..., t[N-1] where each token id is in
    {0, 1, ..., V-1}. For a fixed sequence length L (= seq_len), we construct
    training examples using non-overlapping segments with stride L:

    start = k * L
    x = t[start : start + L]
    y = t[start + 1 : start + L + 1]

    Thus each example is a pair (x, y) where x and y are both length-L sequences of
    integers, and y is the one-step (next-token) shift of x within the same segment.

    Notes
    ---------
    - Non-overlapping means that different examples use disjoint x segments (no
    shared tokens across different k, except for the one-token shift needed to
    form y).
    - Only indices k for which both slices are valid are used. Equivalently, we
    require start + L + 1 <= N so that y has length L.
    - This design is used in the homework to avoid the strong correlation introduced
    by a stride-1 sliding window, while keeping the implementation simple and the
    dataset size manageable.
    """
    def __init__(self, token_ids: List[int], seq_len: int):
        # Implement according to the class description
        #
        self.token_ids = token_ids
        self.seq_len = seq_len

        # Need start + seq_len + 1 <= N
        N = len(token_ids)
        if N < (seq_len + 1):
            self.num_examples = 0
        else:
            self.num_examples = 1 + (N - (seq_len + 1)) // seq_len

    def __len__(self) -> int:
        # Implement according to the class description
        #        
        return self.num_examples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Implement according to the class description
        #
        start = idx * self.seq_len
        x = torch.tensor(self.token_ids[start: start + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.token_ids[start + 1: start + self.seq_len + 1], dtype=torch.long)
        return x, y


def make_loader(token_ids: List[int], seq_len: int, batch_size: int, shuffle: bool) -> DataLoader:
    ds = NonOverlappingNextTokenDataset(token_ids, seq_len=seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


# ----------------------------
# Model
# ----------------------------

class GRULanguageModel(nn.Module):
    """
    A simple GRU-based language model for next-token prediction.

    The model has the following structure:
        token ids -> Embedding -> multi-layer GRU -> Linear -> logits over vocabulary

    The GRU may have multiple stacked layers, controlled by ``num_layers``.
    Dropout is applied between GRU layers when ``num_layers > 1``.

    This model is trained with cross-entropy loss on next-token prediction,
    where the target sequence is a one-step shift of the input sequence.

    Shapes
    ------
    Input
        x : (B, L)
            Batch of token ID sequences, where
            B is the batch size and L is the sequence length.

    Intermediate
        emb : (B, L, E)
            Embedded token representations, where E = emb_dim.
        h : (B, L, H)
            GRU hidden states for all time steps, where H = hidden_dim.

    Output
        logits : (B, L, V)
            Unnormalized log-probabilities over the vocabulary for each
            position in the sequence, where V = vocab_size.

    Notes
    -----
    - The GRU is created with ``batch_first=True``.
    - The final Linear layer is applied to each time step independently.
    - Padding tokens (if any) should be ignored in the loss using ``pad_token_id``.
    """
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        pad_token_id: int,
    ):
        """
        Implement this function according to the class description.
        """    
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.pad_token_id = pad_token_id

        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
        )
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self, x: torch.Tensor, h0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Implement this function according to the class description.

        Forward pass of the GRU language model.

        Given a batch of token IDs, this function computes token embeddings,
        applies the (possibly multi-layer) GRU, and maps the GRU outputs to
        vocabulary logits for next-token prediction.

        Computation
        -----------
            1) Look up token embeddings using the embedding layer.
            2) Pass the embedded sequence through the GRU.
            - If ``h0`` is provided, it is used as the initial hidden state.
            - If ``h0`` is None, the GRU initializes the hidden state to zeros.
            3) Apply a linear layer to the GRU outputs at each time step to produce
            logits over the vocabulary.

        Shapes
        ------
        Input
            x : (B, L)
                Batch of token ID sequences, where
                B is the batch size and L is the sequence length.
            h0 : Optional[(num_layers, B, H)]
                 Initial hidden state for the GRU, where H = hidden_dim.
                 If None, the hidden state is initialized internally.

        Output
            logits : (B, L, V)
                 Unnormalized log-probabilities over the vocabulary for each
                 time step, where V = vocab_size.
            hn : (num_layers, B, H)
                  Final hidden state of the GRU after processing the full sequence.

        Returns
        -------
            Tuple[torch.Tensor, torch.Tensor]
                A tuple ``(logits, hn)`` containing the vocabulary logits and the
              final hidden state.

        Notes
        -----
            - The GRU is created with ``batch_first=True``.
            - The linear layer is applied independently at each time step.
            - The returned hidden state ``hn`` can be reused for autoregressive
              generation.
        """
        emb = self.embed(x)
        out, hn = self.gru(emb, h0)
        logits = self.lm_head(out)
        return logits, hn


def count_parameters(model: nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)


# ----------------------------
# Loss / Eval / Train
# ----------------------------

def compute_ce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy loss for next-token prediction.

    logits:  (B, T, V)
    targets: (B, T)

    Returns mean loss over all tokens in the batch.
    """
    B, T, V = logits.shape
    return F.cross_entropy(logits.reshape(B * T, V), targets.reshape(B * T))


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits, _ = model(xb)
        loss = compute_ce_loss(logits, yb)

        total_loss += float(loss.item()) * int(yb.numel())
        total_tokens += int(yb.numel())

    if total_tokens == 0:
        return float("nan")
    return total_loss / total_tokens


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: Optional[float],
) -> float:
    """
    Implement according to the following description.


    Train a language model for one full epoch and return the average
    cross-entropy loss per token.

    This function performs one pass over the training DataLoader.
    The model is trained using next-token prediction with cross-entropy
    loss, where loss is averaged over *all tokens* seen in the epoch
    (not averaged per batch).

    Training procedure
    ------------------
    For each mini-batch (x, y) from the DataLoader:
      1. Move input tokens x and target tokens y to the specified device.
      2. Run the model forward pass to obtain logits of shape (B, L, V),
         where B is batch size, L is sequence length, and V is vocabulary size.
      3. Compute cross-entropy loss between logits and targets y.
      4. Clear previous gradients.
      5. Backpropagate the loss using PyTorch autograd.
      6. Optionally clip gradients to improve training stability.
      7. Update model parameters using the given optimizer.

    Loss accounting
    ----------------
    The loss returned by this function is the total negative log-likelihood
    over all tokens divided by the total number of tokens processed:

        average_loss = (sum of per-token losses) / (number of tokens)

    This makes losses comparable across different batch sizes and sequence
    lengths.

    Parameters
    ----------
    model : nn.Module
        The GRU language model to be trained.
    loader : DataLoader
        DataLoader providing batches of (x, y), where
        x, y have shape (B, L) and y is the one-step shift of x.
    optimizer : torch.optim.Optimizer
        Optimizer used to update model parameters.
    device : torch.device
        Device on which training is performed (CPU or CUDA).
    grad_clip : Optional[float]
        Maximum gradient norm for gradient clipping.
        If None or non-positive, gradient clipping is disabled.

    Returns
    -------
    float
        Average cross-entropy loss per token over the entire epoch.
        Returns NaN if no tokens were processed.
    """
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits, _ = model(xb)
        loss = compute_ce_loss(logits, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        total_loss += float(loss.item()) * int(yb.numel())
        total_tokens += int(yb.numel())

    if total_tokens == 0:
        return float("nan")
    return total_loss / total_tokens


# ----------------------------
# Generation
# ----------------------------

@torch.no_grad()
def generate_text(
    model: GRULanguageModel,
    tokenizer: GPT2TokenizerFast,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
) -> str:
    """
    Implement according to the following description.

    Generate text autoregressively from a trained GRU language model.

    This function performs next-token generation starting from a given
    text prompt. Generation is autoregressive: at each step, the model
    predicts a distribution over the vocabulary for the next token,
    samples one token, appends it to the sequence, and feeds it back
    into the model.

    Generation strategy
    -------------------
    1) The input prompt is first tokenized using the GPT-2 tokenizer.
    2) The full prompt sequence is run through the GRU *once* to obtain
       the final hidden state corresponding to the last prompt token.
    3) Starting from the last token of the prompt, new tokens are
       generated one-by-one:
         - The current token and hidden state are passed to the model.
         - The model outputs logits for the next token.
         - Logits are scaled by the temperature parameter.
         - A token is sampled from the resulting probability distribution.
         - The sampled token is appended and used as input for the next step.

    Temperature
    -----------
    The ``temperature`` parameter controls randomness in sampling:
      - temperature = 1.0 : standard sampling from the model distribution
      - temperature < 1.0 : sharper distribution, more deterministic output
      - temperature > 1.0 : flatter distribution, more random output
    To avoid numerical issues, temperature is clamped to a small positive value.

    Implementation requirements
    ----------------------------
    - The model should be in evaluation mode (``model.eval()``).
    - Gradients should NOT be computed during generation.
      (This function is decorated with ``@torch.no_grad()``.)
    - The hidden state returned by the GRU should be reused across time steps
      so that generation is efficient (do NOT re-run the full sequence
      at every step).
    - Token sampling should be done using ``torch.multinomial`` over a
      softmax-normalized probability vector.

    Parameters
    ----------
    model : GRULanguageModel
        A trained GRU language model.
    tokenizer : GPT2TokenizerFast
        GPT-2 tokenizer used to convert between text and token IDs.
    prompt : str
        Initial text prompt used to start generation.
    max_new_tokens : int
        Number of new tokens to generate after the prompt.
    temperature : float
        Sampling temperature controlling randomness.
    device : torch.device
        Device on which generation is performed.

    Returns
    -------
    str
        The generated text, consisting of the original prompt followed
        by the newly generated continuation.

    Errors
    ------
    ValueError
        Raised if the prompt tokenizes to an empty token sequence.

    Notes
    -----
    - Generation length is ``len(prompt_tokens) + max_new_tokens``.
    - This function is for qualitative evaluation and demonstration;
      generation quality does not directly affect the homework score,
      but the function should run correctly without errors.
    """
    model.eval()
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(prompt_ids) == 0:
        raise ValueError("Prompt produced no tokens.")

    x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

    # Run prompt to get hidden state
    _, h = model(x, h0=None)

    # Start from last token in prompt
    cur_id = x[:, -1:]  # (1, 1)
    generated = list(prompt_ids)

    for _ in range(max_new_tokens):
        logits, h = model(cur_id, h0=h)  # logits: (1, 1, V)
        next_logits = logits[0, 0, :] / max(float(temperature), 1e-8)
        probs = torch.softmax(next_logits, dim=0)

        next_id = int(torch.multinomial(probs, num_samples=1).item())
        generated.append(next_id)
        cur_id = torch.tensor([[next_id]], dtype=torch.long, device=device)

    return tokenizer.decode(generated)


# ----------------------------
# Checkpoint I/O
# ----------------------------

def save_checkpoint(path: str, model: GRULanguageModel, config: Dict) -> None:
    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": config,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    map_location: str | torch.device = "cpu",
) -> Tuple[GRULanguageModel, Dict]:
    ckpt = torch.load(path, map_location=map_location)
    config = ckpt["config"]

    model = GRULanguageModel(
        vocab_size=int(config["vocab_size"]),
        emb_dim=int(config["emb_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
        pad_token_id=int(config["pad_token_id"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model, config    


# ----------------------------
# Preset model configs (small + similar size)
# ----------------------------

def preset_model_configs() -> List[Dict]:
    """
    A few small GRU configs with similar parameter counts.
    """
    return [
        {"name": "A", "emb_dim": 192, "hidden_dim": 384, "num_layers": 1, "dropout": 0.1},
        {"name": "B", "emb_dim": 256, "hidden_dim": 256, "num_layers": 2, "dropout": 0.1},
        {"name": "C", "emb_dim": 128, "hidden_dim": 512, "num_layers": 1, "dropout": 0.1},
    ]


# ----------------------------
# Config + main
# ----------------------------

@dataclass
class RunConfig:
    # paths
    train_path: str = "data/train.txt"
    val_path: str = "data/val.txt"     # visible to students
    test_path: str = "data/test.txt"   # hidden test evaluated with autograder
    save_path: str = "outputs/gru_model.pt"
    load_path: Optional[str] = None

    # data / optimization
    seq_len: int = 32
    batch_size: int = 64
    epochs: int = 10
    lr: float = 2e-3
    weight_decay: float = 1e-3
    grad_clip: float = 1.0

    seed: int = 42

    # sweep / selection
    chosen_preset: str = "best"  

    # generation
    prompt: str = "Little Red-Cap said"
    gen_max_new_tokens: int = 64
    gen_temperature: float = 0.5

def train_and_validate(
    *,
    tokenizer: GPT2TokenizerFast,
    pad_token_id: int,
    train_ids: List[int],
    val_ids: List[int],
    device: torch.device,
    cfg: RunConfig,
    model_cfg: Dict,
) -> Tuple[GRULanguageModel, float, float]:
    """
    Implement according to the following description.

    Train one preset GRU language model on ``train_ids`` and evaluate on ``val_ids``.

    This function is called by the preset sweep in ``main()``. For each preset
    configuration in ``preset_model_configs()``, the sweep:
      1) builds a ``GRULanguageModel`` using hyperparameters from ``model_cfg``,
      2) constructs DataLoaders using the provided ``make_loader`` function,
      3) trains for ``cfg.epochs`` epochs by calling the student-implemented
         ``train_one_epoch``,
      4) evaluates validation loss by calling the provided ``evaluate_loss``,
      5) returns the trained model and the final train/val losses.

    DataLoader usage
    ----------------
    You should use the provided ``make_loader`` function to create loaders.
    In this homework, ``make_loader`` constructs the non-overlapping dataset
    (stride = ``seq_len``) and yields batches (x, y) with:
      - x shape (B, L) token IDs
      - y shape (B, L) next-token targets (one-step shift of x within each segment)

    Training
    --------
    Training should be performed by calling the helper function implemented earlier:
        train_loss = train_one_epoch(model=model, loader=train_loader, optimizer=optimizer,
                                     device=device, grad_clip=cfg.grad_clip)
    This function returns the average cross-entropy loss per token over the epoch.

    Evaluation
    ----------
    Validation loss should be computed by calling:
        val_loss = evaluate_loss(model=model, loader=val_loader, device=device)
    This returns the average cross-entropy loss per token over the validation loader.

    Logging
    -------
    - This function prints a short model summary (preset name, dimensions, parameter count).
    - It prints one line per epoch with training loss/token and elapsed time.
    - Batch-level printing is optional. If included, a coarse interval
      (for example, every 200 batches) is recommended. Students may choose
      a different interval.

    Parameters
    ----------
    tokenizer : GPT2TokenizerFast
        GPT-2 tokenizer; used here to set the vocabulary size.
    pad_token_id : int
        Token ID treated as padding (kept for completeness; may be used for masking).
    train_ids : List[int]
        Token IDs from train.txt.
    val_ids : List[int]
        Token IDs from val.txt. Validation is required; this must not be None.
    device : torch.device
        CPU or CUDA device for training and evaluation.
    cfg : RunConfig
        Run hyperparameters (seq_len, batch_size, epochs, lr, weight_decay, grad_clip, ...).
    model_cfg : Dict
        Preset model hyperparameters. Expected keys include:
        "name", "emb_dim", "hidden_dim", "num_layers", "dropout".

    Returns
    -------
    (model, train_loss, val_loss) : Tuple[GRULanguageModel, float, float]
        model : GRULanguageModel
            The trained model instance (left on ``device``).
        train_loss : float
            Final-epoch average cross-entropy loss per token returned by ``train_one_epoch``.
        val_loss : float
            Average cross-entropy loss per token returned by ``evaluate_loss`` on the
            validation loader.

    Consistency note
    ----------------
    The autograder may reload the saved checkpoint and recompute train and validation
    losses using the same ``make_loader`` and ``evaluate_loss`` definitions. For this
    reason, the loss definition and batching rule should remain unchanged.
    """
    if val_ids is None:
        raise ValueError("val_ids must be provided.")

    model = GRULanguageModel(
        vocab_size=tokenizer.vocab_size,
        emb_dim=int(model_cfg["emb_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        dropout=float(model_cfg["dropout"]),
        pad_token_id=pad_token_id,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    train_loader = make_loader(
        train_ids,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    val_loader = make_loader(
        val_ids,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    print(
        f"[model {model_cfg['name']}] emb={model_cfg['emb_dim']} hidden={model_cfg['hidden_dim']} "
        f"layers={model_cfg['num_layers']} params={count_parameters(model):,}"
    )
    print(f"[train] tokens={len(train_ids)}  examples={len(train_loader.dataset)} (non-overlapping)")

    train_loss = float("nan")
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=cfg.grad_clip,
        )
        dt = time.time() - t0
        print(f"[epoch {epoch:02d}] train loss/token = {train_loss:.4f}   time={dt:.1f}s")

    val_loss = evaluate_loss(model=model, loader=val_loader, device=device)
    print(f"[val] loss/token = {val_loss:.4f}")

    return model, train_loss, val_loss


# ----------------------------
# Main
# ----------------------------

def parse_args() -> RunConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=int, default=1)  # 1=train, 2=load+eval
    args = p.parse_args()

    cfg = RunConfig()

    if args.mode == 2:
        cfg.load_path = cfg.save_path  # reuse default save_path

    return cfg

def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"[device] {device}")

    tokenizer_name="gpt2"
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)

    # GPT-2 has no pad token by default; for fixed-length batches we do not need padding.
    # Keep pad_token_id in config for completeness and for loss masking if needed.
    pad_token_id = int(tokenizer.eos_token_id)

    # ------------------------------------------------------------
    # Mode 2 (autograder): load checkpoint, compute train/val/test
    # ------------------------------------------------------------
    if cfg.load_path is not None:
        model, saved_cfg = load_checkpoint(cfg.load_path, map_location=device)
        model.to(device)
        nparams = count_parameters(model)
        
        tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)

        print(f"[load] Loaded checkpoint from {cfg.load_path}")
        print(
            f"[loaded config] preset={saved_cfg.get('preset_name','?')} "
            f"emb={saved_cfg['emb_dim']} hidden={saved_cfg['hidden_dim']} layers={saved_cfg['num_layers']} "
            f"seq_len={saved_cfg.get('seq_len','?')}"
        )

        # Evaluate on train/val/test if files exist (train/val should exist for consistency).
        if not os.path.exists(cfg.train_path):
            raise FileNotFoundError(f"train file not found: {cfg.train_path}")
        if not os.path.exists(cfg.val_path):
            raise FileNotFoundError(f"val file not found: {cfg.val_path}")

        train_ids = tokenize_text(tokenizer, read_text(cfg.train_path))
        val_ids = tokenize_text(tokenizer, read_text(cfg.val_path))

        seq_len = int(saved_cfg.get("seq_len", cfg.seq_len))
        batch_size = int(getattr(cfg, "batch_size", 64))

        train_loader = make_loader(train_ids, seq_len=seq_len, batch_size=batch_size, shuffle=False)
        val_loader = make_loader(val_ids, seq_len=seq_len, batch_size=batch_size, shuffle=False)

        train_loss = evaluate_loss(model=model, loader=train_loader, device=device)
        val_loss = evaluate_loss(model=model, loader=val_loader, device=device)

        test_loss = float("nan")
        if os.path.exists(cfg.test_path):
            test_ids = tokenize_text(tokenizer, read_text(cfg.test_path))
            test_loader = make_loader(test_ids, seq_len=seq_len, batch_size=batch_size, shuffle=False)
            test_loss = evaluate_loss(model=model, loader=test_loader, device=device)
        else:
            # In autograder this should exist; locally for students it may not.
            print(f"[warn] test file not found (expected hidden): {cfg.test_path}")

        print(
            "\n[eval]\n"
            f"  number of model paramers = {nparams:,}\n"
            f"  train loss/token = {train_loss:.6f}\n"
            f"  val   loss/token = {val_loss:.6f}\n"
            f"  test  loss/token = {test_loss:.6f}"
        )

    # ------------------------------------------------------------
    # Mode 1 (student): train on train, sweep on val, save checkpoint
    # ------------------------------------------------------------
    else:
        if not os.path.exists(cfg.train_path):
            raise FileNotFoundError(f"train file not found: {cfg.train_path}")
        if not os.path.exists(cfg.val_path):
            raise FileNotFoundError(f"val file not found: {cfg.val_path}")

        train_ids = tokenize_text(tokenizer, read_text(cfg.train_path))
        val_ids = tokenize_text(tokenizer, read_text(cfg.val_path))

        presets = preset_model_configs()

        results = []
        best_val = float("inf")
        best_model = None
        best_save_config = None

        for mcfg in presets:
            model, train_loss, val_loss = train_and_validate(
                tokenizer=tokenizer,
                pad_token_id=pad_token_id,
                train_ids=train_ids,
                val_ids=val_ids,
                device=device,
                cfg=cfg,
                model_cfg=mcfg,
            )

            save_config = {
                "vocab_size": tokenizer.vocab_size,
                "emb_dim": int(mcfg["emb_dim"]),
                "hidden_dim": int(mcfg["hidden_dim"]),
                "num_layers": int(mcfg["num_layers"]),
                "dropout": float(mcfg["dropout"]),
                "pad_token_id": int(pad_token_id),
                "seq_len": int(cfg.seq_len),
                "preset_name": str(mcfg["name"]),
            }

            nparams = count_parameters(model)
            results.append((mcfg["name"], save_config["emb_dim"], save_config["hidden_dim"],
                            save_config["num_layers"], nparams, train_loss, val_loss))

            is_best = np.isfinite(val_loss) and (val_loss < best_val)

            if is_best:
                # Free previous best model before replacing it.
                if best_model is not None:
                    del best_model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                best_val = val_loss
                best_model = model
                best_save_config = dict(save_config)
            else:
                # Not best: free it.
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        if best_model is None or best_save_config is None:
            raise RuntimeError("Sweep failed to produce a finite validation loss.")

        # Print sweep results as LaTeX table
        print("\n[sweep results latex]")
        print(r"\begin{tabular}{ccccc}")
        print(r"\hline")
        print(r"Config & $(e,h,\text{layers})$ & \#params (approx.) & Train loss & Val loss \\")
        print(r"\hline")
        for name, e, h, layers, nparams, tr_loss, va_loss in results:
            print(
                f"{name} & $({e},{h},{layers})$ & {nparams:,} & "
                f"{tr_loss:.4f} & {va_loss:.4f} \\\\"
            )
        print(r"\hline")
        print(rf"\multicolumn{{5}}{{c}}{{Best config: {best_save_config['preset_name']}}} \\")
        print(r"\hline")
        print(r"\end{tabular}")

        print(
            f"\n[select] best preset = {best_save_config['preset_name']}  "
            f"val loss/token = {best_val:.4f}"
        )

        save_checkpoint(
            cfg.save_path,
            model=best_model,
            config=best_save_config,
        )
        print(f"[save] {cfg.save_path}")

        # In training mode, we generate from the best model we just trained.
        model = best_model

    # ------------------------
    # Generation (always)
    # ------------------------
    gen = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=cfg.prompt,
        max_new_tokens=cfg.gen_max_new_tokens,
        temperature=cfg.gen_temperature,
        device=device,
    )
    print("\n[generation]")
    print(gen)



if __name__ == "__main__":
    main()
