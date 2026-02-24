#!/usr/bin/env python3
"""
Text generation: GPT-2 fine-tuning vs training from scratch (causal language modeling).

This script:
  - Loads ./data/train.txt and ./data/val.txt 
  - Builds fixed-length token chunks with attention masks 
  - Fine-tunes GPT-2 for multiple epoch settings
  - Trains a small Transformer LM from scratch for multiple epoch settings
  - Reports train/val losses and generates one sample per run using the prompt "A fox sits"
  - Prints LaTeX code (tables + samples) for copy-paste into the writeup

Note on padding:
  - The dataset builder pads the final chunk and sets attention_mask=0 on padded positions.
  - To ensure padded positions do not affect training loss, we set labels=-100 where attention_mask==0.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.trainer_utils import set_seed as hf_set_seed


# -------------------------
# Reproducibility
# -------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------
# Tokenizer + dataset
# -------------------------

def make_tokenizer(model_name: str = "gpt2") -> GPT2TokenizerFast:
    tok = GPT2TokenizerFast.from_pretrained(model_name)
    tok.pad_token = tok.eos_token  # GPT-2 has no pad token by default
    return tok

def _tokenize_and_chunk_with_padding(tokenizer: GPT2TokenizerFast, context_length: int):
    """
    Provided helper:
      - Tokenize each line (empty lines are kept as "")
      - Concatenate into one token stream while preserving sequence:
          * between consecutive non-empty lines: insert a single space
          * for a run of empty lines: insert a single EOS token
      - Split into non-overlapping chunks of length context_length
      - Pad last chunk with pad_token_id and set attention_mask=0 on padded positions
    """
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    # Precompute a "space" separator (avoids gluing words across wrapped lines)
    space_ids = tokenizer(" ", add_special_tokens=False)["input_ids"]

    def _fn(examples: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        texts = examples["text"]

        # Tokenize non-empty lines only; keep alignment with texts via an index map
        nonempty_idx: List[int] = []
        nonempty_texts: List[str] = []
        for i, s in enumerate(texts):
            if s != "":
                nonempty_idx.append(i)
                nonempty_texts.append(s)

        tok_nonempty = tokenizer(nonempty_texts, padding=False, add_special_tokens=False)

        ids: List[int] = []
        am: List[int] = []

        j = 0  # index into tok_nonempty
        prev_was_empty = True  # treat start as boundary

        for i, s in enumerate(texts):
            if s == "":
                # Collapse runs of empty lines to a single EOS boundary
                if not prev_was_empty:
                    ids.append(eos_id)
                    am.append(1)
                    prev_was_empty = True
                continue

            # Non-empty line
            input_ids = tok_nonempty["input_ids"][j]
            attn = tok_nonempty["attention_mask"][j]
            j += 1

            # If previous line was non-empty, add a space to avoid word-joining
            if not prev_was_empty and len(space_ids) > 0:
                ids.extend(space_ids)
                am.extend([1] * len(space_ids))

            ids.extend(input_ids)
            am.extend(attn)
            prev_was_empty = False

        # Split into fixed chunks
        chunks = [ids[i : i + context_length] for i in range(0, len(ids), context_length)]
        attn_chunks = [am[i : i + context_length] for i in range(0, len(am), context_length)]

        if len(chunks) == 0:
            chunks = [[pad_id] * context_length]
            attn_chunks = [[0] * context_length]

        if len(chunks[-1]) < context_length:
            pad_len = context_length - len(chunks[-1])
            chunks[-1] = chunks[-1] + [pad_id] * pad_len
            attn_chunks[-1] = attn_chunks[-1] + [0] * pad_len

        return {"input_ids": chunks, "attention_mask": attn_chunks}

    return _fn 

def build_lm_datasets_from_split(
    train_path: str,
    val_path: str,
    tokenizer: GPT2TokenizerFast,
    context_length: int,
):
    raw_train = load_dataset("text", data_files={"train": train_path})["train"]
    raw_val = load_dataset("text", data_files={"val": val_path})["val"]

    chunk_fn = _tokenize_and_chunk_with_padding(tokenizer, context_length=context_length)
    train_ds = raw_train.map(chunk_fn, batched=True, remove_columns=["text"])
    val_ds = raw_val.map(chunk_fn, batched=True, remove_columns=["text"])
    return train_ds, val_ds


# -------------------------
# Collator: fixed length + label masking
# -------------------------

class CausalLMCollatorFixedLen:
    """
    Collator for datasets that already contain fixed-length:
      - input_ids: List[int] of length context_length
      - attention_mask: List[int] of length context_length

    It stacks tensors and creates labels=input_ids, then sets labels=-100 where attention_mask==0,
    so padded positions do not contribute to loss.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        attention_mask = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# -------------------------
# Scratch model config
# -------------------------

@dataclass(frozen=True)
class ScratchConfig:
    n_embd: int = 256
    n_layer: int = 4
    n_head: int = 4
    context_length: int = 64


# -------------------------
# Custom Transformer LM (scratch)
# -------------------------

class CustomGPT2LM(PreTrainedModel):
    """
    A small GPT-2-style causal LM built from GPT2Block.
    Returns {"loss","logits"} so it works with Hugging Face Trainer.
    """
    config_class = GPT2Config

    def __init__(self, config: GPT2Config):
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embeddings = nn.Embedding(config.n_positions, config.n_embd)
        self.blocks = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)])
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.init_weights()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        pos = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
        h = self.embeddings(input_ids) + self.position_embeddings(pos)

        attn = None
        if attention_mask is not None:
            # HF GPT-2 blocks expect additive mask with shape [bsz, 1, 1, seq]
            attn = attention_mask.unsqueeze(1).unsqueeze(2).to(h.dtype)
            attn = (1.0 - attn) * -10000.0

        for blk in self.blocks:
            h = blk(h, attention_mask=attn)[0]

        logits = self.lm_head(h)

        loss = None
        if labels is not None:
            # Standard next-token prediction shift
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return {"loss": loss, "logits": logits}


def build_scratch_model(cfg: ScratchConfig, vocab_size: int, pad_token_id: int) -> PreTrainedModel:
    """
    Implement: Construct a small causal Transformer language model from scratch.

    Inputs:
      - cfg: ScratchConfig with fields n_embd, n_layer, n_head, context_length
      - vocab_size: tokenizer vocabulary size
      - pad_token_id: tokenizer pad token id (EOS)

    Requirements:
      1) Create a GPT2Config using cfg:
         - vocab_size=vocab_size
         - n_positions=cfg.context_length
         - n_ctx=cfg.context_length
         - n_embd=cfg.n_embd
         - n_layer=cfg.n_layer
         - n_head=cfg.n_head
      2) Construct model using CustomGPT2LM
      3) Set model.config.pad_token_id to pad_token_id
      4) Return model
    """
    cfg_hf = GPT2Config(
        vocab_size=vocab_size,
        n_positions=cfg.context_length,
        n_ctx=cfg.context_length,
        n_embd=cfg.n_embd,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
    )
    model = CustomGPT2LM(cfg_hf)
    model.config.pad_token_id = pad_token_id
    return model


def train_scratch_model(
    model: PreTrainedModel,
    tokenizer: GPT2TokenizerFast,
    train_ds,
    val_ds,
    run_cfg: "RunConfig",
    epochs: int,
    run_dir: str,
) -> Tuple[Trainer, Dict[str, float]]:
    """
    Implement: Train and evaluate the scratch model using Hugging Face Trainer.

    Inputs:
      - model: scratch model returned by build_scratch_model
      - tokenizer: GPT-2 tokenizer
      - train_ds, val_ds: datasets with columns input_ids, attention_mask
      - run_cfg: RunConfig with hyperparameters (batch sizes, lr, etc.)
      - epochs: number of training epochs for this run
      - run_dir: output directory for this run (checkpoints/logs)

    Requirements:
      1) Use a collator that sets labels=-100 on padded positions (attention_mask==0).
      2) Create TrainingArguments with:
         - output_dir=run_dir
         - overwrite_output_dir=True
         - per_device_train_batch_size=run_cfg.batch_size
         - per_device_eval_batch_size=run_cfg.eval_batch_size
         - num_train_epochs=epochs
         - learning_rate=run_cfg.lr_scratch
         - weight_decay=run_cfg.weight_decay
         - evaluation_strategy="steps", eval_steps=run_cfg.eval_steps
         - save_strategy="steps", save_steps=run_cfg.save_steps
         - logging_steps=run_cfg.logging_steps
         - report_to=[]
         - seed=run_cfg.seed
         - dataloader_num_workers=0
         - fp16=torch.cuda.is_available()
      3) Create a Trainer, call trainer.train().
      4) Evaluate on training set and validation set:
         - train_m = trainer.evaluate(eval_dataset=train_ds, metric_key_prefix="train")
         - val_m = trainer.evaluate()
      5) Return (trainer, {"train_loss": ..., "eval_loss": ...}).
    """
    collator = CausalLMCollatorFixedLen()

    targs = TrainingArguments(
        output_dir=run_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=run_cfg.batch_size,
        per_device_eval_batch_size=run_cfg.eval_batch_size,
        num_train_epochs=epochs,
        learning_rate=run_cfg.lr_scratch,
        weight_decay=run_cfg.weight_decay,
        evaluation_strategy="steps",
        eval_steps=run_cfg.eval_steps,
        save_strategy="steps",
        save_steps=run_cfg.save_steps,
        logging_steps=run_cfg.logging_steps,
        report_to=[],
        seed=run_cfg.seed,
        dataloader_num_workers=0,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    train_m = trainer.evaluate(eval_dataset=train_ds, metric_key_prefix="train")
    val_m = trainer.evaluate()

    return trainer, {
        "train_loss": float(train_m["train_loss"]),
        "eval_loss": float(val_m["eval_loss"]),
    }


# -------------------------
# Generation + LaTeX report
# -------------------------
@torch.no_grad()
def generate_text(
    model: torch.nn.Module,
    tokenizer: GPT2TokenizerFast,
    prompt_text: str,
    max_new_tokens: int,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 50,
    stop_on_eos: bool = True,
) -> str:
    """
    Causal generation with nucleus (top-p) and optional top-k sampling.
    EOS is allowed and treated as a real boundary token.

    If stop_on_eos=True, generation stops when EOS is sampled.
    """
    device = next(model.parameters()).device
    model.eval()

    enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)

    eos_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        out = model(input_ids=input_ids)
        logits = out["logits"] if isinstance(out, dict) else out.logits
        next_logits = logits[:, -1, :] / max(temperature, 1e-6)

        # Top-k filtering
        if top_k and top_k > 0:
            topk_vals, _ = torch.topk(next_logits, k=min(top_k, next_logits.size(-1)), dim=-1)
            kth = topk_vals[:, -1].unsqueeze(-1)
            next_logits = torch.where(next_logits < kth, torch.full_like(next_logits, -1e10), next_logits)

        # Top-p (nucleus) filtering
        if 0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(next_logits, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)

            mask = cum_probs > top_p
            mask[:, 0] = False  # keep at least one token

            sorted_logits = torch.where(mask, torch.full_like(sorted_logits, -1e10), sorted_logits)

            next_logits = torch.full_like(next_logits, -1e10)
            next_logits.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)

        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        if stop_on_eos and eos_id is not None and int(next_id.item()) == int(eos_id):
            break

        input_ids = torch.cat([input_ids, next_id], dim=1)

    # Decode and convert EOS to readable paragraph breaks
    text = tokenizer.decode(input_ids[0], skip_special_tokens=False)

    if eos_id is not None and tokenizer.eos_token is not None:
        text = text.replace(tokenizer.eos_token, "\n\n")

    text = " ".join(text.split())  # collapse weird spacing
    return text

def latex_escape(s: str) -> str:
    return (
        s.replace("\\", r"\textbackslash ")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde ")
        .replace("^", r"\textasciicircum ")
    )


def print_latex_report(
    finetune_rows: List[Tuple[int, float, float, str]],
    scratch_rows: List[Tuple[int, float, float, str]],
    prompt: str,
) -> None:
    """
    Print a LaTeX snippet (tables + generations). Each row is:
      (epochs, train_loss, val_loss, sample_text)
    """
    print("\n=== LaTeX (copy-paste) ===")
    print(r"\begin{solution}")
    print("")

    print(r"\noindent \textbf{GPT-2 fine-tune (different epochs).}")
    print(r"\begin{tabular}{lcc}")
    print(r"\hline")
    print(r"Epochs & Train loss & Val loss \\")
    print(r"\hline")
    for e, tr, va, _ in finetune_rows:
        print(rf"{e} & {tr:.4f} & {va:.4f} \\")
    print(r"\hline")
    print(r"\end{tabular}")
    print("")

    print(r"\noindent \textbf{Scratch Transformer (different epochs).}")
    print(r"\begin{tabular}{lcc}")
    print(r"\hline")
    print(r"Epochs & Train loss & Val loss \\")
    print(r"\hline")
    for e, tr, va, _ in scratch_rows:
        print(rf"{e} & {tr:.4f} & {va:.4f} \\")
    print(r"\hline")
    print(r"\end{tabular}")
    print("")

    print(r"\medskip")
    print(rf"\noindent Prompt: {latex_escape(prompt)}\\")
    print("")

    print(r"\noindent \textbf{Generated samples (GPT-2 fine-tune).}\\")
    for e, _, _, sample in finetune_rows:
        print(rf"\noindent Epochs {e}: {latex_escape(sample)}\\")
    print("")

    print(r"\noindent \textbf{Generated samples (Scratch Transformer).}\\")
    for e, _, _, sample in scratch_rows:
        print(rf"\noindent Epochs {e}: {latex_escape(sample)}\\")
    print("")

    print(r"\end{solution}")


# -------------------------
# Experiment runners
# -------------------------

def finetune_run(
    run_cfg: "RunConfig",
    tokenizer: GPT2TokenizerFast,
    train_ds,
    val_ds,
    epochs: int,
    run_dir: str,
) -> Tuple[float, float, str]:
    model = GPT2LMHeadModel.from_pretrained(run_cfg.pretrained_name)
    model.config.pad_token_id = tokenizer.pad_token_id

    collator = CausalLMCollatorFixedLen()

    targs = TrainingArguments(
        output_dir=run_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=run_cfg.batch_size,
        per_device_eval_batch_size=run_cfg.eval_batch_size,
        num_train_epochs=epochs,
        learning_rate=run_cfg.lr_finetune,
        weight_decay=run_cfg.weight_decay,
        evaluation_strategy="steps",
        eval_steps=run_cfg.eval_steps,
        save_strategy="steps",
        save_steps=run_cfg.save_steps,
        logging_steps=run_cfg.logging_steps,
        report_to=[],
        seed=run_cfg.seed,
        dataloader_num_workers=0,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    train_m = trainer.evaluate(eval_dataset=train_ds, metric_key_prefix="train")
    val_m = trainer.evaluate()
    train_loss = float(train_m["train_loss"])
    val_loss = float(val_m["eval_loss"])

    sample = generate_text(
        trainer.model,
        tokenizer,
        prompt_text=run_cfg.prompt,
        max_new_tokens=run_cfg.max_new_tokens,
        temperature=run_cfg.temperature,
    )
    return train_loss, val_loss, sample


def scratch_run(
    run_cfg: "RunConfig",
    tokenizer: GPT2TokenizerFast,
    train_ds,
    val_ds,
    epochs: int,
    run_dir: str,
) -> Tuple[float, float, str]:
    cfg = ScratchConfig(
        n_embd=run_cfg.scratch_embd,
        n_layer=run_cfg.scratch_layers,
        n_head=run_cfg.scratch_heads,
        context_length=run_cfg.context_length,
    )
    model = build_scratch_model(cfg=cfg, vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id)

    trainer, metrics = train_scratch_model(
        model=model,
        tokenizer=tokenizer,
        train_ds=train_ds,
        val_ds=val_ds,
        run_cfg=run_cfg,
        epochs=epochs,
        run_dir=run_dir,
    )

    train_loss = float(metrics["train_loss"])
    val_loss = float(metrics["eval_loss"])

    sample = generate_text(
        trainer.model,
        tokenizer,
        prompt_text=run_cfg.prompt,
        max_new_tokens=run_cfg.max_new_tokens,
        temperature=run_cfg.temperature,
    )
    return train_loss, val_loss, sample


# -------------------------
# Config (no command line arguments)
# -------------------------

@dataclass
class RunConfig:
    # data
    train_path: str = "./data/train.txt"
    val_path: str = "./data/val.txt"

    # pretrained model
    pretrained_name: str = "gpt2"
    output_dir: str = "./outputs/textgen"

    # sequence
    context_length: int = 64

    # optimization
    seed: int = 42
    batch_size: int = 16
    eval_batch_size: int = 16

    # epoch sweeps
    finetune_epochs: List[int] = field(default_factory=lambda: [1, 3, 5])
    scratch_epochs: List[int] = field(default_factory=lambda: [5, 10, 15])

    lr_finetune: float = 1e-4
    lr_scratch: float = 3e-4
    weight_decay: float = 0.0

    # scratch model size
    scratch_embd: int = 256
    scratch_layers: int = 4
    scratch_heads: int = 4

    # trainer bookkeeping
    logging_steps: int = 50
    eval_steps: int = 200
    save_steps: int = 200

    # generation demo
    prompt: str = "A fox sits"
    max_new_tokens: int = 60
    temperature: float = 0.8


def get_config() -> RunConfig:
    return RunConfig()


def main() -> None:
    cfg = get_config()
    set_seed(cfg.seed)

    os.makedirs(cfg.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    if not os.path.exists(cfg.train_path) or not os.path.exists(cfg.val_path):
        raise FileNotFoundError("Could not find ./data/train.txt and ./data/val.txt.")

    tokenizer = make_tokenizer(cfg.pretrained_name)
    print(f"vocab_size={len(tokenizer)} pad_token_id={tokenizer.pad_token_id}")

    train_ds, val_ds = build_lm_datasets_from_split(
        train_path=cfg.train_path,
        val_path=cfg.val_path,
        tokenizer=tokenizer,
        context_length=cfg.context_length,
    )

    finetune_rows: List[Tuple[int, float, float, str]] = []
    scratch_rows: List[Tuple[int, float, float, str]] = []

    for e in cfg.finetune_epochs:
        run_dir = os.path.join(cfg.output_dir, f"finetune_e{e}")
        print(f"\n=== GPT-2 fine-tune: {e} epochs ===")
        tr_loss, va_loss, sample = finetune_run(
            cfg, tokenizer, train_ds, val_ds,
            epochs=e, run_dir=run_dir
        )
        finetune_rows.append((e, tr_loss, va_loss, sample))
        print(f"train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}")
        print(sample)

    for e in cfg.scratch_epochs:
        run_dir = os.path.join(cfg.output_dir, f"scratch_e{e}")
        print(f"\n=== Scratch Transformer: {e} epochs ===")
        tr_loss, va_loss, sample = scratch_run(
            cfg, tokenizer, train_ds, val_ds,
            epochs=e, run_dir=run_dir
        )
        scratch_rows.append((e, tr_loss, va_loss, sample))
        print(f"train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}")
        print(sample)

    print_latex_report(
        finetune_rows=finetune_rows,
        scratch_rows=scratch_rows,
        prompt=cfg.prompt,
    )


if __name__ == "__main__":
    main()
