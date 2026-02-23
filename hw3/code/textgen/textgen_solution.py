#!/usr/bin/env python3
"""
Text generation: GPT-2 fine-tuning vs training from scratch (causal language modeling).

This script:
  - Loads ./data/train.txt and ./data/val.txt (one example per line)
  - Builds fixed-length token chunks with attention masks (EOS inserted between lines)
  - Fine-tunes GPT-2 for multiple epoch settings 
  - Trains a small Transformer LM from scratch for multiple epoch settings 
  - Reports train/val losses and generates one sample per run using the prompt "A fox sits"
  - Prints LaTeX code (tables + samples) for copy-paste into the writeup
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
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
    DataCollatorForLanguageModeling,
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
      - Tokenize each line
      - Concatenate lines into one token stream, inserting EOS between lines
      - Split into non-overlapping chunks of length context_length
      - Pad last chunk with pad_token_id and set attention_mask=0 on padded positions
    """
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    def _fn(examples: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        tok = tokenizer(examples["text"], padding=False)
        ids: List[int] = []
        am: List[int] = []

        for k, (input_ids, attn) in enumerate(zip(tok["input_ids"], tok["attention_mask"])):
            if k > 0:
                ids.append(eos_id)
                am.append(1)
            ids.extend(input_ids)
            am.extend(attn)

        chunks = [ids[i:i + context_length] for i in range(0, len(ids), context_length)]
        attn_chunks = [am[i:i + context_length] for i in range(0, len(am), context_length)]

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
    Returns {"loss","logits"} so it works with Hugging Face Trainer + LM collator.
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
            attn = attention_mask.unsqueeze(1).unsqueeze(2).to(h.dtype)
            attn = (1.0 - attn) * -10000.0

        for blk in self.blocks:
            h = blk(h, attention_mask=attn)[0]

        logits = self.lm_head(h)

        loss = None
        if labels is not None:
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
    args: argparse.Namespace,
    epochs: int,
    run_dir: str,
) -> Tuple[Trainer, Dict[str, float]]:
    """
    Implement: Train and evaluate the scratch model using Hugging Face Trainer.

    Inputs:
      - model: scratch model returned by build_scratch_model
      - tokenizer: GPT-2 tokenizer
      - train_ds, val_ds: datasets with columns input_ids, attention_mask
      - args: argparse Namespace with hyperparameters (batch sizes, lr, etc.)
      - epochs: number of training epochs for this run
      - run_dir: output directory for this run (checkpoints/logs)

    Requirements:
      1) Use DataCollatorForLanguageModeling with tokenizer and masked language model as False.
      2) Create TrainingArguments with:
         - output_dir=run_dir
         - overwrite_output_dir=True
         - per_device_train_batch_size=args.batch_size
         - per_device_eval_batch_size=args.eval_batch_size
         - num_train_epochs=epochs
         - learning_rate=args.lr_scratch
         - weight_decay=args.weight_decay
         - evaluation_strategy="steps", eval_steps=args.eval_steps
         - save_strategy="steps", save_steps=args.save_steps
         - logging_steps=args.logging_steps
         - report_to=[]
         - seed=args.seed
         - dataloader_num_workers=0
         - fp16=torch.cuda.is_available()
      3) Create a Trainer, call trainer.train().
      4) Evaluate on training set and validation set:
         - train_m = trainer.evaluate(eval_dataset=train_ds, metric_key_prefix="train")
         - val_m = trainer.evaluate()
      5) Return (trainer, {"train_loss": ..., "eval_loss": ...}).
    """
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    targs = TrainingArguments(
        output_dir=run_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=epochs,
        learning_rate=args.lr_scratch,
        weight_decay=args.weight_decay,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=0,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
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
    temperature: float,
    device: Optional[torch.device] = None,
) -> str:

    device=model.device
    model.eval()

    enc = tokenizer(prompt_text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    for _ in range(max_new_tokens):
        out = model(input_ids=input_ids)
        logits = out["logits"] if isinstance(out, dict) else out.logits
        next_logits = logits[:, -1, :] / (temperature + 1e-12)
        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_id], dim=1)

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def latex_escape(s: str) -> str:
    return (s.replace("\\", r"\textbackslash ")
             .replace("&", r"\&")
             .replace("%", r"\%")
             .replace("$", r"\$")
             .replace("#", r"\#")
             .replace("_", r"\_")
             .replace("{", r"\{")
             .replace("}", r"\}")
             .replace("~", r"\textasciitilde ")
             .replace("^", r"\textasciicircum "))


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

    # Table: finetune
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

    # Table: scratch
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

    # Samples
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
    args: argparse.Namespace,
    tokenizer: GPT2TokenizerFast,
    train_ds,
    val_ds,
    device: torch.device,
    epochs: int,
    run_dir: str,
) -> Tuple[float, float, str]:
    model = GPT2LMHeadModel.from_pretrained(args.pretrained_name)
    model.config.pad_token_id = tokenizer.pad_token_id

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    targs = TrainingArguments(
        output_dir=run_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=epochs,
        learning_rate=args.lr_finetune,
        weight_decay=args.weight_decay,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=0,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    trainer.train()

    train_m = trainer.evaluate(eval_dataset=train_ds, metric_key_prefix="train")
    val_m = trainer.evaluate()
    train_loss = float(train_m["train_loss"])
    val_loss = float(val_m["eval_loss"])

    sample = generate_text(
        trainer.model,
        tokenizer,
        prompt_text=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=device,
    )
    return train_loss, val_loss, sample


def scratch_run(
    args: argparse.Namespace,
    tokenizer: GPT2TokenizerFast,
    train_ds,
    val_ds,
    device: torch.device,
    epochs: int,
    run_dir: str,
) -> Tuple[float, float, str]:
    cfg = ScratchConfig(
        n_embd=args.scratch_embd,
        n_layer=args.scratch_layers,
        n_head=args.scratch_heads,
        context_length=args.context_length,
    )
    model = build_scratch_model(cfg=cfg, vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id)

    trainer, metrics = train_scratch_model(
        model=model,
        tokenizer=tokenizer,
        train_ds=train_ds,
        val_ds=val_ds,
        args=args,
        epochs=epochs,
        run_dir=run_dir,
    )

    train_loss = float(metrics["train_loss"])
    val_loss = float(metrics["eval_loss"])

    sample = generate_text(
        trainer.model,
        tokenizer,
        prompt_text=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=device,
    )
    return train_loss, val_loss, sample


# -------------------------
# CLI
# -------------------------

def parse_int_list(s: str) -> List[int]:
    # e.g. "1,2,3"
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--train_path", type=str, default="./data/train.txt")
    p.add_argument("--val_path", type=str, default="./data/val.txt")

    p.add_argument("--pretrained_name", type=str, default="gpt2")
    p.add_argument("--output_dir", type=str, default="./outputs/textgen")

    p.add_argument("--context_length", type=int, default=64)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=16)

    # Fine-tune epochs list 
    p.add_argument("--finetune_epochs", type=str, default="1,3,5")

    # Scratch epochs list
    p.add_argument("--scratch_epochs", type=str, default="5,10,15")

    # learning rates
    p.add_argument("--lr_finetune", type=float, default=1e-4)
    p.add_argument("--lr_scratch", type=float, default=3e-4)

    p.add_argument("--weight_decay", type=float, default=0.0)

    # scratch model size
    p.add_argument("--scratch_embd", type=int, default=256)
    p.add_argument("--scratch_layers", type=int, default=4)
    p.add_argument("--scratch_heads", type=int, default=4)

    # trainer bookkeeping
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=200)

    # generation demo
    p.add_argument("--prompt", type=str, default="A fox sits")
    p.add_argument("--max_new_tokens", type=int, default=60)
    p.add_argument("--temperature", type=float, default=0.8)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    if not os.path.exists(args.train_path) or not os.path.exists(args.val_path):
        raise FileNotFoundError("Could not find ./data/train.txt and ./data/val.txt.")

    tokenizer = make_tokenizer(args.pretrained_name)
    print(f"vocab_size={len(tokenizer)} pad_token_id={tokenizer.pad_token_id}")

    # Datasets
    train_ds, val_ds = build_lm_datasets_from_split(
        train_path=args.train_path,
        val_path=args.val_path,
        tokenizer=tokenizer,
        context_length=args.context_length,
    )

    finetune_epochs = parse_int_list(args.finetune_epochs)
    scratch_epochs = parse_int_list(args.scratch_epochs)

    finetune_rows: List[Tuple[int, float, float, str]] = []
    scratch_rows: List[Tuple[int, float, float, str]] = []

    # Fine-tune runs
    for e in finetune_epochs:
        run_dir = os.path.join(args.output_dir, f"finetune_e{e}")
        print(f"\n=== GPT-2 fine-tune: {e} epochs ===")
        tr_loss, va_loss, sample = finetune_run(args, tokenizer, train_ds, val_ds, device, epochs=e, run_dir=run_dir)
        finetune_rows.append((e, tr_loss, va_loss, sample))
        print(f"train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}")
        print(sample)

    # Scratch runs
    for e in scratch_epochs:
        run_dir = os.path.join(args.output_dir, f"scratch_e{e}")
        print(f"\n=== Scratch Transformer: {e} epochs ===")
        tr_loss, va_loss, sample = scratch_run(args, tokenizer, train_ds, val_ds, device, epochs=e, run_dir=run_dir)
        scratch_rows.append((e, tr_loss, va_loss, sample))
        print(f"train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}")
        print(sample)

    print_latex_report(
        finetune_rows=finetune_rows,
        scratch_rows=scratch_rows,
        prompt=args.prompt,
    )


if __name__ == "__main__":
    main()
