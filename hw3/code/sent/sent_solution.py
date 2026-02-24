#!/usr/bin/env python3
"""
sent_solution.py

Fine-tune GPT-2 for binary sentiment classification using Hugging Face Trainer.

Data format:
- data/train.txt, data/val.txt, (optional) data/test.txt
  Each line: <label><whitespace><text>
  label is the first character on the line and must be '0' or '1'.

Default behavior:
- Run sweep on (train, val)
- Select best config by validation loss (tie-break by validation accuracy)
- Retrain best config on (train + val)
- Save ONE checkpoint to outputs/sent_model.pt
- Print LaTeX sweep table by default (for copy-paste)

Hidden test behavior:
- test is hidden, so by default we do NOT load/evaluate test.
- If RunConfig.mode == "test", the script will:
  (i) load the saved checkpoint, and
  (ii) evaluate it on train, val, and (optional) test if RunConfig.test_path is not None.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset

from transformers import (
    GPT2TokenizerFast,
    GPT2ForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
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
# Data loading
# -------------------------

def read_data(path: str) -> Tuple[List[str], List[int]]:
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            c = line[0]
            if c not in {"0", "1"}:
                raise ValueError(f"Invalid label '{c}' in line: {line[:80]}")
            label = int(c)
            text = line[1:].strip()
            texts.append(text)
            labels.append(label)
    return texts, labels


# -------------------------
# Tokenization & Dataset
# -------------------------

def build_tokenizer(model_name: str = "gpt2") -> GPT2TokenizerFast:
    tok = GPT2TokenizerFast.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    return tok


def tokenize_text(text: str, tokenizer: GPT2TokenizerFast, max_len: int) -> Dict[str, List[int]]:
    """
    Implement:
        Tokenize a single text example for GPT-2.

        Requirements:
          - Return dict with keys: "input_ids" and "attention_mask".
          - Truncate to max_len.
          - Do NOT pad here (padding should be done dynamically by the collator).

        Args:
            text: input string
            tokenizer: GPT-2 tokenizer
            max_len: maximum sequence length

        Returns:
            dict with keys "input_ids" and "attention_mask" (both are lists of ints)
    """
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_len,
        padding=False,
        return_attention_mask=True,
    )
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}


class SentimentDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: GPT2TokenizerFast, max_len: int):
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length")
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Implement:
            Return a dictionary with keys:
              - "input_ids": List[int]
              - "attention_mask": List[int]
              - "labels": int

            This should call tokenize_text(...) and then add the "labels" field.

        Args:
            idx: index into the dataset

        Returns:
            A dict suitable for Hugging Face Trainer.
        """
        item = tokenize_text(self.texts[idx], self.tokenizer, self.max_len)
        item["labels"] = int(self.labels[idx])
        return item


def build_data_collator(tokenizer: GPT2TokenizerFast) -> DataCollatorWithPadding:
    """
    Implement:
        Return a dynamic padding collator for Trainer.

        Requirements:
          - Pad to the longest sequence in the batch.
          - Return PyTorch tensors for input_ids, attention_mask, labels.

        Args:
            tokenizer: GPT-2 tokenizer with pad_token set

        Returns:
            A DataCollatorWithPadding (or equivalent) for Trainer.
    """
    return DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt")


# -------------------------
# Model & Metrics
# -------------------------

def build_model(
    model_name: str = "gpt2",
    num_labels: int = 2,
    pad_token_id: Optional[int] = None,
) -> GPT2ForSequenceClassification:
    """
    Implement:
        Build GPT2ForSequenceClassification with num_labels=2.

        Requirements:
          - Use from_pretrained(model_name, num_labels=2)
          - Set model.config.pad_token_id = pad_token_id (tokenizer.pad_token_id)

        Args:
            model_name: e.g. "gpt2"
            num_labels: number of classes (2)
            pad_token_id: tokenizer.pad_token_id (EOS)

        Returns:
            A GPT2ForSequenceClassification model.
    """
    model = GPT2ForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    if pad_token_id is not None:
        model.config.pad_token_id = pad_token_id
    return model


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = float((preds == labels).mean())
    return {"accuracy": acc}


# -------------------------
# Sweep config
# -------------------------

@dataclass(frozen=True)
class SweepConfig:
    name: str
    learning_rate: float
    batch_size: int
    num_train_epochs: int
    weight_decay: float = 0.0


def default_sweep() -> List[SweepConfig]:
    return [
        SweepConfig("A", 2e-4, 8, 1, 0.01),
        SweepConfig("B", 1e-4, 8, 2, 0.001),
        SweepConfig("C", 1e-4, 8, 3, 0.01),
    ]


def pick_best(results: List[Dict[str, float]]) -> Dict[str, float]:
    def key(m: Dict[str, float]):
        return (m.get("eval_loss", float("inf")), -m.get("eval_accuracy", float("-inf")))
    return sorted(results, key=key)[0]


# -------------------------
# Trainer
# -------------------------

def build_trainer(
    cfg: SweepConfig,
    model: GPT2ForSequenceClassification,
    tokenizer: GPT2TokenizerFast,
    train_ds: Dataset,
    val_ds: Dataset,
    seed: int,
) -> Trainer:
    """
    Implement:
        Build TrainingArguments (from cfg) and return a Hugging Face Trainer.

        Requirements:
          - Use cfg.learning_rate, cfg.batch_size, cfg.num_train_epochs, cfg.weight_decay
          - evaluation_strategy="epoch"
          - save_strategy="no"
          - report_to=[]
          - Use build_data_collator(tokenizer)
          - Use compute_metrics for accuracy

        Args:
            cfg: sweep configuration
            model: GPT-2 classifier
            tokenizer: GPT-2 tokenizer
            train_ds: training dataset
            val_ds: validation dataset
            seed: random seed

        Returns:
            A Trainer instance.
    """
    out_dir = os.path.join("outputs", f"sent_{cfg.name}")
    os.makedirs(out_dir, exist_ok=True)

    args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.num_train_epochs,
        weight_decay=cfg.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_strategy="steps",
        report_to=[],
        seed=seed,
        data_seed=seed,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=build_data_collator(tokenizer),
        compute_metrics=compute_metrics,
    )
    return trainer


# -------------------------
# Checkpoint save/load
# -------------------------

def save_checkpoint_pt(
    path: str,
    model: GPT2ForSequenceClassification,
    tokenizer: GPT2TokenizerFast,
    best_metrics: Dict[str, float],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_name": "gpt2",
            "num_labels": 2,
            "pad_token_id": int(tokenizer.pad_token_id),
            "state_dict": model.state_dict(),
            "best_metrics": dict(best_metrics),
        },
        path,
    )


def load_checkpoint_pt(path: str, map_location: Optional[str] = None) -> Tuple[GPT2ForSequenceClassification, Dict[str, Any]]:
    if map_location is None:
        map_location = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(path, map_location=map_location)

    model_name = ckpt.get("model_name", "gpt2")
    num_labels = int(ckpt.get("num_labels", 2))
    pad_token_id = int(ckpt.get("pad_token_id", 50256))

    model = GPT2ForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.config.pad_token_id = pad_token_id
    model.load_state_dict(ckpt["state_dict"])

    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    return model, meta


# -------------------------
# Inference
# -------------------------

@torch.no_grad()
def predict_sentiment(
    model: GPT2ForSequenceClassification,
    tokenizer: GPT2TokenizerFast,
    text: str,
    max_len: int = 256,
) -> int:
    """
    Implement:
        Predict the binary sentiment label (0 or 1) for a single input text.

        Steps:
          1) Use the device where the model parameters are located.
          2) Set model to eval mode.
          3) Tokenize with truncation (max_len) and padding, returning tensors and attention_mask.
          4) Move inputs to model.device, run model, and return argmax(logits).

        Args:
            model: fine-tuned GPT-2 classifier
            tokenizer: GPT-2 tokenizer with pad_token set
            text: input string
            max_len: truncation length

        Returns:
            int: 0 or 1
    """
    device = next(model.parameters()).device
    model.eval()

    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_len,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    outputs = model(**enc)
    pred = torch.argmax(outputs.logits, dim=-1).item()
    return int(pred)


# -------------------------
# LaTeX table
# -------------------------

def format_sweep_table_latex(all_metrics: List[Dict[str, float]], sweep: List[SweepConfig], best_config: str) -> str:
    lines: List[str] = []
    lines.append(r"\begin{tabular}{cccc}")
    lines.append(r"\hline")
    lines.append(r"Config & Train epochs & Val loss & Val acc \\")
    lines.append(r"\hline")

    metric_map = {m["config"]: m for m in all_metrics}
    for cfg in sweep:
        m = metric_map.get(cfg.name, {})
        loss = float(m.get("eval_loss", float("nan")))
        acc = float(m.get("eval_accuracy", float("nan")))
        lines.append(f"{cfg.name} & {cfg.num_train_epochs} & {loss:.4f} & {acc:.4f} \\\\")
    lines.append(r"\hline")
    lines.append(rf"\multicolumn{{4}}{{c}}{{Best config:\ {best_config}}} \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


# -------------------------
# Training / Sweep
# -------------------------

def run_sweep(
    train_path: str,
    val_path: str,
    max_len: int,
    seed: int,
    out_ckpt: str,
) -> Dict[str, Any]:
    set_seed(seed)
    tokenizer = build_tokenizer("gpt2")

    train_texts, train_labels = read_data(train_path)
    val_texts, val_labels = read_data(val_path)

    train_ds = SentimentDataset(train_texts, train_labels, tokenizer, max_len)
    val_ds = SentimentDataset(val_texts, val_labels, tokenizer, max_len)

    sweep = default_sweep()
    all_metrics: List[Dict[str, float]] = []

    for cfg in sweep:
        set_seed(seed)
        model = build_model("gpt2", 2, tokenizer.pad_token_id)
        trainer = build_trainer(cfg, model, tokenizer, train_ds, val_ds, seed)
        trainer.train()
        m = trainer.evaluate()
        all_metrics.append(
            {
                "config": cfg.name,
                "eval_loss": float(m.get("eval_loss", np.nan)),
                "eval_accuracy": float(m.get("eval_accuracy", np.nan)),
            }
        )

    best = pick_best(all_metrics)
    best_cfg = next(c for c in sweep if c.name == best["config"])

    # Retrain on train+val
    combined_texts = train_texts + val_texts
    combined_labels = train_labels + val_labels
    trainval_ds = SentimentDataset(combined_texts, combined_labels, tokenizer, max_len)

    set_seed(seed)
    final_model = build_model("gpt2", 2, tokenizer.pad_token_id)
    # eval_dataset is unused since evaluation_strategy="epoch" but fine if present
    final_trainer = build_trainer(best_cfg, final_model, tokenizer, trainval_ds, val_ds, seed)
    final_trainer.train()

    save_checkpoint_pt(out_ckpt, final_model, tokenizer, best)

    return {
        "all_metrics": all_metrics,
        "best_metrics": best,
        "sweep": sweep,
        "checkpoint_path": out_ckpt,
    }


def evaluate_checkpoint(
    ckpt_path: str,
    train_path: str,
    val_path: str,
    test_path: Optional[str],
    max_len: int,
    seed: int,
) -> Dict[str, float]:
    """
    Load the saved checkpoint and evaluate on train, val, and (optional) test.
    If test_path is None, evaluate only on train/val.
    """
    set_seed(seed)
    tokenizer = build_tokenizer("gpt2")

    model, _ = load_checkpoint_pt(ckpt_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_texts, train_labels = read_data(train_path)
    val_texts, val_labels = read_data(val_path)

    train_ds = SentimentDataset(train_texts, train_labels, tokenizer, max_len)
    val_ds = SentimentDataset(val_texts, val_labels, tokenizer, max_len)

    eval_args = TrainingArguments(
        output_dir=os.path.join("outputs", "sent_eval"),
        per_device_eval_batch_size=16,
        report_to=[],
        seed=seed,
        data_seed=seed,
        fp16=torch.cuda.is_available(),
    )

    eval_trainer = Trainer(
        model=model,
        args=eval_args,
        tokenizer=tokenizer,
        data_collator=build_data_collator(tokenizer),
        compute_metrics=compute_metrics,
    )

    train_m = eval_trainer.evaluate(eval_dataset=train_ds)
    val_m = eval_trainer.evaluate(eval_dataset=val_ds)

    out: Dict[str, float] = {
        "train_loss": float(train_m.get("eval_loss", np.nan)),
        "train_acc": float(train_m.get("eval_accuracy", np.nan)),
        "val_loss": float(val_m.get("eval_loss", np.nan)),
        "val_acc": float(val_m.get("eval_accuracy", np.nan)),
    }

    if test_path is not None:
        test_texts, test_labels = read_data(test_path)
        test_ds = SentimentDataset(test_texts, test_labels, tokenizer, max_len)
        test_m = eval_trainer.evaluate(eval_dataset=test_ds)
        out["test_loss"] = float(test_m.get("eval_loss", np.nan))
        out["test_acc"] = float(test_m.get("eval_accuracy", np.nan))

    return out


# -------------------------
# Main (no argparse except optional mode; you can also remove mode)
# -------------------------

@dataclass
class RunConfig:
    # defaults (edit here, not via CLI)
    mode: str = "train"  # "train" or "test"
    train_path: str = "data/train.txt"
    val_path: str = "data/val.txt"
    test_path: str = "data/test.txt"
    max_len: int = 256
    seed: int = 0

    save_path: str = "outputs/sent_model.pt"  # where training saves
    load_path: Optional[str] = None  # set automatically in eval mode if None

    print_json: bool = False  # default prints LaTeX table


def parse_args() -> RunConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, default="train")
    args = p.parse_args()

    cfg = RunConfig(mode=args.mode)
    if cfg.mode != "train" and cfg.load_path is None:
        # this is only for grading as testset is hidden
        cfg.load_path = cfg.save_path
    return cfg


def main() -> None:
    cfg = parse_args()

    if cfg.mode == "train":
        results = run_sweep(
            train_path=cfg.train_path,
            val_path=cfg.val_path,
            max_len=cfg.max_len,
            seed=cfg.seed,
            out_ckpt=cfg.save_path,
        )

        if cfg.print_json:
            print(json.dumps(results, indent=2))
        else:
            print(
                format_sweep_table_latex(
                    results["all_metrics"],
                    results["sweep"],
                    results["best_metrics"]["config"],
                )
            )

    else:
        eval_metrics = evaluate_checkpoint(
            ckpt_path=cfg.load_path if cfg.load_path is not None else cfg.save_path,
            train_path=cfg.train_path,
            val_path=cfg.val_path,
            test_path=cfg.test_path,   # can be None
            max_len=cfg.max_len,
            seed=cfg.seed,
        )

        if cfg.print_json:
            print(json.dumps({"loaded_checkpoint_eval": eval_metrics}, indent=2))
        else:
            print("\n# Loaded-checkpoint evaluation (train/val/test)")
            print(f"train: loss={eval_metrics['train_loss']:.4f} acc={eval_metrics['train_acc']:.4f}")
            print(f"val:   loss={eval_metrics['val_loss']:.4f} acc={eval_metrics['val_acc']:.4f}")
            if "test_loss" in eval_metrics:
                print(f"test:  loss={eval_metrics['test_loss']:.4f} acc={eval_metrics['test_acc']:.4f}")


if __name__ == "__main__":
    main()
