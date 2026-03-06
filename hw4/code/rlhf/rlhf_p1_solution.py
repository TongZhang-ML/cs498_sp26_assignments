#!/usr/bin/env python3
"""
Part 1 (RLHF): reward model training on preference data.

This file is the Python version of the original notebook implementation.
Students should implement functions whose docstring starts with "Implement".
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
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
# Data preprocessing
# -------------------------


def concatenate_conversation(messages: List[Dict[str, str]]) -> str:
    chunks = []
    for msg in messages:
        chunks.append(f"{msg['role'].capitalize()}: {msg['content'].strip()}")
    return "\n\n".join(chunks)


def tokenize_example(example: Dict[str, Any], tokenizer, max_length: int = 256) -> Dict[str, Any]:
    chosen_text = concatenate_conversation(example["chosen"])
    rejected_text = concatenate_conversation(example["rejected"])

    chosen = tokenizer(chosen_text, padding="max_length", truncation=True, max_length=max_length)
    rejected = tokenizer(rejected_text, padding="max_length", truncation=True, max_length=max_length)
    return {
        "chosen_input_ids": chosen["input_ids"],
        "chosen_attention_mask": chosen["attention_mask"],
        "rejected_input_ids": rejected["input_ids"],
        "rejected_attention_mask": rejected["attention_mask"],
    }


@dataclass
class CustomDataCollatorWithPadding:
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        chosen_features = [
            {"input_ids": f["chosen_input_ids"], "attention_mask": f["chosen_attention_mask"]}
            for f in features
        ]
        rejected_features = [
            {"input_ids": f["rejected_input_ids"], "attention_mask": f["rejected_attention_mask"]}
            for f in features
        ]
        chosen_batch = collator(chosen_features)
        rejected_batch = collator(rejected_features)
        return {
            "chosen_input_ids": chosen_batch["input_ids"],
            "chosen_attention_mask": chosen_batch["attention_mask"],
            "rejected_input_ids": rejected_batch["input_ids"],
            "rejected_attention_mask": rejected_batch["attention_mask"],
        }


# -------------------------
# Evaluation
# -------------------------

def evaluate_model(model, tokenized_dataset, data_collator, dataset_name: str, device: str) -> Tuple[float, float]:
    dataloader = DataLoader(tokenized_dataset, batch_size=16, collate_fn=data_collator)

    chosen_logits, rejected_logits = [], []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Eval {dataset_name}"):
            chosen_outputs = model(
                input_ids=batch["chosen_input_ids"].to(device),
                attention_mask=batch["chosen_attention_mask"].to(device),
            )
            rejected_outputs = model(
                input_ids=batch["rejected_input_ids"].to(device),
                attention_mask=batch["rejected_attention_mask"].to(device),
            )
            chosen_logits.append(chosen_outputs.logits.cpu())
            rejected_logits.append(rejected_outputs.logits.cpu())

    chosen_logits = torch.cat(chosen_logits, dim=0)
    rejected_logits = torch.cat(rejected_logits, dim=0)

    # Compute metric loss in fp32 for numerical stability when model uses mixed precision.
    chosen_logits = torch.nan_to_num(chosen_logits.float(), nan=0.0, posinf=30.0, neginf=-30.0)
    rejected_logits = torch.nan_to_num(rejected_logits.float(), nan=0.0, posinf=30.0, neginf=-30.0)
    diff = torch.clamp(chosen_logits - rejected_logits, min=-30.0, max=30.0)
    target = torch.ones_like(diff)
    loss = F.binary_cross_entropy_with_logits(diff, target)
    acc = (chosen_logits > rejected_logits).float().mean().item()
    tie = (chosen_logits == rejected_logits).float().mean().item()
    acc = acc + 0.5 * tie

    print(f"{dataset_name}: loss={loss:.6f}, acc={acc:.6f}")
    return float(loss.item()), float(acc)


class BTTrainer(Trainer):
    """Bradley-Terry trainer for pairwise preference reward modeling."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Implement according to the assignment description.

        Goal
        ----
        Compute Bradley-Terry loss from chosen/rejected pairs.

        Required computation
        --------------------
        1) Forward chosen and rejected responses through the reward model.
        2) Let chosen_logits, rejected_logits be scalar rewards.
        3) Compute logit difference: d = chosen_logits - rejected_logits.
        4) Use BCE-with-logits target=1 so that chosen is preferred:
           loss = BCEWithLogits(d, ones).

        Returns
        -------
        loss, or (loss, (chosen_logits, rejected_logits)) when return_outputs=True.
        """
        chosen_out = model(
            input_ids=inputs["chosen_input_ids"],
            attention_mask=inputs["chosen_attention_mask"],
        )
        rejected_out = model(
            input_ids=inputs["rejected_input_ids"],
            attention_mask=inputs["rejected_attention_mask"],
        )
        chosen_logits = chosen_out.logits.squeeze(-1)
        rejected_logits = rejected_out.logits.squeeze(-1)

        # Keep loss math in fp32 to avoid fp16/bf16 overflow producing NaNs.
        chosen_logits = torch.nan_to_num(chosen_logits.float(), nan=0.0, posinf=30.0, neginf=-30.0)
        rejected_logits = torch.nan_to_num(rejected_logits.float(), nan=0.0, posinf=30.0, neginf=-30.0)
        diff = torch.clamp(chosen_logits - rejected_logits, min=-30.0, max=30.0)
        target = torch.ones_like(diff)
        loss = F.binary_cross_entropy_with_logits(diff, target)
        return (loss, (chosen_logits, rejected_logits)) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        """
        Implement according to the assignment description.

        Goal
        ----
        Run an evaluation step compatible with Hugging Face Trainer.

        Requirements
        ------------
        - Compute loss and predictions with no gradients.
        - If prediction_loss_only=True, return only loss.
        - Return tuple: (loss, predictions, labels) where labels=None for this task.
        """
        model.eval()
        with torch.no_grad():
            loss, preds = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            preds = None
        return loss, preds, None


# -------------------------
# Main
# -------------------------

if __name__ == "__main__":
    seed = 42
    set_seed(seed)
    data_dir = "./data"
    output_root = "./outputs"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_root, exist_ok=True)
    train_txt = os.path.join(data_dir, "train_prefs.txt")
    test_txt = os.path.join(data_dir, "test_prefs.txt")

    if os.path.isfile(train_txt) and os.path.isfile(test_txt):
        dataset = load_dataset(
            "json",
            data_files={"train_prefs": train_txt, "test_prefs": test_txt},
            cache_dir=data_dir,
        )
        trainset = dataset["train_prefs"]
        testset = dataset["test_prefs"]
    else:
        dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", cache_dir=data_dir)
        trainset = dataset["train_prefs"].select(range(2500))
        testset = dataset["test_prefs"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=data_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1, cache_dir=data_dir
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    tokenized_train = trainset.map(tokenize_example, batched=False, fn_kwargs={"tokenizer": tokenizer})
    tokenized_test = testset.map(tokenize_example, batched=False, fn_kwargs={"tokenizer": tokenizer})
    collator = CustomDataCollatorWithPadding(tokenizer=tokenizer)

    evaluate_model(model, tokenized_train, collator, "train_before", device)
    evaluate_model(model, tokenized_test, collator, "test_before", device)

    args = TrainingArguments(
        output_dir=os.path.join(output_root, "part1_reward"),
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.0,
        warmup_steps=0,
        logging_steps=50,
        remove_unused_columns=False,
        bf16=False,
        fp16=False,
        seed=seed,
        data_seed=seed,
    )

    trainer = BTTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        data_collator=collator,
    )
    trainer.train()

    evaluate_model(model, tokenized_train, collator, "train_after", device)
    evaluate_model(model, tokenized_test, collator, "test_after", device)
