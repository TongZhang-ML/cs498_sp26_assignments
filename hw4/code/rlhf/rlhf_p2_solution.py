#!/usr/bin/env python3
"""
Part 2 (RLHF): DPO training with fixed reference model.

This file is the Python version of the original notebook implementation.
Students should implement functions whose docstring starts with "Implement".
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
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
class CustomDataCollator:
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
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
# DPO reward model
# -------------------------

class DPOReward(nn.Module):
    def __init__(self, model_rlhf, model_ref, device: str, beta: float = 0.2):
        super().__init__()
        self.beta = beta
        self.model = model_rlhf.to(device)
        self.model_ref = model_ref.to(device)
        self.model_ref.eval()
        self.device = device

        for p in self.model.parameters():
            p.requires_grad = True
        for p in self.model_ref.parameters():
            p.requires_grad = False

    def _nll_per_example(self, outputs, labels, attention_mask, score_mask=None):
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous().float()
        if score_mask is not None:
            shift_score = score_mask[..., 1:].contiguous().float()
            shift_mask = shift_mask * shift_score

        nll = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        )
        nll = nll.view(shift_labels.size())
        token_sum = (nll * shift_mask).sum(dim=1)
        token_cnt = shift_mask.sum(dim=1).clamp_min(1)
        return token_sum, token_cnt

    def _reward(self, input_ids, attention_mask, normalize_by_length: bool, score_mask=None):
        labels = input_ids.clone()
        with torch.no_grad():
            out_ref = self.model_ref(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss_ref, cnt_ref = self._nll_per_example(out_ref, labels, attention_mask, score_mask=score_mask)

        out_rlhf = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss_rlhf, cnt_rlhf = self._nll_per_example(
            out_rlhf, labels, attention_mask, score_mask=score_mask
        )

        if normalize_by_length:
            loss_ref = loss_ref / cnt_ref
            loss_rlhf = loss_rlhf / cnt_rlhf

        return self.beta * (loss_ref - loss_rlhf)

    def forward(self, input_ids, attention_mask):
        # Keep training behavior as originally implemented (sum over tokens).
        return self._reward(input_ids, attention_mask, normalize_by_length=False)

    def score_normalized(self, input_ids, attention_mask, score_mask=None):
        # Length-normalized reward for fair generation comparison.
        return self._reward(
            input_ids, attention_mask, normalize_by_length=True, score_mask=score_mask
        )


# -------------------------
# Evaluation
# -------------------------

def evaluate_model(model, tokenized_dataset, data_collator, dataset_name: str, device: str):
    dataloader = DataLoader(tokenized_dataset, batch_size=16, collate_fn=data_collator)

    chosen_scores, rejected_scores = [], []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Eval {dataset_name}"):
            chosen_scores.append(
                model(
                    input_ids=batch["chosen_input_ids"].to(device),
                    attention_mask=batch["chosen_attention_mask"].to(device),
                ).cpu()
            )
            rejected_scores.append(
                model(
                    input_ids=batch["rejected_input_ids"].to(device),
                    attention_mask=batch["rejected_attention_mask"].to(device),
                ).cpu()
            )

    chosen_scores = torch.cat(chosen_scores, dim=0)
    rejected_scores = torch.cat(rejected_scores, dim=0)
    target = torch.ones_like(chosen_scores)
    loss = F.binary_cross_entropy_with_logits(chosen_scores - rejected_scores, target)
    acc = (chosen_scores > rejected_scores).float().mean().item()
    tie = (chosen_scores == rejected_scores).float().mean().item()
    acc = acc + 0.5 * tie

    print(f"{dataset_name}: loss={loss:.6f}, acc={acc:.6f}")


# -------------------------
# Trainer
# -------------------------

class DPOTrainer(Trainer):
    """Trainer for DPO-style pairwise optimization using implicit reward."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Implement according to the assignment description.

        Goal
        ----
        Compute DPO pairwise loss from chosen/rejected rewards.

        Required computation
        --------------------
        1) Get implicit reward scores for chosen and rejected from DPOReward model.
        2) Compute preference difference d = chosen - rejected.
        3) Use BCE-with-logits target=1:
           loss = BCEWithLogits(d, ones).

        Returns
        -------
        loss, or (loss, (chosen_scores, rejected_scores)) when return_outputs=True.
        """
        chosen = model(
            input_ids=inputs["chosen_input_ids"],
            attention_mask=inputs["chosen_attention_mask"],
        )
        rejected = model(
            input_ids=inputs["rejected_input_ids"],
            attention_mask=inputs["rejected_attention_mask"],
        )
        diff = chosen - rejected
        target = torch.ones_like(diff)
        loss = F.binary_cross_entropy_with_logits(diff, target)
        return (loss, (chosen, rejected)) if return_outputs else loss

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
        - Return tuple: (loss, predictions, labels) where labels=None.
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

    train_size = 2500
    test_size = -1
    max_eval_prompts = 256
    max_input_length = 256
    max_new_tokens = 128
    num_train_epochs = 1

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
        train_count = min(train_size, len(dataset["train_prefs"]))
        trainset = dataset["train_prefs"].select(range(train_count))
        testset = dataset["test_prefs"]
    if test_size > 0:
        test_count = min(test_size, len(testset))
        testset = testset.select(range(test_count))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen3-0.6B"

    model_ref = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=data_dir)
    model_rlhf = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=data_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, cache_dir=data_dir)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        model_ref.resize_token_embeddings(len(tokenizer))
        model_rlhf.resize_token_embeddings(len(tokenizer))

    tokenized_train = trainset.map(tokenize_example, batched=False, fn_kwargs={"tokenizer": tokenizer})
    tokenized_test = testset.map(tokenize_example, batched=False, fn_kwargs={"tokenizer": tokenizer})
    collator = CustomDataCollator(tokenizer=tokenizer)

    reward_model = DPOReward(
        model_rlhf=model_rlhf,
        model_ref=model_ref,
        beta=0.1,
        device=device,
    )

    evaluate_model(reward_model, tokenized_train, collator, "train_before", device)
    evaluate_model(reward_model, tokenized_test, collator, "test_before", device)

    args = TrainingArguments(
        output_dir=os.path.join(output_root, "part2_dpo"),
        report_to="none",
        save_strategy="no",
        learning_rate=5e-6,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=num_train_epochs,
        remove_unused_columns=False,
        bf16=False,
        fp16=False,
        seed=seed,
        data_seed=seed,
    )

    trainer = DPOTrainer(
        model=reward_model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        data_collator=collator,
    )
    trainer.train()

    evaluate_model(reward_model, tokenized_train, collator, "train_after", device)
    evaluate_model(reward_model, tokenized_test, collator, "test_after", device)
