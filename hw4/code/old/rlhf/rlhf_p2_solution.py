#!/usr/bin/env python3
"""
Part 2 (RLHF): DPO training with fixed reference model.

This file is the Python version of the original notebook implementation.
Students should implement functions whose docstring starts with "Implement".
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List

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

    def _nll_per_example(self, outputs, labels, attention_mask):
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()

        nll = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        )
        nll = nll.view(shift_labels.size())
        token_sum = (nll * shift_mask).sum(dim=1)
        token_cnt = shift_mask.sum(dim=1).clamp_min(1)
        return token_sum, token_cnt

    def _reward(self, input_ids, attention_mask, normalize_by_length: bool):
        labels = input_ids.clone()
        with torch.no_grad():
            out_ref = self.model_ref(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss_ref, cnt_ref = self._nll_per_example(out_ref, labels, attention_mask)

        out_rlhf = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss_rlhf, cnt_rlhf = self._nll_per_example(out_rlhf, labels, attention_mask)

        if normalize_by_length:
            loss_ref = loss_ref / cnt_ref
            loss_rlhf = loss_rlhf / cnt_rlhf

        return self.beta * (loss_ref - loss_rlhf)

    def forward(self, input_ids, attention_mask):
        # Keep training behavior as originally implemented (sum over tokens).
        return self._reward(input_ids, attention_mask, normalize_by_length=False)

    def score_normalized(self, input_ids, attention_mask):
        # Length-normalized reward for fair generation comparison.
        return self._reward(input_ids, attention_mask, normalize_by_length=True)


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


def _build_prompt_messages(chosen_messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    last_assistant_idx = -1
    for i, msg in enumerate(chosen_messages):
        if msg.get("role") == "assistant":
            last_assistant_idx = i
    if last_assistant_idx >= 0:
        return chosen_messages[:last_assistant_idx]
    return chosen_messages


def _build_prompt_text(tokenizer, messages: List[Dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt = concatenate_conversation(messages)
    return prompt + "\n\nAssistant:"


def evaluate_generated_avg_dpo_reward(
    reward_model,
    generation_model,
    tokenizer,
    raw_testset,
    device: str,
    batch_size: int = 8,
    max_prompts: int = 256,
    max_input_length: int = 256,
    max_new_tokens: int = 64,
) -> float:
    prompts = []
    for i in range(min(max_prompts, len(raw_testset))):
        messages = _build_prompt_messages(raw_testset[i]["chosen"])
        prompts.append(_build_prompt_text(tokenizer, messages))

    reward_model.to(device)
    generation_model.to(device)
    reward_model.eval()
    generation_model.eval()

    reward_sum = 0.0
    total = 0
    with torch.no_grad():
        for start in tqdm(range(0, len(prompts), batch_size), desc="Eval generated DPO reward"):
            batch_prompts = prompts[start : start + batch_size]
            enc = tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=max_input_length,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            sequences = generation_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            seq_attention = (sequences != tokenizer.pad_token_id).long()
            rewards = reward_model.score_normalized(input_ids=sequences, attention_mask=seq_attention)

            reward_sum += rewards.sum().item()
            total += rewards.numel()

    return reward_sum / max(total, 1)


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


if __name__ == "__main__":
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
    model_name = "Qwen/Qwen2-0.5B-Instruct"

    model_ref = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=data_dir)
    model_rlhf = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=data_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, cache_dir=data_dir)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    tokenized_train = trainset.map(tokenize_example, batched=False, fn_kwargs={"tokenizer": tokenizer})
    tokenized_test = testset.map(tokenize_example, batched=False, fn_kwargs={"tokenizer": tokenizer})
    collator = CustomDataCollator(tokenizer=tokenizer)

    reward_model = DPOReward(model_rlhf=model_rlhf, model_ref=model_ref, beta=0.2, device=device)

    evaluate_model(reward_model, tokenized_train, collator, "train_before", device)
    evaluate_model(reward_model, tokenized_test, collator, "test_before", device)

    args = TrainingArguments(
        output_dir=os.path.join(output_root, "part2_dpo"),
        report_to="none",
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=2e-6,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        remove_unused_columns=False,
        bf16=False,
        fp16=False,
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

    base_avg_reward = evaluate_generated_avg_dpo_reward(
        reward_model=reward_model,
        generation_model=model_ref,
        tokenizer=tokenizer,
        raw_testset=testset,
        device=device,
    )
    dpo_avg_reward = evaluate_generated_avg_dpo_reward(
        reward_model=reward_model,
        generation_model=reward_model.model,
        tokenizer=tokenizer,
        raw_testset=testset,
        device=device,
    )
    print(f"base_model_test_prompt_avg_dpo_reward={base_avg_reward:.6f}")
    print(f"dpo_model_test_prompt_avg_dpo_reward={dpo_avg_reward:.6f}")
    print(f"avg_reward_improvement={dpo_avg_reward - base_avg_reward:.6f}")
