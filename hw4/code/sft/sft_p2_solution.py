#!/usr/bin/env python3
"""
Part 2 (SFT): instruction tuning + LLM-as-a-judge evaluation.

This version mirrors the demo notebook style:
- one render_chat() helper for conversation formatting and token boundary
- prompt masking based on assistant-response boundary
- Hugging Face Trainer for SFT
"""

from __future__ import annotations

import os
import random
from typing import Dict

import numpy as np
import torch
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers.trainer_utils import set_seed as hf_set_seed


# -------------------------
# Reproducibility
# -------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------
# GPT-as-judge evaluation
# -------------------------

def gpt_eval_responses(
    client: OpenAI,
    prompt: str,
    response1: str,
    response2: str,
    model: str = "gpt-5-nano",
) -> float:
    """Return 1 if response1 wins, 0 if response2 wins, 0.5 for tie."""
    query = f"""
I have two responses to the same prompt. Decide which one is better.
Evaluate relevance, accuracy, and fluency.

Prompt: {prompt}
Response 1: {response1}
Response 2: {response2}

Return exactly one of:
- Response 1 is better
- Response 2 is better
- They are equally good
"""

    out = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": query}],
    )
    text = out.choices[0].message.content
    if "Response 1 is better" in text:
        return 1.0
    if "Response 2 is better" in text:
        return 0.0
    return 0.5


def evaluate_responses(client: OpenAI, dataset: Dataset, field_1: str, field_2: str) -> float:
    """
    Implement according to the assignment description.

    Goal
    ----
    Evaluate which response field wins more often on a dataset using GPT judge.

    Requirements
    ------------
    - For each example, evaluate both orderings:
      score_ab = judge(prompt, field_1, field_2)
      score_ba = judge(prompt, field_2, field_1)
    - Debias with average:
      score = 0.5 * (score_ab + (1 - score_ba))
    - Return dataset-level average in [0,1], where larger means field_1 wins more.
    """
    total = len(dataset)
    wins = 0.0
    for ex in tqdm(dataset, desc=f"Evaluating {field_1} vs {field_2}"):
        prompt = ex["prompt"]
        r1 = ex[field_1]
        r2 = ex[field_2]
        score_ab = gpt_eval_responses(client, prompt, r1, r2)
        score_ba = gpt_eval_responses(client, prompt, r2, r1)
        wins += 0.5 * (score_ab + (1.0 - score_ba))
    return wins / total


# -------------------------
# Chat formatting + tokenization
# -------------------------

def render_chat(prompt: str, response: str | None = None, tokenizer=None, max_length: int = 256):
    """
    Build chat-formatted text and optionally tokenize once for reuse.

    Returns a dict with:
      - prompt_text / full_text
      - boundary (assistant response start in token space)
      - prompt_ids, prompt_attention_mask, full_ids, full_attention_mask
        when tokenizer is provided
    """
    prompt_text = f"User: {prompt}\nAssistant:"
    full_text = prompt_text if response is None else f"User: {prompt}\nAssistant: {response}"

    out = {
        "prompt_text": prompt_text,
        "full_text": full_text,
        "boundary": None,
    }

    if tokenizer is not None:
        prompt_tok = tokenizer(
            prompt_text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        )
        full_tok = tokenizer(
            full_text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        )
        boundary = min(len(prompt_tok["input_ids"]), len(full_tok["input_ids"]))
        out.update(
            {
                "boundary": boundary,
                "prompt_ids": prompt_tok["input_ids"],
                "prompt_attention_mask": prompt_tok["attention_mask"],
                "full_ids": full_tok["input_ids"],
                "full_attention_mask": full_tok["attention_mask"],
            }
        )

    return out


def _clean_assistant_text(text: str) -> str:
    """Keep one assistant reply for cleaner before/after comparison."""
    out = text.strip()
    while out.startswith("Assistant:"):
        out = out[len("Assistant:") :].strip()
    if "\nAssistant:" in out:
        out = out.split("\nAssistant:")[0].strip()
    if "\nUser:" in out:
        out = out.split("\nUser:")[0].strip()
    return out


def generate_responses(batch, model, tokenizer, device: torch.device, field_name: str):
    prompt_texts = [render_chat(p)["prompt_text"] for p in batch["prompt"]]
    inputs = tokenizer(
        prompt_texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    continuation = outputs[:, input_ids.shape[1] :]
    batch[field_name] = [
        _clean_assistant_text(tokenizer.decode(ids, skip_special_tokens=True))
        for ids in continuation
    ]
    return batch


def tokenize_with_prompt_masking(examples, tokenizer):
    """
    Implement according to the assignment description.

    Goal
    ----
    Create SFT training features that compute loss only on response tokens.

    Requirements
    ------------
    - Use render_chat(prompt, response, tokenizer=..., max_length=...) to get:
      (i) full tokenized ids/masks and (ii) assistant-start boundary.
    - Build labels from input_ids.
    - Set labels to -100 for every token before the assistant response start,
      including user text and role markers.
    - Keep labels unchanged only for assistant response tokens.
    - Pad/truncate to fixed max_length and return dict with
      input_ids, attention_mask, labels.
    """
    max_length = 256
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    all_input_ids = []
    all_attention_mask = []
    all_labels = []

    # Right padding keeps boundary index unchanged after padding.
    for prompt, response in zip(examples["prompt"], examples["response"]):
        chat = render_chat(prompt, response, tokenizer=tokenizer, max_length=max_length)

        full_ids = list(chat["full_ids"])
        full_attn = list(chat["full_attention_mask"])
        boundary = int(chat["boundary"])

        labels = full_ids.copy()
        for i in range(boundary):
            labels[i] = -100

        seq_len = len(full_ids)
        if seq_len < max_length:
            pad_len = max_length - seq_len
            full_ids = full_ids + [pad_id] * pad_len
            full_attn = full_attn + [0] * pad_len
            labels = labels + [-100] * pad_len
        elif seq_len > max_length:
            full_ids = full_ids[:max_length]
            full_attn = full_attn[:max_length]
            labels = labels[:max_length]

        all_input_ids.append(full_ids)
        all_attention_mask.append(full_attn)
        all_labels.append(labels)

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels,
    }


# -------------------------
# Main
# -------------------------

if __name__ == "__main__":
    seed = 42
    set_seed(seed)
    load_dotenv(os.path.expanduser("~/.env"), override=False)
    client = OpenAI()

    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if has_cuda:
        device = torch.device("cuda")
    elif has_mps:
        # MPS can be unstable with some Trainer/model combinations.
        device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    print(f"cuda available = {has_cuda}")
    print(f"mps available  = {has_mps}")
    print(f"using device   = {device}")

    # Prefer student-facing data/ paths, with fallback to cwd for convenience.
    sft_path = "./data/sft_dataset.json" if os.path.exists("./data/sft_dataset.json") else "sft_dataset.json"
    eval_path = (
        "./data/eval_dataset.json" if os.path.exists("./data/eval_dataset.json") else "eval_dataset.json"
    )

    sft_dataset = load_dataset("json", data_files=sft_path)["train"]
    eval_dataset = load_dataset("json", data_files=eval_path)["train"]

    # Use GPT-2 family to match the demo notebook style.
    model_name = "gpt2-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    base_model.config.pad_token_id = tokenizer.pad_token_id

    eval_dataset = eval_dataset.map(
        generate_responses,
        batched=True,
        batch_size=8,
        fn_kwargs={
            "field_name": "response_base",
            "model": base_model,
            "tokenizer": tokenizer,
            "device": device,
        },
    )

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id

    tokenized_sft = sft_dataset.shuffle(seed=seed).map(
        tokenize_with_prompt_masking,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
    )

    train_args = TrainingArguments(
        output_dir="./outputs/part2_sft",
        per_device_train_batch_size=4,
        num_train_epochs=1,
        learning_rate=1e-5,
        weight_decay=0.0,
        logging_steps=20,
        save_strategy="no",
        report_to=[],
        use_cpu=(not has_cuda),
        fp16=has_cuda,
        seed=seed,
        data_seed=seed,
    )

    model = model.to(device)
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_sft,
    )

    train_output = trainer.train()
    model = trainer.model
    print(f"train_loss = {float(train_output.training_loss):.6f}")

    eval_dataset = eval_dataset.map(
        generate_responses,
        batched=True,
        batch_size=8,
        fn_kwargs={
            "field_name": "response_sft",
            "model": model,
            "tokenizer": tokenizer,
            "device": device,
        },
    )

    base_vs_sft = evaluate_responses(client, eval_dataset, "response_sft", "response_base")
    print(f"Win rate (SFT over base): {base_vs_sft:.4f}")
