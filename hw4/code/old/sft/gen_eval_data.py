#!/usr/bin/env python3
"""
Utility script: generate a small evaluation dataset for SFT judging.

This is a Python conversion of the original notebook helper.
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List

import torch
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def generate_gpt_response(
    client: OpenAI,
    prompt: str,
    messages: List[Dict[str, str]],
    model: str = "gpt-5-nano",
) -> List[Dict[str, str]]:
    messages = list(messages)
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    messages.append({"role": "assistant", "content": content})
    return messages


def self_instruct(client: OpenAI, dataset_list: List[Dict]) -> None:
    system_prompt = "You curate high-quality instruction-tuning datasets."
    messages = [{"role": "system", "content": system_prompt}]

    k = 32
    user_prompt = (
        f"Generate {k} new instruction examples as JSON list. "
        "Each example must contain instruction, instances, rating (1-5). "
        "Keep only high quality examples.\n"
    )
    for ex in random.sample(dataset_list, 8):
        user_prompt += json.dumps(
            {
                "instruction": ex["instruction"],
                "instances": [
                    {
                        "input": ex["instances"][0]["input"],
                        "output": ex["instances"][0]["output"],
                    }
                ],
            },
            ensure_ascii=False,
        )
        user_prompt += "\n"

    messages = generate_gpt_response(client, user_prompt, messages)
    raw = messages[-1]["content"].replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(raw)
        for ex in parsed:
            if int(ex.get("rating", 0)) >= 4:
                dataset_list.append(ex)
    except Exception:
        return


def generate_data(seed, num: int = 256) -> List[Dict]:
    # Load OPENAI_API_KEY from ~/.env explicitly.
    load_dotenv(os.path.expanduser("~/.env"), override=False)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Please define it in your environment or ~/.env."
        )

    client = OpenAI()
    dataset_list = seed.to_list()
    start = len(dataset_list)

    with tqdm(total=num, initial=0, desc="Generating eval candidates") as pbar:
        while len(dataset_list) - start < num:
            self_instruct(client, dataset_list)
            pbar.update(max(0, len(dataset_list) - start - pbar.n))

    return dataset_list[start : start + num]


def get_embedding(model, tokenizer, text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()


def list2dataset(dataset_list: List[Dict], threshold: float = 0.8) -> Dataset:
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    result = []
    embeddings = []
    for ex in dataset_list:
        inst = ex["instances"][0]
        if inst["input"] == "":
            prompt = ex["instruction"]
        else:
            prompt = f"{ex['instruction']}\\nInput: {inst['input']}"

        emb = get_embedding(model, tokenizer, prompt)
        is_new = True
        for emb_prev in embeddings:
            if cosine_similarity([emb], [emb_prev]) > threshold:
                is_new = False
                break

        if is_new:
            result.append({"prompt": prompt, "response": inst["output"]})
            embeddings.append(emb)

    return Dataset.from_list(result)


if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(repo_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "eval_dataset.json")

    seed = load_dataset("HuggingFaceH4/self-instruct-seed")["train"]
    generated = generate_data(seed, num=256)
    dataset = list2dataset(generated)
    dataset.to_json(out_path)
    print(f"Saved {out_path}")
