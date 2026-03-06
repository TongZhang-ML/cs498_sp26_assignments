#!/usr/bin/env python3
"""
Part 1 (SFT): self-instruct style data generation.

This file is the Python version of the original notebook implementation.
Students should implement functions whose docstring starts with "Implement".
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List

from datasets import Dataset
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


def build_client() -> OpenAI:
    # Explicitly load OPENAI_API_KEY from ~/.env for local runs.
    load_dotenv(os.path.expanduser("~/.env"), override=False)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Please define it in your environment or ~/.env."
        )
    return OpenAI()


def generate_gpt_response(
    client: OpenAI,
    prompt: str,
    messages: List[Dict[str, str]],
    model: str = "gpt-5-nano",
) -> List[Dict[str, str]]:
    """Append a user prompt, query the model, and append assistant response."""
    messages = list(messages)
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    messages.append({"role": "assistant", "content": content})
    return messages


def _format_seed_examples(dataset_list: List[Dict], k: int = 8) -> str:
    picked = random.sample(dataset_list, k)
    lines = []
    for ex in picked:
        lines.append(
            json.dumps(
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
        )
    return "\n".join(lines)


def self_instruct(client: OpenAI, dataset_list: List[Dict]) -> None:
    """
    Implement according to the assignment description.

    Goal
    ----
    Run one simplified self-instruct iteration:
    1) Randomly sample 8 existing examples from dataset_list.
    2) Ask the model to generate 8 NEW instruction-tuning examples in JSON.
    3) Require each generated example to include an integer rating in [1, 5].
    4) Add only examples with rating >= 4 back into dataset_list.

    Input
    -----
    client:
        OpenAI client instance used for chat completion.
    dataset_list:
        A mutable list of instruction examples, where each entry has keys
        "instruction" and "instances" (self-instruct-seed format).

    Output
    ------
    None. Mutate dataset_list in-place.

    Notes
    -----
    - Use defensive parsing because model output may include markdown fences.
    - If parsing fails, silently skip this iteration (do not crash).
    """
    system_prompt = (
        "You act as a human curator generating high-quality instruction-tuning data."
    )
    messages = [{"role": "system", "content": system_prompt}]

    k = 32
    prompt = (
        f"Generate {k} new and diverse instruction-tuning examples as JSON list.\n"
        "Each item must contain fields: instruction, instances, rating.\n"
        "instances must have one item with keys input/output.\n"
        "rating must be integer from 1 to 5.\n"
        "Only include high-quality tasks.\n"
        "Here are seed examples:\n"
    )
    prompt += _format_seed_examples(dataset_list, k=8)

    messages = generate_gpt_response(client, prompt, messages, model="gpt-5-nano")
    raw = messages[-1]["content"].strip()
    cleaned = raw.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(cleaned)
        for ex in parsed:
            if int(ex.get("rating", 0)) >= 4:
                dataset_list.append(ex)
    except Exception:
        return


def generate_data(seed: Dataset, num: int, client: OpenAI) -> Dataset:
    """
    Implement according to the assignment description.

    Goal
    ----
    Build an instruction-following dataset of exactly `num` examples with fields:
      {"prompt": ..., "response": ...}

    Requirements
    ------------
    1) Start from `seed` (Hugging Face Dataset in self-instruct format).
    2) Repeatedly call `self_instruct(...)` to grow candidate examples.
    3) Convert newly added entries into prompt/response pairs:
       - If input is empty, prompt = instruction.
       - Else prompt = instruction + "\nInput: <input>".
       - response = output.
    4) Return a Dataset with exactly `num` rows.

    Returns
    -------
    datasets.Dataset
        Dataset with columns "prompt" and "response".
    """
    dataset_list = seed.to_list()
    original_size = len(dataset_list)
    result: List[Dict[str, str]] = []

    with tqdm(total=num, desc="Generating SFT data") as pbar:
        while len(result) < num:
            self_instruct(client, dataset_list)

            while original_size < len(dataset_list) and len(result) < num:
                ex = dataset_list[original_size]
                instance = ex["instances"][0]
                input_text = instance.get("input", "")
                if input_text == "":
                    prompt = ex["instruction"]
                else:
                    prompt = f"{ex['instruction']}\\nInput: {input_text}"

                result.append({"prompt": prompt, "response": instance["output"]})
                original_size += 1
                pbar.update(1)

    return Dataset.from_list(result[:num])


if __name__ == "__main__":
    from datasets import load_dataset

    seed = load_dataset("HuggingFaceH4/self-instruct-seed")["train"]
    client = build_client()
    dataset = generate_data(seed=seed, num=1024, client=client)
    os.makedirs("./data", exist_ok=True)
    out_path = "./data/sft_dataset.json"
    dataset.to_json(out_path)
    print(f"Saved {out_path}")
