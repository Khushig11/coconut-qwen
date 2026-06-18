from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GSM8KExample:
    question: str
    cot_steps: list[str]
    answer: str


def get_single_gsm8k_example() -> GSM8KExample:
    return GSM8KExample(
        question=(
            "Natalia sold clips to 48 of her friends in April, and then she sold "
            "half as many clips in May. How many clips did Natalia sell altogether "
            "in April and May?"
        ),
        cot_steps=[
            "Natalia sold 48/2 = 24 clips in May.",
            "Natalia sold 48+24 = 72 clips altogether.",
        ],
        answer="72",
    )


def build_training_text(
    example: GSM8KExample,
    stage: int = 1,
    k_latents: int = 2,
) -> tuple[str, str]:
    removed = min(stage, len(example.cot_steps))
    remaining_steps = example.cot_steps[removed:]

    prompt = f"Question: {example.question}\nAnswer:"
    latent_block = " <bot>" + (" <latent>" * k_latents) + " <eot>" if k_latents > 0 else ""

    if remaining_steps:
        target = " " + " ".join(remaining_steps) + f" Therefore, the answer is {example.answer}."
    else:
        target = f" The answer is {example.answer}."

    return prompt, prompt + latent_block + target


def build_prefix_text(prompt: str, k_latents: int) -> str:
    return prompt + " <bot>" + (" <latent>" * k_latents) + " <eot>"


def prepare_training_batch(
    tokenizer,
    example: GSM8KExample,
    stage: int,
    k_latents: int,
    device: torch.device,
) -> dict:
    prompt_text, train_text = build_training_text(
        example,
        stage=stage,
        k_latents=k_latents,
    )

    tokenized = tokenizer(
        train_text,
        return_tensors="pt",
        add_special_tokens=False,
    )

    input_ids = tokenized["input_ids"].to(device)

    labels = input_ids.clone()
    labels[:] = -100

    prefix_text = build_prefix_text(prompt_text, k_latents)

    prefix_ids = tokenizer(
        prefix_text,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"]

    prefix_len = prefix_ids.shape[1]

    labels[:, prefix_len:] = input_ids[:, prefix_len:]

    return {
        "input_ids": input_ids,
        "labels": labels.to(device),
        "prompt_text": prompt_text,
        "train_text": train_text,
        "prefix_len": prefix_len,
        "total_tokens": input_ids.shape[1],
        "supervised_tokens": int((labels != -100).sum().item()),
    }