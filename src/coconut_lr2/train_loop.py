from __future__ import annotations

from collections.abc import Callable

import torch
from torch.optim import AdamW
from tqdm.auto import trange


def train_one_example(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    steps: int,
    learning_rate: float,
    max_grad_norm: float = 1.0,
    log_every: int = 10,
    log_fn: Callable[[str], None] = print,
) -> list[float]:
    model.train()

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
    )

    losses: list[float] = []

    for step in trange(steps, desc="Training"):
        optimizer.zero_grad(set_to_none=True)

        outputs = model(
            input_ids=input_ids,
            labels=labels,
        )

        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_grad_norm,
        )

        optimizer.step()

        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)

        if step % log_every == 0:
            log_fn(f"step={step} loss={loss_value:.4f}")

    log_fn(f"Final loss: {losses[-1]:.4f}")

    return losses