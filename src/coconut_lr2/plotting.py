from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import matplotlib.pyplot as plt


def save_loss_curve(
    losses: Sequence[float],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4))
    plt.plot(losses)
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Single-example Coconut overfit loss")
    plt.grid(True)
    plt.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close()