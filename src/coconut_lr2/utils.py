from __future__ import annotations

import random
from pathlib import Path

import torch


def get_device(device_override: str | None = None) -> torch.device:
    if device_override:
        return torch.device(device_override)

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path