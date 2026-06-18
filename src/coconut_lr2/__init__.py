"""Modular Coconut latent-reasoning training package."""

from .config import TrainConfig
from .data import (
    GSM8KExample,
    get_single_gsm8k_example,
    build_training_text,
    prepare_training_batch,
)
from .model import CoconutLM
from .trainer import run_training_pipeline

__all__ = [
    "TrainConfig",
    "GSM8KExample",
    "get_single_gsm8k_example",
    "build_training_text",
    "prepare_training_batch",
    "CoconutLM",
    "run_training_pipeline",
]