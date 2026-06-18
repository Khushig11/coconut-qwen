"""
Run this file from the repo root:

    python final_train.py

It executes the full modular Coconut pipeline and saves everything in outputs/.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from coconut_lr2.config import TrainConfig
from coconut_lr2.trainer import run_training_pipeline


def main() -> None:
    config = TrainConfig(
        model_name="Qwen/Qwen3-0.6B",
        k_latents=2,
        stage=1,
        steps=200,
        learning_rate=2e-5,
        output_dir="outputs",
        save_model=True,
    )

    run_training_pipeline(config)


if __name__ == "__main__":
    main()