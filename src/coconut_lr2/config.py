from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional
import json


@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen3-0.6B"
    trust_remote_code: bool = True

    k_latents: int = 2
    stage: int = 1

    steps: int = 200
    learning_rate: float = 2e-5
    max_grad_norm: float = 1.0
    log_every: int = 10
    seed: int = 42

    max_new_tokens: int = 80

    output_dir: str = "outputs"
    save_model: bool = True
    checkpoint_dir_name: str = "overfit_model"

    device: Optional[str] = None

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    @property
    def checkpoint_path(self) -> Path:
        return self.output_path / self.checkpoint_dir_name

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")