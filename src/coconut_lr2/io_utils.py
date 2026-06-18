from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json(
    data: dict[str, Any],
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )


def save_model_and_tokenizer(
    coconut_model,
    save_dir: str | Path,
) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    coconut_model.model.save_pretrained(save_dir)
    coconut_model.tokenizer.save_pretrained(save_dir)