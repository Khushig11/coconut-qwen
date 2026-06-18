from __future__ import annotations

from pathlib import Path

from .config import TrainConfig
from .data import get_single_gsm8k_example, prepare_training_batch
from .io_utils import save_json, save_model_and_tokenizer
from .model import CoconutLM
from .plotting import save_loss_curve
from .train_loop import train_one_example
from .utils import ensure_dir, get_device, set_seed


def run_training_pipeline(config: TrainConfig) -> dict:
    set_seed(config.seed)

    output_dir = ensure_dir(config.output_path)

    config.save(output_dir / "training_config.json")

    device = get_device(config.device)
    print(f"Using device: {device}")

    example = get_single_gsm8k_example()

    coconut = CoconutLM(
        model_name=config.model_name,
        trust_remote_code=config.trust_remote_code,
    ).to(device)

    print("Loaded model:", config.model_name)
    print("Special token ids:", coconut.bot_id, coconut.latent_id, coconut.eot_id)

    batch = prepare_training_batch(
        tokenizer=coconut.tokenizer,
        example=example,
        stage=config.stage,
        k_latents=config.k_latents,
        device=device,
    )

    print("Training text:\n", batch["train_text"])
    print("Total tokens:", batch["total_tokens"])
    print("Masked prefix tokens:", batch["prefix_len"])
    print("Supervised tokens:", batch["supervised_tokens"])

    losses = train_one_example(
        model=coconut,
        input_ids=batch["input_ids"],
        labels=batch["labels"],
        steps=config.steps,
        learning_rate=config.learning_rate,
        max_grad_norm=config.max_grad_norm,
        log_every=config.log_every,
    )

    save_loss_curve(
        losses,
        output_dir / "loss_curve.png",
    )

    generated = coconut.generate_with_latents(
        batch["prompt_text"],
        k_latents=config.k_latents,
        max_new_tokens=config.max_new_tokens,
    )

    print("Generated answer:\n", generated)

    results = {
        "model_name": config.model_name,
        "k_latents": config.k_latents,
        "stage": config.stage,
        "steps": config.steps,
        "learning_rate": config.learning_rate,
        "final_loss": losses[-1],
        "training_text": batch["train_text"],
        "ground_truth_answer": example.answer,
        "generated": generated,
        "losses": losses,
    }

    save_json(
        results,
        output_dir / "overfit_results.json",
    )

    if config.save_model:
        save_model_and_tokenizer(
            coconut,
            config.checkpoint_path,
        )
        print("Saved model checkpoint to:", config.checkpoint_path)

    print("Saved outputs to:", Path(output_dir).resolve())

    return results