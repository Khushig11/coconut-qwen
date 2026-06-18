# Coconut-Qwen

This project trains `Qwen/Qwen3-0.6B` on a single GSM8K math example, replaces part of the explicit chain-of-thought with latent reasoning tokens, and saves the training outputs automatically.

The main entry point is:

```bash
python final_train.py
```

Running this script handles the full experiment: model loading, tokenizer setup, latent-token preparation, single-example training, loss-curve plotting, answer generation, result saving, and optional checkpoint saving.

## Repository Structure

```text
coconut-qwen/
│
├── final_train.py              # Main script to run the full Coconut-Qwen training pipeline
├── requirements.txt            # Python dependencies
│
├── notebooks/
│   └── coconut_bottom_line.ipynb # Original notebook for bottom line task
│   └── coconut_ideal.ipynb # notebook for the implemntation of the ideal task
│
└── src/
    └── coconut_lr2/
        ├── __init__.py         # Package initializer
        ├── config.py           # Training configuration dataclass
        ├── data.py             # GSM8K example, prompt creation, and label masking
        ├── io_utils.py         # JSON saving and model/tokenizer checkpoint helpers
        ├── model.py            # CoconutLM wrapper around Qwen with latent-token logic
        ├── plotting.py         # Loss-curve plotting utility
        ├── train_loop.py       # Single-example training loop
        ├── trainer.py          # Full training pipeline orchestration
        └── utils.py            # Device selection, seed setting, and directory helpers
```

## Installation

Clone the repository:

```bash
git clone git@github.com:Khushig11/coconut-qwen.git
cd coconut-qwen
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Important: keep `transformers` below version 5 because this project uses the version range specified in `requirements.txt`.

## Running the Experiment

From the repository root, run:

```bash
python final_train.py
```

The script uses this default configuration:

```python
TrainConfig(
    model_name="Qwen/Qwen3-0.6B",
    k_latents=2,
    stage=1,
    steps=200,
    learning_rate=2e-5,
    output_dir="outputs",
    save_model=True,
)
```

## Output Files

After training, the project creates an `outputs/` folder containing:

```text
outputs/
├── training_config.json   # Saved training configuration
├── loss_curve.png         # Training loss plot
├── overfit_results.json   # Final loss, generated answer, losses, and metadata
└── overfit_model/         # Saved model and tokenizer checkpoint, if save_model=True
```


## Requirements

Main dependencies:

- PyTorch
- Transformers
- Accelerate
- Matplotlib
- tqdm
- safetensors

Install all required packages using:

```bash
pip install -r requirements.txt
```

