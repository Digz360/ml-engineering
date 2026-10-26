from pathlib import Path

import mlflow

from src.minigpt.model_loader import load_minigpt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = PROJECT_ROOT / "models" / "minigpt_large_v2_sft_best.pt"
TOKENIZER_PATH = PROJECT_ROOT / "models" / "tokenizer_large.json"


# Load the real model checkpoint
model, tokenizer, checkpoint, device = load_minigpt(
    checkpoint_path=CHECKPOINT_PATH,
    tokenizer_path=TOKENIZER_PATH,
)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("MiniGPT")

with mlflow.start_run(run_name="minigpt_v2_sft_step_28500"):

    mlflow.log_params({
        "model_name": "MiniGPT",
        "model_version": "v2_sft",
        "vocab_size": checkpoint["vocab_size"],
        "n_embd": checkpoint["n_embd"],
        "num_layers": checkpoint["num_layers"],
        "num_heads": checkpoint["num_heads"],
        "block_size": checkpoint["block_size"],
        "num_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
        ),
        "checkpoint_step": checkpoint["step"],
        "device": str(device),
    })

    mlflow.log_metrics({
        "train_loss": checkpoint["train_loss"],
        "val_loss": checkpoint["val_loss"],
    })

    mlflow.pytorch.log_model(
        model,
        name="minigpt",
        serialization_format="pickle"

    )

    print("MLflow run completed successfully!")