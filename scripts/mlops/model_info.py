from pathlib import Path
import json
import torch

MODEL_PATH = Path("models/minigpt_large_v2_sft_best.pt")
REPORT_PATH = Path("reports/model_info.json")

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False,
)

report = {
    "step": checkpoint["step"],
    "train_loss": checkpoint["train_loss"],
    "val_loss": checkpoint["val_loss"],
    "vocab_size": checkpoint["vocab_size"],
    "n_embd": checkpoint["n_embd"],
    "num_heads": checkpoint["num_heads"],
    "num_layers": checkpoint["num_layers"],
    "block_size": checkpoint["block_size"],
}

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(REPORT_PATH, "w") as f:
    json.dump(report, f, indent=2)

print("Model information generated successfully.")
print(json.dumps(report, indent=2))