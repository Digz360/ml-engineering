import torch
from pathlib import Path

SOURCE = Path("models/minigpt_large_v2_sft_best.pt")
OUTPUT = Path("models/minigpt_v1_inference.pt")

print(f"Loading training checkpoint: {SOURCE}")

checkpoint = torch.load(
    SOURCE,
    map_location="cpu",
    weights_only=False,
)

inference_checkpoint = {
    "model_state_dict": checkpoint["model_state_dict"],

    # Architecture needed to reconstruct MiniGPT
    "vocab_size": checkpoint["vocab_size"],
    "n_embd": checkpoint["n_embd"],
    "num_heads": checkpoint["num_heads"],
    "num_layers": checkpoint["num_layers"],
    "block_size": checkpoint["block_size"],
    "dropout": checkpoint["dropout"],
}

torch.save(inference_checkpoint, OUTPUT)

training_size = SOURCE.stat().st_size / (1024 ** 2)
inference_size = OUTPUT.stat().st_size / (1024 ** 2)

print(f"Training checkpoint: {training_size:.2f} MB")
print(f"Inference artifact:  {inference_size:.2f} MB")
print(f"Reduction: {(1 - inference_size / training_size) * 100:.1f}%")
print(f"Saved to: {OUTPUT}")