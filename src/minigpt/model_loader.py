from pathlib import Path

import torch
from tokenizers import Tokenizer

from src.minigpt.model import MiniGPT


def load_minigpt(
    checkpoint_path,
    tokenizer_path,
    device=None,
):
    """
    Load a trained MiniGPT checkpoint and tokenizer.
    """

    checkpoint_path = Path(checkpoint_path)
    tokenizer_path = Path(tokenizer_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {tokenizer_path}"
        )

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = torch.device(device)

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    # --------------------------------------------------------
    # Recover model configuration from checkpoint
    # --------------------------------------------------------

    vocab_size = checkpoint["vocab_size"]
    n_embd = checkpoint["n_embd"]
    num_heads = checkpoint["num_heads"]
    num_layers = checkpoint["num_layers"]
    block_size = checkpoint["block_size"]
    dropout = checkpoint["dropout"]

    # --------------------------------------------------------
    # Reconstruct architecture
    # --------------------------------------------------------

    model = MiniGPT(
        vocab_size=vocab_size,
        n_embd=n_embd,
        num_heads=num_heads,
        block_size=block_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    # --------------------------------------------------------
    # Load trained weights
    # --------------------------------------------------------

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    tokenizer = Tokenizer.from_file(
        str(tokenizer_path)
    )

    return model, tokenizer, checkpoint, device