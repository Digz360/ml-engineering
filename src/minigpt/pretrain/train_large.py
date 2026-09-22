import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from src.minigpt.model import MiniGPT


# ============================================================
# CONFIG
# ============================================================

batch_size = 16
block_size = 1024

n_embd = 768
num_heads = 12
num_layers = 12

dropout = 0.1
learning_rate = 3e-4

eval_interval = 500
eval_iters = 20

# ~1 corpus-equivalente adicional:
# 1,807,519,192 / (16 * 1024) ~= 110,322
additional_steps = 110_322

TOKENIZER_FILE = "/workspace/llm/tokenizer_large.json"

TRAIN_BIN = "/root/llm-data/pretrain_large_train.bin"
VAL_BIN = "/root/llm-data/pretrain_large_val.bin"

BASE_CHECKPOINT = "/workspace/llm/minigpt_large_best.pt"

# Novos ficheiros para não destruir os checkpoints do primeiro ciclo.
LATEST_CHECKPOINT = "/workspace/llm/minigpt_large_cycle2_latest.pt"
BEST_CHECKPOINT = "/workspace/llm/minigpt_large_cycle2_best.pt"


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# TOKENIZER
# ============================================================

tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
vocab_size = tokenizer.get_vocab_size()

print(f"Vocab size: {vocab_size}")


# ============================================================
# DATASET
# ============================================================

print("A carregar datasets binários...")

train_data = np.memmap(
    TRAIN_BIN,
    dtype=np.uint16,
    mode="r",
)

val_data = np.memmap(
    VAL_BIN,
    dtype=np.uint16,
    mode="r",
)

print(f"Train tokens: {len(train_data):,}")
print(f"Val tokens:   {len(val_data):,}")

tokens_per_step = batch_size * block_size
target_tokens = additional_steps * tokens_per_step

print(f"Batch size: {batch_size}")
print(f"Tokens por step: {tokens_per_step:,}")
print(f"Steps adicionais: {additional_steps:,}")
print(f"Tokens alvo neste ciclo: {target_tokens:,}")


# ============================================================
# BATCHES
# ============================================================

def get_batch(data):

    ix = torch.randint(
        0,
        len(data) - block_size - 1,
        (batch_size,),
    )

    x = torch.stack([
        torch.from_numpy(
            np.array(
                data[i:i + block_size],
                dtype=np.int64,
            )
        )
        for i in ix
    ])

    y = torch.stack([
        torch.from_numpy(
            np.array(
                data[i + 1:i + block_size + 1],
                dtype=np.int64,
            )
        )
        for i in ix
    ])

    return x.to(device), y.to(device)


# ============================================================
# MODEL
# ============================================================

model = MiniGPT(
    vocab_size=vocab_size,
    n_embd=n_embd,
    num_heads=num_heads,
    block_size=block_size,
    num_layers=num_layers,
    dropout=dropout,
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
)

print(
    f"Parâmetros: "
    f"{sum(p.numel() for p in model.parameters()):,}"
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

cycle_step = 0
best_val_loss = float("inf")
original_step = None


# Se o ciclo 2 já começou anteriormente, retomamos daqui.
if os.path.exists(LATEST_CHECKPOINT):

    print()
    print("Checkpoint do CICLO 2 encontrado.")
    print("A retomar ciclo 2...")

    checkpoint = torch.load(
        LATEST_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    cycle_step = checkpoint["cycle_step"] + 1
    best_val_loss = checkpoint["best_val_loss"]
    original_step = checkpoint["original_step"]

    print(
        f"Ciclo 2 retomado no step {cycle_step:,} | "
        f"best val = {best_val_loss:.4f}"
    )


# Caso contrário, começamos o ciclo 2 a partir do BEST
# do primeiro treino.
else:

    print()
    print("A carregar BEST checkpoint do primeiro ciclo...")

    checkpoint = torch.load(
        BASE_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    original_step = checkpoint["step"]
    best_val_loss = checkpoint["best_val_loss"]

    print(
        f"Base carregada do step {original_step:,} | "
        f"best val = {best_val_loss:.4f}"
    )


# Garantir explicitamente o LR deste ciclo.
for param_group in optimizer.param_groups:
    param_group["lr"] = learning_rate

print(f"Learning rate: {learning_rate}")


# ============================================================
# ESTIMATE LOSS
# ============================================================

@torch.no_grad()
def estimate_loss():

    model.eval()

    results = {}

    for name, dataset in [
        ("train", train_data),
        ("val", val_data),
    ]:

        losses = []

        for _ in range(eval_iters):

            xb, yb = get_batch(dataset)

            logits = model(xb)

            B, T, V = logits.shape

            loss = F.cross_entropy(
                logits.reshape(B * T, V),
                yb.reshape(B * T),
            )

            losses.append(loss.item())

        results[name] = sum(losses) / len(losses)

    model.train()

    return results


# ============================================================
# TRAIN
# ============================================================

model.train()

training_start = time.perf_counter()

print()
print("=" * 70)
print("A INICIAR PRETRAINING — CICLO 2")
print("=" * 70)


for step in range(cycle_step, additional_steps):

    step_start = time.perf_counter()

    xb, yb = get_batch(train_data)

    logits = model(xb)

    B, T, V = logits.shape

    loss = F.cross_entropy(
        logits.reshape(B * T, V),
        yb.reshape(B * T),
    )

    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    optimizer.step()

    # ========================================================
    # LOG SIMPLES
    # ========================================================

    if step % 100 == 0:

        torch.cuda.synchronize()

        step_time = time.perf_counter() - step_start
        tokens_per_second = tokens_per_step / step_time

        print(
            f"Cycle step {step:,}/{additional_steps:,} | "
            f"loss {loss.item():.4f} | "
            f"{tokens_per_second:,.0f} tok/s"
        )


    # ========================================================
    # EVALUATION + CHECKPOINT
    # ========================================================

    if step % eval_interval == 0:

        losses = estimate_loss()

        print()
        print(
            f"[EVAL] Cycle step {step:,} | "
            f"train = {losses['train']:.4f} | "
            f"val = {losses['val']:.4f}"
        )

        is_best = losses["val"] < best_val_loss

        if is_best:
            best_val_loss = losses["val"]

        checkpoint_data = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),

            "cycle_step": step,
            "original_step": original_step,

            "best_val_loss": best_val_loss,

            "vocab_size": vocab_size,
            "n_embd": n_embd,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "block_size": block_size,
            "dropout": dropout,

            "batch_size": batch_size,
            "learning_rate": learning_rate,

            "val_loss": losses["val"],
            "train_loss": losses["train"],
        }

        torch.save(
            checkpoint_data,
            LATEST_CHECKPOINT,
        )

        print("LATEST guardado.")

        if is_best:

            torch.save(
                checkpoint_data,
                BEST_CHECKPOINT,
            )

            print(
                f"Novo BEST do ciclo 2! "
                f"val = {best_val_loss:.4f}"
            )

        print()


# ============================================================
# FINAL
# ============================================================

elapsed = time.perf_counter() - training_start

print()
print("=" * 70)
print("CICLO 2 TERMINADO")
print("=" * 70)

print(f"Tempo total: {elapsed / 3600:.2f} h")
print(f"Best val loss: {best_val_loss:.4f}")