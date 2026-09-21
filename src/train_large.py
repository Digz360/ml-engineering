import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from model import MiniGPT
import numpy as np
import os


# ============================================================
# CONFIG
# ============================================================

batch_size = 8
block_size = 1024

n_embd = 768
num_heads = 12
num_layers = 12

dropout = 0.1
learning_rate = 3e-4

eval_interval = 500
eval_iters = 20

TOKENIZER_FILE = "/workspace/llm/tokenizer_large.json"

TRAIN_FILE = "/workspace/llm/pretrain_large_train.txt"
VAL_FILE = "/workspace/llm/pretrain_large_val.txt"

LATEST_CHECKPOINT = "/workspace/llm/minigpt_large_latest.pt"
BEST_CHECKPOINT = "/workspace/llm/minigpt_large_best.pt"

## GPU ##

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")


## CARREGAR TOKENIZER LARGE ##

tokenizer = Tokenizer.from_file(
    TOKENIZER_FILE
)

vocab_size = tokenizer.get_vocab_size()

print(f"Vocab size: {vocab_size}")


## PALAVRA -> TOKEN ##

def encode(s):
    return tokenizer.encode(s).ids


## TOKEN -> PALAVRA ##

def decode(ids):
    return tokenizer.decode(ids)


## ============================================================
# DATASET TOKENIZADO
# ============================================================

TRAIN_BIN = "/workspace/llm/pretrain_large_train.bin"
VAL_BIN = "/workspace/llm/pretrain_large_val.bin"

print("A carregar datasets binários...")

train_data = np.memmap(
    TRAIN_BIN,
    dtype=np.uint16,
    mode="r"
)

val_data = np.memmap(
    VAL_BIN,
    dtype=np.uint16,
    mode="r"
)

print(f"Train tokens: {len(train_data):,}")
print(f"Val tokens:   {len(val_data):,}")

tokens_per_step = batch_size * block_size

print(f"Tokens por step: {tokens_per_step:,}")
print(
    f"Steps ≈ 1 passagem pelo dataset: "
    f"{len(train_data) / tokens_per_step:,.0f}"
)


## CRIAR BATCHES ##

def get_batch(data):
    
    ix = torch.randint(
        0,
        len(data) - block_size - 1,
        (batch_size,)
    )

    x = torch.stack([
        torch.from_numpy(
            np.array(
                data[i:i + block_size],
                dtype=np.int64
            )
        )
        for i in ix
    ])

    y = torch.stack([
        torch.from_numpy(
            np.array(
                data[i + 1:i + block_size + 1],
                dtype=np.int64
            )
        )
        for i in ix
    ])

    return x.to(device), y.to(device)


## CRIAR MODELO NOVO ##

model = MiniGPT(
    vocab_size=vocab_size,
    n_embd=n_embd,
    num_heads=num_heads,
    block_size=block_size,
    num_layers=num_layers,
    dropout=dropout
).to(device)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)


print()
print(
    f"Parâmetros: "
    f"{sum(p.numel() for p in model.parameters()):,}"
)


## CARREGAR CHECKPOINT LARGE ##

start_step = 0
best_val_loss = float("inf")

if os.path.exists(LATEST_CHECKPOINT):
    
    print("Checkpoint encontrado. A retomar treino...")

    checkpoint = torch.load(
        LATEST_CHECKPOINT,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    start_step = checkpoint["step"] + 1
    best_val_loss = checkpoint["best_val_loss"]

    print(
        f"Treino retomado no step {start_step:,} | "
        f"best val = {best_val_loss:.4f}"
    )

else:

    print("Nenhum checkpoint encontrado.")
    print("A iniciar modelo novo.")

## ESTIMAR LOSS ##

@torch.no_grad()
def estimate_loss(model, eval_iters=eval_iters):

    model.eval()

    results = {}

    for name, dataset in [
        ("train", train_data),
        ("val", val_data)
    ]:

        losses = []

        for _ in range(eval_iters):

            xb, yb = get_batch(dataset)

            logits = model(xb)

            B, T, V = logits.shape

            logits = logits.view(B * T, V)
            targets = yb.view(B * T)

            loss = F.cross_entropy(
                logits,
                targets
            )

            losses.append(loss.item())

        results[name] = (
            sum(losses) / len(losses)
        )

    model.train()

    return results


## TREINO ##

max_steps = 21_200


for step in range(start_step, max_steps):

    xb, yb = get_batch(train_data)

    # Forward
    logits = model(xb)

    B, T, V = logits.shape

    logits = logits.view(B * T, V)
    targets = yb.view(B * T)

    loss = F.cross_entropy(
        logits,
        targets
    )

    # Backprop
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


    ## AVALIAÇÃO ##

    if step % eval_interval  == 0:

        losses = estimate_loss(model)

        print(
            f"Step {step}: "
            f"train = {losses['train']:.4f} | "
            f"val = {losses['val']:.4f}"
        )


        # ============================================================
        # CHECKPOINTS
        # ============================================================

        is_best = losses["val"] < best_val_loss

        if is_best:
            best_val_loss = losses["val"]


        checkpoint_data = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),

            "step": step,
            "best_val_loss": best_val_loss,

            "vocab_size": vocab_size,
            "n_embd": n_embd,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "block_size": block_size,
            "dropout": dropout,

            "val_loss": losses["val"],
            "train_loss": losses["train"],
        }


        # Guardar sempre o estado mais recente
        torch.save(
            checkpoint_data,
            LATEST_CHECKPOINT
        )


        # Guardar separadamente se for o melhor modelo
        if is_best:

            torch.save(
                checkpoint_data,
                BEST_CHECKPOINT
            )

            print(
                f"Novo BEST guardado! "
                f"val = {best_val_loss:.4f}"
            )