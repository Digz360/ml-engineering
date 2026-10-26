import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from model import MiniGPT


## VARIÁVEIS GLOBAIS ##

batch_size = 4
block_size = 512

n_embd = 256
num_heads = 8
num_layers = 8

dropout = 0.1

learning_rate = 3e-4


## GPU ##

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")


## CARREGAR TOKENIZER V4 ##

tokenizer = Tokenizer.from_file(
    "src/data/llm/tokenizer_v4.json"
)

vocab_size = tokenizer.get_vocab_size()

print(f"Vocab size: {vocab_size}")


## PALAVRA -> TOKEN ##

def encode(s):
    return tokenizer.encode(s).ids


## TOKEN -> PALAVRA ##

def decode(ids):
    return tokenizer.decode(ids)


## CARREGAR DATASET REAL ##

print("A carregar train...")

with open(
    "src/data/llm/pretrain_v4_train.txt",
    "r",
    encoding="utf-8"
) as f:
    train_text = f.read()


print("A carregar validation...")

with open(
    "src/data/llm/pretrain_v4_val.txt",
    "r",
    encoding="utf-8"
) as f:
    val_text = f.read()


## TOKENIZAR ##

print("A tokenizar train...")

train_data = torch.tensor(
    encode(train_text),
    dtype=torch.long
)

print("A tokenizar validation...")

val_data = torch.tensor(
    encode(val_text),
    dtype=torch.long
)


## ESTATÍSTICAS DO DATASET ##

print()
print("=" * 60)
print("DATASET")
print("=" * 60)

print(f"Train tokens: {len(train_data):,}")
print(f"Val tokens:   {len(val_data):,}")

tokens_per_step = batch_size * block_size

print(f"Tokens por step: {tokens_per_step:,}")

steps_per_epoch = len(train_data) / tokens_per_step

print(f"Steps ≈ 1 epoch: {steps_per_epoch:,.0f}")


## CRIAR BATCHES ##

def get_batch(data):

    ix = torch.randint(
        len(data) - block_size,
        (batch_size,)
    )

    x = torch.stack([
        data[i:i + block_size]
        for i in ix
    ])

    y = torch.stack([
        data[i + 1:i + block_size + 1]
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


## CARREGAR CHECKPOINT V4 ##

checkpoint = torch.load(
    "src/data/llm/minigpt_v4_base.pt",
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

optimizer.load_state_dict(
    checkpoint["optimizer_state_dict"]
)

best_val_loss = checkpoint["val_loss"]

print(
    f"Base V4 carregado | "
    f"val_loss = {best_val_loss:.4f}"
)

## ESTIMAR LOSS ##

@torch.no_grad()
def estimate_loss(model, eval_iters=50):

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


## PRIMEIRA AVALIAÇÃO ##

initial_losses = estimate_loss(model)

print()
print(
    f"Modelo NOVO | "
    f"train = {initial_losses['train']:.4f} | "
    f"val = {initial_losses['val']:.4f}"
)


## TREINO ##

max_steps = 21_200


for step in range(max_steps):

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

    if step % 50 == 0:

        losses = estimate_loss(model)

        print(
            f"Step {step}: "
            f"train = {losses['train']:.4f} | "
            f"val = {losses['val']:.4f}"
        )


        ## GUARDAR MELHOR MODELO ##

        if losses["val"] < best_val_loss:

            best_val_loss = losses["val"]

            torch.save({

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "vocab_size":
                    vocab_size,

                "n_embd":
                    n_embd,

                "num_heads":
                    num_heads,

                "num_layers":
                    num_layers,

                "block_size":
                    block_size,

                "dropout":
                    dropout,

                "val_loss":
                    losses["val"],

                "train_loss":
                    losses["train"]

            }, "src/data/llm/minigpt_v4_base.pt")


            print(
                f"Novo melhor modelo guardado! "
                f"val = {best_val_loss:.4f}"
            )