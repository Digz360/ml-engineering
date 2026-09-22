import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from model import MiniGPT


## VARIÁVEIS GLOBAIS ##

batch_size = 16
block_size = 128

n_embd = 128
num_heads = 4
num_layers = 4

dropout = 0.1

## GPU ##

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")

## CARREGAR TOKENIZER E MODELO ##

checkpoint = torch.load("src/data/llm/minigpt_v2_base.pt")

tokenizer = Tokenizer.from_file(
    "src/data/llm/tokenizer_v2.json"
)
vocab_size = tokenizer.get_vocab_size()

## CRIAR TOKENS ## 

#tokenizer = Tokenizer(
#    BPE(unk_token="[UNK]")
#)

#tokenizer.pre_tokenizer = ByteLevel(
#    add_prefix_space=False
#)

#tokenizer.decoder = ByteLevelDecoder()

#trainer = BpeTrainer(
#    vocab_size=2000,
#    special_tokens=["[UNK]"],
#    initial_alphabet=ByteLevel.alphabet()
#)

#tokenizer.train(
#    [
#        "src/data/llm/pt.txt",
#        "src/data/llm/aritmetica.txt",
#        "src/data/llm/chat.txt"
#    ],
#    trainer
#)

#tokenizer.save(
#    "src/data/llm/tokenizer_v2.json"
#)


## PALAVRA -> TOKEN

def encode(s):
    return tokenizer.encode(s).ids

## TOKEN -> PALAVRA

def decode(ids):
    return tokenizer.decode(ids)


## USAR TEXTO E ARITMETICA PARA TREINAR ##

with open(
    "src/data/llm/pt.txt",
    "r",
    encoding="utf-8"
) as f:
    portuguese_text = f.read()

with open(
    "src/data/llm/aritmetica.txt",
    "r",
    encoding="utf-8"
) as f:
    math_text = f.read()


#Separar train de val

pt_data = torch.tensor(
    encode(portuguese_text),
    dtype=torch.long
)

math_data = torch.tensor(
    encode(math_text),
    dtype=torch.long
)

pt_n = int(0.9 * len(pt_data))
math_n = int(0.9 * len(math_data))

train_data = torch.cat([
    pt_data[:pt_n],
    math_data[:math_n]
])

val_data = torch.cat([
    pt_data[pt_n:],
    math_data[math_n:]
])

#Criar batches


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
        data[i + 1:i + block_size + 1] #shift right
        for i in ix
    ])

    return x.to(device), y.to(device)




## CARREGAR O MODELO E OPTIMIZER ##

model = MiniGPT(
    vocab_size=vocab_size,
    n_embd=n_embd,
    num_heads=num_heads,
    block_size=block_size,
    num_layers=num_layers,
    dropout=dropout
).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4
)

optimizer.load_state_dict(
    checkpoint["optimizer_state_dict"]
)

print(
    f"Parâmetros: "
    f"{sum(p.numel() for p in model.parameters()):,}"
)

## PARAR PARA AVALIAR A LOSS ##

@torch.no_grad()
def estimate_loss(model, eval_iters=100):
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

            loss = F.cross_entropy(logits, targets)

            losses.append(loss.item())

        results[name] = sum(losses) / len(losses)

    model.train()

    return results

best_val_loss = checkpoint["val_loss"] ## MELHOT LOSS GUARDADA

## TREINAR ##

for step in range(3000):
    
    xb, yb = get_batch(train_data)

    # Forward
    logits = model(xb)

    B, T, V = logits.shape
    logits = logits.view(B * T, V)
    targets = yb.view(B * T)

    loss = F.cross_entropy(logits, targets)

    # Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 250 == 0:
        losses = estimate_loss(model) ## PARAR PARA AVALIAR A LOSS 

        print(
            f"Step {step}: "
            f"train = {losses['train']:.4f} | "
            f"val = {losses['val']:.4f}"
        )

        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]

            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "vocab_size": vocab_size,
                "n_embd": n_embd,
                "num_heads": num_heads,
                "num_layers": num_layers,
                "block_size": block_size,
                "dropout": dropout,
                "val_loss": losses["val"],
                "train_loss": losses["train"]
            }, "src/data/llm/minigpt_v2_base.pt")

            print(f"Novo melhor modelo guardado! val = {best_val_loss:.4f}")



context = torch.tensor(
    [encode("Olá como estás neste dia?")],
    dtype=torch.long,
    device=device
)

generated = model.generate(
    context,
    max_new_tokens=2000
)

print(decode(generated[0].tolist()))