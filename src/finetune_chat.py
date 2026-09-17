import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from model import MiniGPT


## CONFIGURAÇÃO DO MODELO ##

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


## CARREGAR MODELO ##

checkpoint = torch.load(
    "src/data/llm/minigpt_v2_chat.pt",
    map_location=device
)

model = MiniGPT(
    vocab_size=checkpoint["vocab_size"],
    n_embd=checkpoint["n_embd"],
    num_heads=checkpoint["num_heads"],
    block_size=checkpoint["block_size"],
    num_layers=checkpoint["num_layers"],
    dropout=checkpoint["dropout"]
).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print(
    f"Chat carregado | "
    f"val_loss = {checkpoint['val_loss']:.4f}"
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)

best_val_loss = checkpoint["val_loss"]


## TOKENIZER ##

tokenizer = Tokenizer.from_file(
    "src/data/llm/tokenizer_v2.json"
)

vocab_size = tokenizer.get_vocab_size()

pad_id = tokenizer.token_to_id("[UNK]")


def encode(text):
    return tokenizer.encode(text).ids


def decode(ids):
    return tokenizer.decode(ids)


## DATASET CONVERSACIONAL ##

with open(
    "src/data/llm/chat.txt",
    "r",
    encoding="utf-8"
) as f:
    chat_text = f.read()


chat_data = torch.tensor(
    encode(chat_text),
    dtype=torch.long
)

conversations = [
    conversation.strip()
    for conversation in chat_text.split("\n\n")
    if conversation.strip()
]


## CRIAR AS CONVERSAS ##


def prepare_conversation(conversation):
    user_part, assistant_part = conversation.split("Assistente:", 1)

    context_text = user_part + "Assistente:"
    full_text = context_text + assistant_part

    context_ids = encode(context_text)
    full_ids = encode(full_text)

    loss_mask = (
        [0] * len(context_ids)
        + [1] * (len(full_ids) - len(context_ids))
    )

    x = torch.tensor(full_ids[:-1], dtype=torch.long)
    y = torch.tensor(full_ids[1:], dtype=torch.long)
    mask = torch.tensor(loss_mask[1:], dtype=torch.float32)

    return x, y, mask

x, y, mask = prepare_conversation(conversations[0])


## SPLIT DA DATA ## 

n = int(0.9 * len(conversations))

train_conversations = conversations[:n]
val_conversations = conversations[n:]


## DEFINIR BATCHES, AQUI ESTAMOS A RETIRAR BATCHES AO ACASO PELO INPUT E A ENCHER COM 0 AQUELES QUE TÊM ##
## MENOS TOKENS, PARA ALÉM DISSO TAMOS TAMBÉM A DIZER QUE ESSES NÃO INTERESSAM PARA A LOSS ##

def get_batch(conversations):
    indices = torch.randint(
        len(conversations),
        (batch_size,)
    )

    samples = [
        prepare_conversation(conversations[i])
        for i in indices
    ]

    max_len = max(len(x) for x, _, _ in samples)
    max_len = min(max_len, block_size)

    X = torch.full(
        (batch_size, max_len),
        pad_id,
        dtype=torch.long
    )

    Y = torch.full(
        (batch_size, max_len),
        pad_id,
        dtype=torch.long
    )

    masks = torch.zeros(
        (batch_size, max_len),
        dtype=torch.float32
    )

    for i, (x, y, mask) in enumerate(samples):
        length = min(len(x), max_len)

        X[i, :length] = x[:length]
        Y[i, :length] = y[:length]
        masks[i, :length] = mask[:length]

    return (
        X.to(device),
        Y.to(device),
        masks.to(device)
    )

## ESTIMAR A LOSS ## 

@torch.no_grad()
def estimate_loss(model, eval_iters=50):
    model.eval()

    results = {}

    for name, dataset in [
        ("train", train_conversations),
        ("val", val_conversations)
    ]:
        losses = []

        for _ in range(eval_iters):
            xb, yb, mask = get_batch(dataset)

            logits = model(xb)

            B, T, V = logits.shape

            logits = logits.view(B * T, V)
            targets = yb.view(B * T)
            mask = mask.view(B * T)

            token_losses = F.cross_entropy(
                logits,
                targets,
                reduction="none"
            )

            loss = (
                (token_losses * mask).sum()
                / mask.sum()
            )

            losses.append(loss.item())

        results[name] = sum(losses) / len(losses)

    model.train()

    return results


## TREINO + GUARDAR MELHOR CHAT ## 


for step in range(6000):

    xb, yb, mask = get_batch(train_conversations)

    # Forward
    logits = model(xb)

    B, T, V = logits.shape

    logits = logits.view(B * T, V)
    targets = yb.view(B * T)
    mask = mask.view(B * T)

    # Assistant-only loss
    token_losses = F.cross_entropy(
        logits,
        targets,
        reduction="none"
    )

    loss = (
        (token_losses * mask).sum()
        / mask.sum()
    )

    # Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Avaliação
    if step % 50 == 0:

        losses = estimate_loss(model)

        print(
            f"Step {step}: "
            f"train = {losses['train']:.4f} | "
            f"val = {losses['val']:.4f}"
        )

        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]

            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab_size": vocab_size,
                "n_embd": n_embd,
                "num_heads": num_heads,
                "num_layers": num_layers,
                "block_size": block_size,
                "dropout": dropout,
                "val_loss": losses["val"],
                "train_loss": losses["train"]
            }, "src/data/llm/minigpt_v2_chat.pt")

            print(
                f"Novo melhor CHAT guardado! "
                f"val = {best_val_loss:.4f}"
            )

## CARREGAR MELHOR MODELO DEPOIS DO TREINO ##

best_checkpoint = torch.load(
    "src/data/llm/minigpt_v2_chat.pt",
    map_location=device
)

model.load_state_dict(
    best_checkpoint["model_state_dict"]
)

model.eval()

print(
    f"\nMelhor modelo carregado | "
    f"val_loss = {best_checkpoint['val_loss']:.4f}"
)


## TESTAR RESPOSTAS ##

questions = [
    "Olá, como estás?",
    "O que é uma rede neuronal?",
    "Quanto é 7 + 5?",
    "Como posso organizar melhor o meu dia?",
    "Portugal fica em que continente?"
]

with torch.no_grad():

    for question in questions:

        prompt = f"Utilizador: {question} Assistente:"

        context = torch.tensor(
            [encode(prompt)],
            dtype=torch.long,
            device=device
        )

        generated = model.generate(
            context,
            max_new_tokens=80
        )

        text = decode(
            generated[0].tolist()
        )

        print("\n" + "=" * 60)
        print(text)
