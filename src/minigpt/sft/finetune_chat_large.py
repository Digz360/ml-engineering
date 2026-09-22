import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from model import MiniGPT
import json


## CONFIGURAÇÃO DO MODELO ##

batch_size = 4
BASE_CHECKPOINT = "/workspace/llm/minigpt_large_best.pt"
TRAIN_FILE = "/workspace/llm/chat_large_train.jsonl"
VAL_FILE = "/workspace/llm/chat_large_val.jsonl"

## GPU ##

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")


## CARREGAR MODELO ##

checkpoint = torch.load(
    BASE_CHECKPOINT,
    map_location=device,
    weights_only=False
)

vocab_size = checkpoint["vocab_size"]
n_embd = checkpoint["n_embd"]
num_heads = checkpoint["num_heads"]
num_layers = checkpoint["num_layers"]
block_size = checkpoint["block_size"]
dropout = checkpoint["dropout"]

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

print(
    f"Large Base carregado | "
    f"step = {checkpoint['step']:,} | "
    f"val_loss = {checkpoint['val_loss']:.4f}"
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)

best_val_loss = float("inf")

## TOKENIZER ##

tokenizer = Tokenizer.from_file(
    "/workspace/llm/tokenizer_large.json"
)

vocab_size = tokenizer.get_vocab_size()

pad_id = tokenizer.token_to_id("[PAD]")
eos_id = tokenizer.token_to_id("[EOS]")

print(f"PAD = {pad_id} | EOS = {eos_id}")


def encode(text):
    return tokenizer.encode(text).ids


def decode(ids):
    return tokenizer.decode(ids)


## DATASET CONVERSACIONAL ##

def load_conversations(path):
    conversations = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)

            conversations.append(
                record["messages"]
            )

    return conversations


train_conversations = load_conversations(TRAIN_FILE)
val_conversations = load_conversations(VAL_FILE)

print(
    f"Conversas carregadas | "
    f"train = {len(train_conversations):,} | "
    f"val = {len(val_conversations):,}"
)


## CRIAR AS CONVERSAS ##


def prepare_conversation(conversation):
    all_ids = []
    loss_mask = []

    for message in conversation:
        role = message["role"]
        content = message["content"]

        if role == "system":
            text = f"Sistema: {content}\n"
            ids = encode(text)

            all_ids.extend(ids)
            loss_mask.extend([0] * len(ids))

        elif role == "user":
            text = f"Utilizador: {content}\n"
            ids = encode(text)

            all_ids.extend(ids)
            loss_mask.extend([0] * len(ids))

        elif role == "assistant":
            # O prefixo faz parte do contexto,
            # mas não entra na loss.
            prefix_ids = encode("Assistente: ")
            content_ids = encode(content + "\n")

            all_ids.extend(prefix_ids)
            loss_mask.extend([0] * len(prefix_ids))

            all_ids.extend(content_ids)
            loss_mask.extend([1] * len(content_ids))

    # Fim da conversa.
    all_ids.append(eos_id)

    # Queremos ensinar o modelo a terminar a resposta/conversa.
    loss_mask.append(1)

    x = torch.tensor(
        all_ids[:-1],
        dtype=torch.long
    )

    y = torch.tensor(
        all_ids[1:],
        dtype=torch.long
    )

    mask = torch.tensor(
        loss_mask[1:],
        dtype=torch.float32
    )

    return x, y, mask

x, y, mask = prepare_conversation(
    train_conversations[0]
)

print("Shape:", x.shape)
print("Tokens supervisionados:", int(mask.sum().item()))
print("Tokens totais:", len(mask))
print("Texto:")
print(decode(x.tolist()))

raise SystemExit

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


for step in range(1):

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
            }, "src/data/llm/minigpt_v3_chat.pt")

            print(
                f"Novo melhor CHAT guardado! "
                f"val = {best_val_loss:.4f}"
            )

## CARREGAR MELHOR MODELO DEPOIS DO TREINO ##

best_checkpoint = torch.load(
    "src/data/llm/minigpt_v4_chat.pt",
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
    "Quanto é 23 + 41?",
    "Quanto é 8 * 7?",
    "O que é a memória RAM?",
    "Como posso organizar melhor o meu dia?",
    "Como posso aprender uma coisa nova?",
    "Portugal fica em que continente?"
]

with torch.no_grad():

    for question in questions:

        prompt = f"Utilizador: {question}\nAssistente:"

        context = torch.tensor(
            [encode(prompt)],
            dtype=torch.long,
            device=device
        )

        generated = context.clone()

        # Máximo de 50 tokens por resposta
        for _ in range(50):

            # Só podemos dar ao Transformer até block_size tokens
            input_tokens = generated[:, -block_size:]

            logits = model(input_tokens)

            # Só interessa a previsão do próximo token
            next_logits = logits[:, -1, :]

            probs = F.softmax(
                next_logits,
                dim=-1
            )

            next_token = torch.multinomial(
                probs,
                num_samples=1
            )

            # Se o modelo disser EOS, acabou a resposta
            if next_token.item() == eos_id:
                break

            generated = torch.cat(
                [generated, next_token],
                dim=1
            )

        # Mostrar apenas a resposta,
        # sem repetir o prompt
        response_ids = generated[
            0,
            context.shape[1]:
        ].tolist()

        print("\n" + "=" * 60)
        print(f"PERGUNTA: {question}")
        print(f"RESPOSTA: {decode(response_ids)}")
