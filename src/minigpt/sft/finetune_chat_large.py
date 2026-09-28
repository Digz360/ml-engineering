import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from src.minigpt.model import MiniGPT


# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 4
LEARNING_RATE = 3e-5

MAX_STEPS = 60_000
EVAL_INTERVAL = 250
EVAL_ITERS = 20
LOG_INTERVAL = 25

SEED = 42

BASE_CHECKPOINT = "/workspace/llm/minigpt_large_v2_best.pt"

LATEST_CHECKPOINT = "/workspace/llm/minigpt_large_v2_sft_latest.pt"
BEST_CHECKPOINT = "/workspace/llm/minigpt_large_v2_sft_best.pt"

TRAIN_FILE = "/root/llm-data/chat_large_curated.jsonl"
VAL_FILE = "/root/llm-data/chat_large_val.jsonl"

TOKENIZER_FILE = "/workspace/llm/tokenizer_large.json"



# ============================================================
# SEED / DEVICE
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")


# ============================================================
# TOKENIZER
# ============================================================

tokenizer = Tokenizer.from_file(TOKENIZER_FILE)

vocab_size = tokenizer.get_vocab_size()

pad_id = tokenizer.token_to_id("[PAD]")
eos_id = tokenizer.token_to_id("[EOS]")

print(
    f"Tokenizer carregado | "
    f"vocab = {vocab_size:,} | "
    f"PAD = {pad_id} | EOS = {eos_id}"
)


def encode(text):
    return tokenizer.encode(text).ids


def decode(ids):
    return tokenizer.decode(ids)


# ============================================================
# CARREGAR BASE MODEL
# ============================================================

base_checkpoint = torch.load(
    BASE_CHECKPOINT,
    map_location=device,
    weights_only=False
)

n_embd = base_checkpoint["n_embd"]
num_heads = base_checkpoint["num_heads"]
num_layers = base_checkpoint["num_layers"]
block_size = base_checkpoint["block_size"]
dropout = base_checkpoint["dropout"]

model = MiniGPT(
    vocab_size=vocab_size,
    n_embd=n_embd,
    num_heads=num_heads,
    block_size=block_size,
    num_layers=num_layers,
    dropout=dropout
).to(device)

model.load_state_dict(
    base_checkpoint["model_state_dict"]
)

print(
    f"Large Base carregado | "
    f"step = {base_checkpoint['cycle_step']:,} | "
    f"val_loss = {base_checkpoint['val_loss']:.4f}"
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# DATASET
# ============================================================

def load_conversations(path):
    conversations = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)

            messages = record.get("messages")

            if not messages:
                continue

            conversations.append(messages)

    return conversations


print("\nA carregar datasets...")

train_conversations = load_conversations(TRAIN_FILE)
val_conversations = load_conversations(VAL_FILE)

print(
    f"Conversas carregadas | "
    f"train = {len(train_conversations):,} | "
    f"val = {len(val_conversations):,}"
)


# ============================================================
# TOKENIZAR UMA MENSAGEM
# ============================================================

def encode_message(message):
    role = message["role"]
    content = message["content"]

    if role == "system":
        ids = encode(
            f"Sistema: {content}\n"
        )

        mask = [0] * len(ids)

        return ids, mask

    if role == "user":
        ids = encode(
            f"Utilizador: {content}\n"
        )

        mask = [0] * len(ids)

        return ids, mask

    if role == "assistant":
        prefix_ids = encode(
            "Assistente: "
        )

        content_ids = encode(content)

        newline_ids = encode("\n")

        ids = (
            prefix_ids
            + content_ids
            + [eos_id]
            + newline_ids
        )

        # Não treinamos no prefixo.
        # Treinamos conteúdo + EOS.
        # Newline é apenas separador/contexto.
        mask = (
            [0] * len(prefix_ids)
            + [1] * len(content_ids)
            + [1]
            + [0] * len(newline_ids)
        )

        return ids, mask

    return [], []


# ============================================================
# PREPARAR CONVERSA
#
# Regra:
# - nunca começamos um sample a meio de uma mensagem;
# - nunca cortamos uma mensagem normal ao meio;
# - mantemos contexto anterior enquanto couber;
# - só supervisionamos respostas do Assistant;
# - mensagens individuais > block_size são ignoradas.
# ============================================================

def prepare_conversation(conversation):

    encoded_messages = []

    for message in conversation:

        ids, mask = encode_message(message)

        if not ids:
            continue

        # Uma única mensagem maior do que a janela
        # não pode ser representada sem a cortar.
        if len(ids) > block_size:
            continue

        encoded_messages.append(
            (ids, mask)
        )

    samples = []

    # Cada resposta do assistant cria uma oportunidade
    # de treino.
    for assistant_index, (assistant_ids, assistant_mask) \
            in enumerate(encoded_messages):

        if sum(assistant_mask) == 0:
            continue

        selected = []
        total_length = 0

        # Construímos contexto de trás para a frente,
        # começando pela resposta atual.
        for index in range(
            assistant_index,
            -1,
            -1
        ):

            ids, mask = encoded_messages[index]

            if total_length + len(ids) > block_size + 1:
                break

            selected.append(
                (ids, mask, index)
            )

            total_length += len(ids)

        selected.reverse()

        all_ids = []
        loss_mask = []

        for ids, mask, index in selected:

            all_ids.extend(ids)

            # Só queremos calcular loss na resposta
            # do assistant que originou ESTE sample.
            #
            # Assistant anteriores servem apenas
            # como contexto.
            if index == assistant_index:
                loss_mask.extend(mask)
            else:
                loss_mask.extend(
                    [0] * len(mask)
                )

        if len(all_ids) < 2:
            continue

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

        if mask.sum().item() == 0:
            continue

        samples.append(
            (x, y, mask)
        )

    return samples


# ============================================================
# BATCH
# ============================================================

def get_batch(conversations):

    samples = []

    while len(samples) < BATCH_SIZE:

        conversation = random.choice(
            conversations
        )

        conversation_samples = prepare_conversation(
            conversation
        )

        if not conversation_samples:
            continue

        sample = random.choice(
            conversation_samples
        )

        samples.append(sample)

    max_len = max(
        len(x)
        for x, _, _ in samples
    )

    X = torch.full(
        (BATCH_SIZE, max_len),
        pad_id,
        dtype=torch.long
    )

    Y = torch.full(
        (BATCH_SIZE, max_len),
        pad_id,
        dtype=torch.long
    )

    masks = torch.zeros(
        (BATCH_SIZE, max_len),
        dtype=torch.float32
    )

    for i, (x, y, mask) in enumerate(samples):

        length = len(x)

        X[i, :length] = x
        Y[i, :length] = y
        masks[i, :length] = mask

    return (
        X.to(device),
        Y.to(device),
        masks.to(device)
    )


# ============================================================
# LOSS
# ============================================================

def calculate_loss(xb, yb, mask):

    logits = model(xb)

    B, T, V = logits.shape

    logits = logits.reshape(
        B * T,
        V
    )

    targets = yb.reshape(
        B * T
    )

    mask = mask.reshape(
        B * T
    )

    token_losses = F.cross_entropy(
        logits,
        targets,
        reduction="none"
    )

    supervised_tokens = mask.sum()

    loss = (
        (token_losses * mask).sum()
        / supervised_tokens
    )

    return loss


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def estimate_loss():

    model.eval()

    results = {}

    for name, conversations in [
        ("train", train_conversations),
        ("val", val_conversations)
    ]:

        losses = []

        for _ in range(EVAL_ITERS):

            xb, yb, mask = get_batch(
                conversations
            )

            loss = calculate_loss(
                xb,
                yb,
                mask
            )

            losses.append(
                loss.item()
            )

        results[name] = (
            sum(losses)
            / len(losses)
        )

    model.train()

    return results


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    step,
    train_loss,
    val_loss,
    best_val_loss
):

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "step":
                step,

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

            "train_loss":
                train_loss,

            "val_loss":
                val_loss,

            "best_val_loss":
                best_val_loss,
        },
        path
    )


# ============================================================
# RESUME SFT
# ============================================================

start_step = 0
best_val_loss = float("inf")

if Path(LATEST_CHECKPOINT).exists():

    print(
        "\nCheckpoint SFT encontrado. "
        "A retomar treino..."
    )

    sft_checkpoint = torch.load(
        LATEST_CHECKPOINT,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        sft_checkpoint[
            "model_state_dict"
        ]
    )

    optimizer.load_state_dict(
        sft_checkpoint[
            "optimizer_state_dict"
        ]
    )
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = LEARNING_RATE

    print(f"Learning rate atual: {optimizer.param_groups[0]['lr']}")

    start_step = (
        sft_checkpoint["step"] + 1
    )

    best_val_loss = (
        sft_checkpoint[
            "best_val_loss"
        ]
    )

    print(
        f"SFT retomado no step "
        f"{start_step:,}"
    )


# ============================================================
# SANITY CHECK
# ============================================================

print("\nSanity check...")

xb, yb, mask = get_batch(
    train_conversations
)

print(
    "Batch:",
    xb.shape
)

print(
    "Tokens supervisionados:",
    int(mask.sum().item())
)

print(
    "Contexto máximo:",
    xb.shape[1],
    "/",
    block_size
)


# ============================================================
# TRAINING
# ============================================================

print("\nA iniciar SFT...\n")

model.train()

for step in range(
    start_step,
    MAX_STEPS
):

    xb, yb, mask = get_batch(
        train_conversations
    )

    loss = calculate_loss(
        xb,
        yb,
        mask
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        1.0
    )

    optimizer.step()

    if step % LOG_INTERVAL == 0:

        print(
            f"Step {step:,} | "
            f"batch_loss = "
            f"{loss.item():.4f}"
        )

    if step % EVAL_INTERVAL == 0:

        losses = estimate_loss()

        train_loss = losses["train"]
        val_loss = losses["val"]

        print(
            "\n"
            f"Step {step:,} | "
            f"train = {train_loss:.4f} | "
            f"val = {val_loss:.4f}"
        )

        # BEST
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            save_checkpoint(
                BEST_CHECKPOINT,
                step,
                train_loss,
                val_loss,
                best_val_loss
            )

            print(
                f"Novo BEST SFT | "
                f"val = {best_val_loss:.4f}"
            )

        # LATEST
        save_checkpoint(
            LATEST_CHECKPOINT,
            step,
            train_loss,
            val_loss,
            best_val_loss
        )

        print(
            "LATEST SFT guardado\n"
        )


# ============================================================
# LOAD BEST
# ============================================================

print("\nTreino terminado.")

best_checkpoint = torch.load(
    BEST_CHECKPOINT,
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    best_checkpoint[
        "model_state_dict"
    ]
)

model.eval()

print(
    f"BEST carregado | "
    f"step = {best_checkpoint['step']:,} | "
    f"val = {best_checkpoint['val_loss']:.4f}"
)


# ============================================================
# GENERATION
# ============================================================

questions = [
    "Olá, como estás?",
    "O que é uma rede neuronal?",
    "Quanto é 7 + 5?",
    "O que é a memória RAM?",
    "Como posso organizar melhor o meu dia?",
    "Portugal fica em que continente?"
]


@torch.no_grad()
def answer(question, max_new_tokens=150):

    prompt = (
        f"Utilizador: {question}\n"
        f"Assistente: "
    )

    context = torch.tensor(
        [encode(prompt)],
        dtype=torch.long,
        device=device
    )

    generated = context.clone()

    for _ in range(max_new_tokens):

        input_tokens = (
            generated[:, -block_size:]
        )

        logits = model(
            input_tokens
        )

        next_logits = (
            logits[:, -1, :]
        )

        # Sampling ligeiramente conservador.
        temperature = 0.8

        next_logits = (
            next_logits / temperature
        )

        probs = F.softmax(
            next_logits,
            dim=-1
        )

        next_token = torch.multinomial(
            probs,
            num_samples=1
        )

        if next_token.item() == eos_id:
            break

        generated = torch.cat(
            [generated, next_token],
            dim=1
        )

    response_ids = generated[
        0,
        context.shape[1]:
    ].tolist()

    return decode(response_ids)


print("\n" + "=" * 70)
print("TESTE DO MINIGPT LARGE SFT")
print("=" * 70)

for question in questions:

    response = answer(question)

    print(
        f"\nUtilizador: {question}"
    )

    print(
        f"Assistente: {response}"
    )