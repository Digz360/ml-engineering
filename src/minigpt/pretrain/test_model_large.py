import torch
from tokenizers import Tokenizer

from src.minigpt.model import MiniGPT

# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_PATH = "/workspace/llm/minigpt_large_v2_best.pt"
TOKENIZER_PATH = "/workspace/llm/tokenizer_large.json"

MAX_NEW_TOKENS = 200
TEMPERATURE = 0.8
TOP_K = 50


# ============================================================
# TOKENIZER
# ============================================================

print("A carregar tokenizer...")

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

print(f"Tokenizer carregado. Vocab: {tokenizer.get_vocab_size():,}")


# ============================================================
# CHECKPOINT
# ============================================================

print(f"A carregar checkpoint: {CHECKPOINT_PATH}")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
)

step = checkpoint.get(
    "cycle_step",
    checkpoint.get("step", "N/A")
)

print(f"Checkpoint step: {step}")
print(f"Best val loss: {checkpoint.get('best_val_loss', 'N/A')}")


# ============================================================
# MODEL
# ============================================================

VOCAB_SIZE = checkpoint["vocab_size"]
BLOCK_SIZE = checkpoint["block_size"]
N_EMBD = checkpoint["n_embd"]
NUM_HEADS = checkpoint["num_heads"]
NUM_LAYERS = checkpoint["num_layers"]
DROPOUT = checkpoint["dropout"]

model = MiniGPT(
    vocab_size=VOCAB_SIZE,
    n_embd=N_EMBD,
    num_heads=NUM_HEADS,
    block_size=BLOCK_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

num_params = sum(p.numel() for p in model.parameters())

print(f"Modelo carregado em: {DEVICE}")
print(f"Parâmetros: {num_params:,} ({num_params / 1e6:.2f}M)")


# ============================================================
# GENERATION
# ============================================================

torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

@torch.no_grad()
def generate(
    prompt,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    top_k=TOP_K,
):
    ids = tokenizer.encode(prompt).ids

    x = torch.tensor(
        [ids],
        dtype=torch.long,
        device=DEVICE,
    )

    for _ in range(max_new_tokens):

        # Só podemos dar ao Transformer até BLOCK_SIZE tokens.
        x_cond = x[:, -BLOCK_SIZE:]

        logits = model(x_cond)

        # Queremos apenas a previsão do próximo token.
        logits = logits[:, -1, :]

        # Temperature
        logits = logits / temperature

        # Top-k sampling
        if top_k is not None:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < values[:, [-1]]] = float("-inf")

        probs = torch.softmax(logits, dim=-1)

        next_token = torch.multinomial(
            probs,
            num_samples=1,
        )

        x = torch.cat((x, next_token), dim=1)

        # EOS = 2 no nosso tokenizer
        if next_token.item() == 2:
            break

    return tokenizer.decode(x[0].tolist())


# ============================================================
# INTERACTIVE TEST
# ============================================================

print("\n" + "=" * 70)
print("MINIGPT BASE LARGE")
print("Ainda NÃO foi feito SFT.")
print("Escreve 'quit' para sair.")
print("=" * 70)

while True:

    prompt = input("\nPrompt: ")

    if prompt.lower().strip() in {"quit", "exit"}:
        break

    if not prompt.strip():
        continue

    output = generate(prompt)

    print("\n--- OUTPUT ---")
    print(output)
    print("--------------")