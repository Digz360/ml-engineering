import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from src.minigpt.model import MiniGPT


# ============================================================
# CONFIG
# ============================================================

CHECKPOINT_FILE = "/workspace/llm/minigpt_large_v2_sft_best.pt"
TOKENIZER_FILE = "/workspace/llm/tokenizer_large.json"

MAX_NEW_TOKENS = 150
TEMPERATURE = 0.8
TOP_K = 50


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")


# ============================================================
# TOKENIZER
# ============================================================

tokenizer = Tokenizer.from_file(TOKENIZER_FILE)

vocab_size = tokenizer.get_vocab_size()
eos_id = tokenizer.token_to_id("[EOS]")

print(f"Tokenizer carregado | vocab = {vocab_size:,}")


def encode(text):
    return tokenizer.encode(text).ids


def decode(ids):
    return tokenizer.decode(ids)


# ============================================================
# CHECKPOINT + MODEL
# ============================================================

print("\nA carregar checkpoint SFT...")

checkpoint = torch.load(
    CHECKPOINT_FILE,
    map_location=device,
    weights_only=False
)

model = MiniGPT(
    vocab_size=vocab_size,
    n_embd=checkpoint["n_embd"],
    num_heads=checkpoint["num_heads"],
    block_size=checkpoint["block_size"],
    num_layers=checkpoint["num_layers"],
    dropout=checkpoint["dropout"]
).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

block_size = checkpoint["block_size"]

print(
    f"SFT carregado | "
    f"step = {checkpoint.get('step', 'N/A')} | "
    f"val = {checkpoint.get('val_loss', 'N/A')}"
)


# ============================================================
# GENERATION
# ============================================================

@torch.no_grad()
def answer(
    question,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    top_k=TOP_K
):

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

        input_tokens = generated[:, -block_size:]

        logits = model(input_tokens)

        next_logits = logits[:, -1, :]

        # Temperature
        next_logits = next_logits / temperature

        # Top-k
        if top_k is not None:
            values, _ = torch.topk(
                next_logits,
                min(top_k, next_logits.size(-1))
            )

            next_logits[
                next_logits < values[:, [-1]]
            ] = float("-inf")

        probs = F.softmax(
            next_logits,
            dim=-1
        )

        next_token = torch.multinomial(
            probs,
            num_samples=1
        )

        # O modelo aprendeu a terminar respostas com EOS
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


# ============================================================
# INTERACTIVE CHAT
# ============================================================

print("\n" + "=" * 70)
print("MINIGPT LARGE SFT")
print("Escreve 'quit' para sair.")
print("=" * 70)

while True:

    question = input("\nTu: ").strip()

    if question.lower() in {"quit", "exit"}:
        print("Até logo!")
        break

    if not question:
        continue

    response = answer(question)

    print(f"\nMiniGPT: {response}")