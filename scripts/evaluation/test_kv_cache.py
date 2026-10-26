import torch

from src.minigpt.model_loader import load_minigpt


MODEL_PATH = "models/minigpt_large_v2_sft_best.pt"
TOKENIZER_PATH = "models/tokenizer_large.json"


model, tokenizer, checkpoint, device = load_minigpt(
    MODEL_PATH,
    TOKENIZER_PATH,
)

model.eval()

prompt = "Utilizador: O que é machine learning?\nAssistente: "

ids = tokenizer.encode(prompt).ids

context = torch.tensor(
    [ids],
    dtype=torch.long,
    device=device,
)


# -------------------------
# 1. PREFILL
# -------------------------

with torch.inference_mode():

    logits, cache = model(
        context,
        use_cache=True,
    )


print("\n--- PREFILL ---")

print("Input shape:", context.shape)
print("Number of layer caches:", len(cache))

k, v = cache[0]

print("Layer 0 K shape:", k.shape)
print("Layer 0 V shape:", v.shape)


# Escolher próximo token
next_logits = logits[:, -1, :] / 0.5
probs = torch.softmax(next_logits, dim=-1)
next_token = torch.multinomial(probs, 1)


# -------------------------
# 2. DECODE COM CACHE
# -------------------------

with torch.inference_mode():

    logits, cache = model(
        next_token,              # APENAS 1 TOKEN
        past_key_values=cache,
        use_cache=True,
    )


print("\n--- DECODE ---")

print("Input shape:", next_token.shape)

k, v = cache[0]

print("Layer 0 K shape:", k.shape)
print("Layer 0 V shape:", v.shape)

# -------------------------
# 3. COMPARAR COM FULL RECOMPUTE
# -------------------------

full_context = torch.cat(
    [context, next_token],
    dim=1,
)

with torch.inference_mode():
    full_logits = model(full_context)

full_last_logits = full_logits[:, -1:, :]

difference = (
    full_last_logits - logits
).abs().max().item()

print("\n--- CORRECTNESS CHECK ---")
print("Max difference:", difference)


import time

print("\n--- GENERATION BENCHMARK ---")

prompt = "Utilizador: Explica o que é machine learning.\nAssistente: "
ids = tokenizer.encode(prompt).ids

x = torch.tensor(
    [ids],
    dtype=torch.long,
    device=device,
)

start = time.perf_counter()

with torch.inference_mode():
    output = model.generate(
        x,
        max_new_tokens=100,
    )

elapsed = time.perf_counter() - start

print("Time:", round(elapsed, 2), "seconds")
print("Generated tokens:", output.shape[1] - x.shape[1])
print(
    "Tokens/sec:",
    round((output.shape[1] - x.shape[1]) / elapsed, 2)
)